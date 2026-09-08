//! Bounded, prioritized context assembly.

use std::{collections::HashMap, fmt::Write as _};

use focus_kernel::{Message, Role};
use serde::{Deserialize, Serialize};

/// Origin and intended use of a context item.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextKind {
    /// Stable agent instructions.
    System,
    /// The currently requested task.
    Task,
    /// Repository facts discovered during exploration.
    Project,
    /// Durable or session-scoped memory.
    Memory,
    /// A compacted account of earlier transcript turns.
    Summary,
    /// Tool output that remains relevant to the task.
    ToolResult,
}

/// A context fragment before token-budget selection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextItem {
    /// Stable caller-assigned identity for de-duplication and incremental updates.
    pub id: String,
    /// Semantic source of this fragment.
    pub kind: ContextKind,
    /// Higher values win budget contention. System and task content should be highest.
    pub priority: u16,
    /// Human-readable provenance, surfaced in diagnostics only.
    pub source: String,
    /// Model-visible content.
    pub content: String,
}

impl ContextItem {
    /// A deliberately conservative token estimate usable without a tokenizer dependency.
    #[must_use]
    pub fn estimated_context_tokens(&self) -> usize {
        estimate_text_context_tokens(&self.content)
    }
}

/// The final bounded model context plus items excluded by budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextWindow {
    /// Selected items in priority order.
    pub items: Vec<ContextItem>,
    /// Input items that did not fit after compaction.
    pub omitted: Vec<ContextItem>,
    /// Local character-based estimate used for bounded context selection.
    pub estimated_context_tokens: usize,
    /// Maximum configured context estimate.
    pub context_budget: usize,
}

impl ContextWindow {
    /// Render one deterministic system message for provider adapters.
    #[must_use]
    pub fn render(&self) -> String {
        let mut rendered = String::new();
        for (index, item) in self.items.iter().enumerate() {
            if index > 0 {
                rendered.push_str("\n\n");
            }
            let _ = write!(
                rendered,
                "## {}: {}\n{}",
                item.kind.label(),
                item.source,
                item.content
            );
        }
        rendered
    }
}

impl ContextKind {
    fn label(&self) -> &'static str {
        match self {
            Self::System => "System",
            Self::Task => "Task",
            Self::Project => "Project",
            Self::Memory => "Memory",
            Self::Summary => "Summary",
            Self::ToolResult => "Tool result",
        }
    }
}

/// Incremental context builder with one canonical selection and compaction algorithm.
#[derive(Debug, Clone)]
pub struct ContextBuilder {
    budget: usize,
    compact_at: usize,
    items: HashMap<String, ContextItem>,
}

impl ContextBuilder {
    /// Create a builder using a model-context token budget.
    #[must_use]
    pub fn new(budget: usize) -> Self {
        Self {
            budget,
            compact_at: budget.saturating_mul(3) / 5,
            items: HashMap::new(),
        }
    }

    /// Upsert an item by ID; later observations supersede earlier ones.
    pub fn upsert(&mut self, item: ContextItem) {
        self.items.insert(item.id.clone(), item);
    }

    /// Build a bounded context, compacting only lower-priority verbose inputs.
    #[must_use]
    pub fn build(self) -> ContextWindow {
        let mut items: Vec<_> = self.items.into_values().collect();
        items.sort_by(|left, right| {
            right
                .priority
                .cmp(&left.priority)
                .then_with(|| left.id.cmp(&right.id))
        });
        let mut selected = Vec::new();
        let mut omitted = Vec::new();
        let mut used = 0;

        for item in items {
            let tokens = item.estimated_context_tokens();
            if tokens > self.compact_at && item.priority < 900 {
                let candidate = compact(item.clone(), self.compact_at);
                let candidate_tokens = candidate.estimated_context_tokens();
                if used + candidate_tokens <= self.budget {
                    used += candidate_tokens;
                    selected.push(candidate);
                } else {
                    omitted.push(item);
                }
            } else if used + tokens <= self.budget {
                used += tokens;
                selected.push(item);
            } else {
                omitted.push(item);
            }
        }
        ContextWindow {
            items: selected,
            omitted,
            estimated_context_tokens: used,
            context_budget: self.budget,
        }
    }
}

fn compact(mut item: ContextItem, max_tokens: usize) -> ContextItem {
    let max_chars = max_tokens.saturating_mul(4);
    let total = item.content.chars().count();
    if total <= max_chars {
        return item;
    }
    let head = max_chars.saturating_mul(3) / 4;
    let tail = max_chars.saturating_sub(head);
    let head_end = byte_index_at_char(&item.content, head);
    let tail_start = byte_index_at_char(&item.content, total - tail);
    item.kind = ContextKind::Summary;
    item.content = format!(
        "{}\n\n[... {} chars compacted ...]\n\n{}",
        &item.content[..head_end],
        total - head - tail,
        &item.content[tail_start..]
    );
    item
}

/// Build a bounded model view of an event-backed transcript without deleting its history.
///
/// The returned vector is intentionally ephemeral: JSONL remains the canonical replay
/// stream, while each provider request receives a fresh compacted view.
#[must_use]
pub fn compact_transcript(messages: &[Message], budget: usize) -> Vec<Message> {
    let total = messages
        .iter()
        .map(estimate_message_context_tokens)
        .sum::<usize>();
    if total <= budget {
        return messages.to_vec();
    }

    let summary_budget = budget.div_ceil(3).max(1);
    let recent_budget = budget.saturating_sub(summary_budget);
    let mut retained_reversed = Vec::new();
    let mut used = 0;
    for message in messages.iter().rev() {
        let tokens = estimate_message_context_tokens(message);
        if used + tokens > recent_budget && !retained_reversed.is_empty() {
            break;
        }
        used += tokens;
        retained_reversed.push(message.clone());
    }
    let retained_count = retained_reversed.len();
    retained_reversed.reverse();
    let omitted = &messages[..messages.len().saturating_sub(retained_count)];
    let max_chars = summary_budget.saturating_mul(4);
    let summary = bounded_transcript_summary(omitted, max_chars);
    let mut compacted = vec![Message::text(
        Role::System,
        format!(
            "Historical transcript summary ({} messages omitted):\n{summary}",
            omitted.len()
        ),
    )];
    compacted.extend(retained_reversed);
    compacted
}

fn bounded_transcript_summary(messages: &[Message], max_chars: usize) -> String {
    let mut summary = String::with_capacity(max_chars.min(4 * 1024));
    let mut remaining = max_chars;
    let mut truncated = false;

    for (index, message) in messages.iter().enumerate() {
        if index > 0 && !append_prefix(&mut summary, "\n", &mut remaining) {
            truncated = true;
            break;
        }
        if !append_prefix(&mut summary, role_prefix(message.role), &mut remaining)
            || !append_prefix(&mut summary, &message.content, &mut remaining)
        {
            truncated = true;
            break;
        }
    }
    if truncated {
        summary.push_str("\n[earlier transcript compacted]");
    }
    summary
}

fn role_prefix(role: Role) -> &'static str {
    match role {
        Role::System => "System: ",
        Role::User => "User: ",
        Role::Assistant => "Assistant: ",
        Role::Tool => "Tool: ",
    }
}

fn append_prefix(output: &mut String, source: &str, remaining: &mut usize) -> bool {
    let mut consumed = 0;
    let mut boundary = source.len();
    for (index, _) in source.char_indices() {
        if consumed == *remaining {
            boundary = index;
            break;
        }
        consumed += 1;
    }
    output.push_str(&source[..boundary]);
    *remaining -= consumed;
    boundary == source.len()
}

fn byte_index_at_char(text: &str, character_index: usize) -> usize {
    text.char_indices()
        .nth(character_index)
        .map_or(text.len(), |(byte_index, _)| byte_index)
}

fn estimate_text_context_tokens(content: &str) -> usize {
    content.chars().count().div_ceil(4).max(1)
}

fn estimate_message_context_tokens(message: &Message) -> usize {
    estimate_text_context_tokens(&message.content)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_token_estimate_is_shared_by_items_and_messages() {
        for content in ["", "a", "abcd", "abcde", "甲乙丙丁戊"] {
            let item = ContextItem {
                id: "item".into(),
                kind: ContextKind::Project,
                priority: 0,
                source: "test".into(),
                content: content.into(),
            };
            let message = Message::text(Role::User, content);
            assert_eq!(
                item.estimated_context_tokens(),
                estimate_message_context_tokens(&message)
            );
        }
    }

    #[test]
    fn append_prefix_stops_at_a_unicode_character_boundary() {
        let mut output = String::new();
        let mut remaining = 2;

        assert!(!append_prefix(&mut output, "甲乙丙", &mut remaining));
        assert_eq!(output, "甲乙");
        assert_eq!(remaining, 0);
    }

    #[test]
    fn short_historical_summary_does_not_reserve_the_full_limit() {
        let summary =
            bounded_transcript_summary(&[Message::text(Role::User, "short history")], 512 * 1024);

        assert_eq!(summary, "User: short history");
        assert!(summary.capacity() <= 4 * 1024);
    }

    #[test]
    fn keeps_high_priority_task_within_a_budget() {
        let mut builder = ContextBuilder::new(12);
        builder.upsert(ContextItem {
            id: "project".into(),
            kind: ContextKind::Project,
            priority: 100,
            source: "repo".into(),
            content: "x".repeat(80),
        });
        builder.upsert(ContextItem {
            id: "task".into(),
            kind: ContextKind::Task,
            priority: 1000,
            source: "user".into(),
            content: "fix tests".into(),
        });

        let window = builder.build();

        assert_eq!(window.items.first().unwrap().id, "task");
        assert!(window.estimated_context_tokens <= 12);
        assert_eq!(window.omitted.len(), 1);
    }

    #[test]
    fn renders_items_in_priority_then_id_order() {
        let mut builder = ContextBuilder::new(100);
        builder.upsert(ContextItem {
            id: "zeta".into(),
            kind: ContextKind::Project,
            priority: 10,
            source: "project".into(),
            content: "z".into(),
        });
        builder.upsert(ContextItem {
            id: "alpha".into(),
            kind: ContextKind::Task,
            priority: 10,
            source: "task".into(),
            content: "a".into(),
        });

        let window = builder.build();

        assert_eq!(
            window.render(),
            "## Task: task\na\n\n## Project: project\nz"
        );
    }

    #[test]
    fn compacts_history_without_discarding_the_latest_message() {
        let messages = vec![
            Message::text(Role::User, "a".repeat(80)),
            Message::text(Role::Assistant, "b".repeat(80)),
            Message::text(Role::User, "latest task"),
        ];

        let compacted = compact_transcript(&messages, 20);

        assert_eq!(compacted[0].role, Role::System);
        assert!(
            compacted[0]
                .content
                .contains("Historical transcript summary")
        );
        assert_eq!(compacted.last().unwrap().content, "latest task");
    }

    #[test]
    fn compaction_preserves_unicode_prefix_and_suffix() {
        let item = compact(
            ContextItem {
                id: "item".into(),
                kind: ContextKind::ToolResult,
                priority: 1,
                source: "test".into(),
                content: "甲乙丙丁戊己庚辛壬癸".into(),
            },
            2,
        );

        assert_eq!(
            item.content,
            "甲乙丙丁戊己\n\n[... 2 chars compacted ...]\n\n壬癸"
        );
    }

    #[test]
    fn compacted_history_keeps_the_legacy_prefix_and_marker() {
        let messages = vec![
            Message::text(Role::User, "甲乙丙丁戊己"),
            Message::text(Role::Assistant, "abcdef"),
            Message::text(Role::User, "latest"),
        ];

        let compacted = compact_transcript(&messages, 5);

        assert_eq!(
            compacted[0].content,
            "Historical transcript summary (2 messages omitted):\nUser: 甲乙\n[earlier transcript compacted]"
        );
        assert_eq!(compacted.last().unwrap().content, "latest");
    }
}
