//! Small Markdown-to-terminal adapter for the Focus transcript.
//!
//! This is intentionally a parser-backed projection rather than a regular
//! expression formatter. It follows Codex CLI's separation of Markdown source
//! and terminal rendering (`codex-rs/tui/src/markdown.rs`, Apache-2.0).

use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct MarkdownStyle {
    pub(super) heading: Option<u8>,
    pub(super) strong: bool,
    pub(super) emphasis: bool,
    pub(super) code: bool,
    pub(super) link: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct MarkdownSpan {
    pub(super) text: String,
    pub(super) style: MarkdownStyle,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct MarkdownLine {
    pub(super) spans: Vec<MarkdownSpan>,
}

impl MarkdownLine {
    pub(super) fn plain_text(&self) -> String {
        self.spans.iter().map(|span| span.text.as_str()).collect()
    }

    fn push(&mut self, text: impl Into<String>, style: MarkdownStyle) {
        let text = text.into();
        if text.is_empty() {
            return;
        }
        if let Some(last) = self.spans.last_mut()
            && last.style == style
        {
            last.text.push_str(&text);
            return;
        }
        self.spans.push(MarkdownSpan { text, style });
    }

    fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }
}

#[derive(Default)]
struct RenderState {
    lines: Vec<MarkdownLine>,
    current: MarkdownLine,
    style: MarkdownStyle,
    lists: Vec<ListState>,
    link_destination: Option<String>,
}

#[derive(Clone, Copy)]
struct ListState {
    ordered: bool,
    next_index: u64,
}

impl RenderState {
    fn push_text(&mut self, text: &str) {
        for (index, part) in text.split('\n').enumerate() {
            if index > 0 {
                self.flush();
            }
            self.current.push(part, self.style);
        }
    }

    fn flush(&mut self) {
        if !self.current.is_empty() {
            self.lines.push(std::mem::take(&mut self.current));
        }
    }

    fn start_item(&mut self) {
        self.flush();
        let Some(list) = self.lists.last_mut() else {
            return;
        };
        let prefix = if list.ordered {
            let current = list.next_index;
            list.next_index += 1;
            format!("{current}. ")
        } else {
            "- ".into()
        };
        self.current.push(prefix, self.style);
    }
}

/// Parse supported Markdown into styled, display-width-bounded terminal lines.
pub(super) fn render_markdown(source: &str, width: usize) -> Vec<MarkdownLine> {
    let options =
        Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TABLES | Options::ENABLE_TASKLISTS;
    let mut state = RenderState::default();

    for event in Parser::new_ext(source, options) {
        match event {
            Event::Start(Tag::Paragraph) => state.flush(),
            Event::End(TagEnd::Paragraph) => state.flush(),
            Event::Start(Tag::Heading { level, .. }) => {
                state.flush();
                state.style.heading = Some(level as u8);
            }
            Event::End(TagEnd::Heading(_)) => {
                state.flush();
                state.style.heading = None;
            }
            Event::Start(Tag::Strong) => state.style.strong = true,
            Event::End(TagEnd::Strong) => state.style.strong = false,
            Event::Start(Tag::Emphasis) => state.style.emphasis = true,
            Event::End(TagEnd::Emphasis) => state.style.emphasis = false,
            Event::Start(Tag::Link { dest_url, .. }) => {
                state.style.link = true;
                state.link_destination = Some(dest_url.into_string());
            }
            Event::End(TagEnd::Link) => {
                state.style.link = false;
                if let Some(destination) = state.link_destination.take()
                    && !destination.is_empty()
                {
                    state.current.push(format!(" ({destination})"), state.style);
                }
            }
            Event::Start(Tag::List(start)) => state.lists.push(ListState {
                ordered: start.is_some(),
                next_index: start.unwrap_or(1),
            }),
            Event::End(TagEnd::List(_)) => {
                state.flush();
                state.lists.pop();
            }
            Event::Start(Tag::Item) => state.start_item(),
            Event::End(TagEnd::Item) => state.flush(),
            Event::Start(Tag::CodeBlock(_)) => {
                state.flush();
                state.style.code = true;
            }
            Event::End(TagEnd::CodeBlock) => {
                state.flush();
                state.style.code = false;
            }
            Event::Code(text) => {
                let mut style = state.style;
                style.code = true;
                state.current.push(text.into_string(), style);
            }
            Event::Text(text) | Event::Html(text) | Event::InlineHtml(text) => {
                state.push_text(&text);
            }
            Event::SoftBreak | Event::HardBreak => state.flush(),
            Event::Rule => {
                state.flush();
                state.current.push("---", state.style);
                state.flush();
            }
            Event::TaskListMarker(checked) => {
                state
                    .current
                    .push(if checked { "[x] " } else { "[ ] " }, state.style);
            }
            _ => {}
        }
    }
    state.flush();
    wrap_lines(state.lines, width)
}

fn wrap_lines(lines: Vec<MarkdownLine>, width: usize) -> Vec<MarkdownLine> {
    let width = width.max(1);
    let mut wrapped = Vec::new();
    for line in lines {
        let mut current = MarkdownLine::default();
        let mut current_width = 0;
        for span in line.spans {
            for grapheme in span.text.graphemes(true) {
                let grapheme_width = UnicodeWidthStr::width(grapheme);
                if !current.is_empty() && current_width + grapheme_width > width {
                    wrapped.push(std::mem::take(&mut current));
                    current_width = 0;
                }
                current.push(grapheme, span.style);
                current_width += grapheme_width;
            }
        }
        if !current.is_empty() {
            wrapped.push(current);
        }
    }
    wrapped
}

#[cfg(test)]
mod tests {
    use super::render_markdown;

    #[test]
    fn renders_heading_list_code_and_link_without_markers() {
        let lines = render_markdown(
            "# Title\n\n- **bold** and `code`\n\n[Focus](https://example.test)",
            80,
        );

        assert_eq!(lines[0].plain_text(), "Title");
        assert_eq!(lines[1].plain_text(), "- bold and code");
        assert_eq!(lines[2].plain_text(), "Focus (https://example.test)");
    }
}
