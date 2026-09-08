//! Codex-style slash-command filtering and selection.
//!
//! Presentation ordering and exact-then-prefix filtering follow Codex CLI's
//! `bottom_pane/command_popup.rs` (Apache-2.0). Focus exposes only commands
//! backed by its own terminal host.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SlashCommand {
    name: &'static str,
    description: &'static str,
}

impl SlashCommand {
    pub(super) const fn name(self) -> &'static str {
        self.name
    }

    pub(super) fn localized_description(self, chinese: bool) -> &'static str {
        if chinese {
            return match self.name {
                "permissions" => "选择 Focus 权限模式",
                "new" => "开始新会话",
                "status" => "查看会话配置",
                "details" => "切换工具详情",
                "clear" => "清空当前界面",
                "help" => "显示可用命令",
                "copy" => "复制本轮回答",
                "language" => "切换界面语言",
                "review" => "审查当前仓库并修复真实问题",
                "simplify" => "在保持行为和性能的前提下简化",
                "exit" | "quit" => "退出 Focus",
                "update" => "切换到已验证版本",
                _ => self.description,
            };
        }
        self.description
    }
}

const COMMANDS: [SlashCommand; 13] = [
    SlashCommand {
        name: "permissions",
        description: "choose what Focus is allowed to do",
    },
    SlashCommand {
        name: "copy",
        description: "copy the current response",
    },
    SlashCommand {
        name: "language",
        description: "change interface language",
    },
    SlashCommand {
        name: "review",
        description: "review the repository and fix real issues",
    },
    SlashCommand {
        name: "simplify",
        description: "simplify code while preserving behavior and performance",
    },
    SlashCommand {
        name: "new",
        description: "start a new chat",
    },
    SlashCommand {
        name: "status",
        description: "show session configuration",
    },
    SlashCommand {
        name: "details",
        description: "toggle tool output details",
    },
    SlashCommand {
        name: "clear",
        description: "clear the visible transcript",
    },
    SlashCommand {
        name: "help",
        description: "show available commands",
    },
    SlashCommand {
        name: "exit",
        description: "exit Focus",
    },
    SlashCommand {
        name: "quit",
        description: "exit Focus",
    },
    SlashCommand {
        name: "update",
        description: "handoff to the verified Focus version",
    },
];

#[derive(Debug, Default)]
pub(super) struct SlashMenu {
    filter: String,
    selected: usize,
    dismissed: bool,
}

impl SlashMenu {
    pub(super) fn update(&mut self, composer: &str) {
        let filter = composer
            .strip_prefix('/')
            .filter(|value| !value.contains(char::is_whitespace))
            .unwrap_or_default();
        if self.filter != filter {
            self.filter = filter.into();
            self.selected = 0;
            self.dismissed = false;
        }
        self.clamp_selection();
    }

    pub(super) fn is_visible(&self, composer: &str) -> bool {
        composer.starts_with('/')
            && !composer.contains(char::is_whitespace)
            && !self.dismissed
            && !self.matches().is_empty()
    }

    pub(super) fn move_up(&mut self) {
        let length = self.matches().len();
        if length > 0 {
            self.selected = (self.selected + length - 1) % length;
        }
    }

    pub(super) fn move_down(&mut self) {
        let length = self.matches().len();
        if length > 0 {
            self.selected = (self.selected + 1) % length;
        }
    }

    pub(super) fn dismiss(&mut self) {
        self.dismissed = true;
    }

    pub(super) fn selected(&self) -> Option<SlashCommand> {
        self.matches().get(self.selected).copied()
    }

    pub(super) fn selected_name(&self) -> Option<&'static str> {
        self.selected().map(SlashCommand::name)
    }

    pub(super) fn complete(&self) -> Option<String> {
        self.selected()
            .map(|command| format!("/{} ", command.name()))
    }

    pub(super) fn matches(&self) -> Vec<SlashCommand> {
        let filter = self.filter.trim();
        let mut exact = Vec::new();
        let mut prefix = Vec::new();
        for command in COMMANDS {
            if filter.is_empty() || command.name.starts_with(filter) {
                if command.name == filter {
                    exact.push(command);
                } else {
                    prefix.push(command);
                }
            }
        }
        exact.extend(prefix);
        exact
    }

    fn clamp_selection(&mut self) {
        self.selected = self.selected.min(self.matches().len().saturating_sub(1));
    }
}

#[cfg(test)]
mod tests {
    use super::SlashMenu;

    #[test]
    fn filters_prefixes_and_tab_completes_the_selected_command() {
        let mut menu = SlashMenu::default();
        menu.update("/per");

        assert_eq!(menu.selected_name(), Some("permissions"));
        assert_eq!(menu.complete(), Some("/permissions ".into()));
    }

    #[test]
    fn exact_matches_sort_before_prefix_matches() {
        let mut menu = SlashMenu::default();
        menu.update("/exit");

        assert_eq!(menu.selected_name(), Some("exit"));
    }

    #[test]
    fn offers_review_and_simplify_commands_for_self_maintenance() {
        let mut menu = SlashMenu::default();

        menu.update("/review");
        assert_eq!(menu.selected_name(), Some("review"));

        menu.update("/simplify");
        assert_eq!(menu.selected_name(), Some("simplify"));
    }
}
