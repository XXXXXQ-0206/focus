//! Order-independent command-line token handling shared by Focus subcommands.

use std::collections::VecDeque;

/// Small parser for the CLI's existing option grammar.
pub(crate) struct Arguments {
    values: VecDeque<String>,
}

impl Arguments {
    pub(crate) fn new(values: Vec<String>) -> Self {
        Self {
            values: values.into(),
        }
    }

    pub(crate) fn next(&mut self) -> Option<String> {
        self.values.pop_front()
    }

    pub(crate) fn take_option(&mut self, option: &str) -> Result<Option<String>, String> {
        let found = self
            .values
            .iter()
            .take_while(|value| value.as_str() != "--")
            .position(|value| value == option);
        match found {
            None => Ok(None),
            Some(index) => {
                self.values.remove(index);
                self.values
                    .get(index)
                    .cloned()
                    .ok_or_else(|| format!("{option} requires a value"))
                    .map(|value| {
                        self.values.remove(index);
                        Some(value)
                    })
            }
        }
    }

    pub(crate) fn take_repeated_option(&mut self, option: &str) -> Result<Vec<String>, String> {
        let mut values = Vec::new();
        while let Some(value) = self.take_option(option)? {
            values.push(value);
        }
        Ok(values)
    }

    pub(crate) fn take_flag(&mut self, flag: &str) -> bool {
        let option_count = self
            .values
            .iter()
            .position(|value| value == "--")
            .unwrap_or(self.values.len());
        let mut found = false;
        let mut kept = VecDeque::with_capacity(self.values.len());
        for (index, value) in std::mem::take(&mut self.values).into_iter().enumerate() {
            if index < option_count && value == flag {
                found = true;
            } else {
                kept.push_back(value);
            }
        }
        self.values = kept;
        found
    }

    pub(crate) fn remaining_task(self) -> Result<String, String> {
        let unexpected = self
            .values
            .iter()
            .take_while(|value| value.as_str() != "--")
            .filter(|value| value.starts_with('-'))
            .cloned()
            .collect::<Vec<_>>();
        if !unexpected.is_empty() {
            return Err(format!("unexpected arguments: {}", unexpected.join(" ")));
        }
        let task = self
            .values
            .into_iter()
            .filter(|value| value != "--")
            .collect::<Vec<_>>();
        if task.is_empty() {
            Err("a task is required after options".into())
        } else {
            Ok(task.join(" "))
        }
    }

    pub(crate) fn ensure_empty(&self) -> Result<(), String> {
        if self.values.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "unexpected arguments: {}",
                self.values.iter().cloned().collect::<Vec<_>>().join(" ")
            ))
        }
    }
}
