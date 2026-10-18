//! Frozen Triton CompilationError source excerpt and caret rendering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompilationFailure {
    pub class: &'static str,
    pub message: String,
}
impl CompilationFailure {
    pub fn at(source: &str, definition: u32, position: (u32, u32), cause: &str) -> Self {
        let line = (position.0 - definition + 1) as usize;
        let lines = source.split('\n').take(line).collect::<Vec<_>>();
        let mut excerpt = lines[lines.len().saturating_sub(12)..].join("\n");
        if lines.is_empty() {
            excerpt = " <source empty>".into();
        } else {
            excerpt.push('\n');
            excerpt.push_str(&" ".repeat(position.1 as usize));
            excerpt.push('^');
        }
        let mut message = format!("at {line}:{}:\n{excerpt}", position.1);
        if !cause.is_empty() {
            message.push('\n');
            message.push_str(cause);
        }
        Self {
            class: "CompilationError",
            message,
        }
    }
}
