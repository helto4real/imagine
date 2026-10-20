//! Original pure processErrorMsg behavior, without Python dependencies.
#[path = "python_error_types.rs"]
mod types;
pub(crate) fn translate(mut text: String) -> String {
    if text.contains("Type") {
        for (before, after) in types::TYPE_NAMES {
            text = text.replace(before, after);
        }
    }
    text
}
