//! Desk and Timeline commands: each action's name, icon, shortcut and
//! availability are defined once and shared by its button and key binding.
use super::*;
use crate::icons::Icon;
use std::borrow::Cow;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Command {
    TogglePrivacy,
    ToggleActualPixels,
    Generate,
    MultiplePrompts,
    Upscale,
    EditImage,
    Animate,
    Cancel,
    TimelineUndo,
    TimelineRedo,
    TimelineDelete,
    TimelineSave,
    TimelineGenerate,
}

/// Key bindings; the first is shown in tooltips.
pub(super) struct Shortcut {
    keys: &'static [(egui::Key, egui::Modifiers)],
    /// Also active while a text field has focus.
    pub while_typing: bool,
}
const CTRL: egui::Modifiers = egui::Modifiers::CTRL;
const CTRL_SHIFT: egui::Modifiers = egui::Modifiers {
    shift: true,
    ..egui::Modifiers::CTRL
};
const NONE: egui::Modifiers = egui::Modifiers::NONE;

impl Shortcut {
    pub fn label(&self) -> String {
        let (key, modifiers) = self.keys[0];
        let mut label = String::new();
        if modifiers.ctrl {
            label += "Ctrl+";
        }
        if modifiers.shift {
            label += "Shift+";
        }
        label + key.name()
    }
    /// Consumes this binding's key press, ignoring auto-repeat.
    pub fn pressed(&self, ctx: &egui::Context) -> bool {
        ctx.input_mut(|input| {
            let mut pressed = false;
            input.events.retain(|event| {
                if let egui::Event::Key {
                    key,
                    modifiers,
                    pressed: true,
                    repeat,
                    ..
                } = event
                    && self.keys.iter().any(|(k, m)| {
                        k == key
                            && m.ctrl == modifiers.ctrl
                            && m.shift == modifiers.shift
                            && !modifiers.alt
                            && !modifiers.mac_cmd
                    })
                {
                    pressed |= !repeat;
                    return false;
                }
                true
            });
            pressed
        })
    }
}

impl Command {
    pub fn shortcut(self) -> Option<Shortcut> {
        let (keys, while_typing): (&'static [_], _) = match self {
            Self::TogglePrivacy => (&[(egui::Key::Space, CTRL)], false),
            Self::ToggleActualPixels => (&[(egui::Key::Z, NONE)], false),
            Self::Generate => (&[(egui::Key::Enter, CTRL)], false),
            Self::TimelineUndo => (&[(egui::Key::Z, CTRL)], false),
            Self::TimelineRedo => (&[(egui::Key::Z, CTRL_SHIFT), (egui::Key::Y, CTRL)], false),
            Self::TimelineDelete => (
                &[(egui::Key::Delete, NONE), (egui::Key::Backspace, NONE)],
                false,
            ),
            // Saving while typing keeps the pending text drafts.
            Self::TimelineSave => (&[(egui::Key::S, CTRL)], true),
            _ => return None,
        };
        Some(Shortcut { keys, while_typing })
    }
    pub fn icon(self) -> Icon {
        match self {
            Self::TogglePrivacy => Icon::EyeOff,
            Self::ToggleActualPixels => Icon::Actual,
            Self::Generate | Self::TimelineGenerate => Icon::Generate,
            Self::MultiplePrompts => Icon::Duplicate,
            Self::Upscale => Icon::Upscale,
            Self::EditImage => Icon::Edit,
            Self::Animate => Icon::Animate,
            Self::Cancel => Icon::Stop,
            Self::TimelineUndo => Icon::Undo,
            Self::TimelineRedo => Icon::Redo,
            Self::TimelineDelete => Icon::Trash,
            Self::TimelineSave => Icon::Save,
        }
    }
    fn primary(self) -> bool {
        matches!(self, Self::Generate | Self::TimelineGenerate)
    }
}

impl ImagineApp {
    pub(super) fn command_label(&self, command: Command) -> &'static str {
        match command {
            Command::TogglePrivacy => "Hide previews",
            Command::ToggleActualPixels => "Actual pixels",
            Command::Generate => match self.workspace {
                Workspace::Image => "Add to queue",
                Workspace::ImageEdit => "Create version",
                _ => "Generate",
            },
            Command::MultiplePrompts => "Multiple prompts",
            Command::Upscale => "Upscale",
            Command::EditImage => "Edit image",
            Command::Animate => "Animate",
            Command::Cancel => "Cancel generation",
            Command::TimelineUndo => "Undo",
            Command::TimelineRedo => "Redo",
            Command::TimelineDelete => "Delete selection",
            Command::TimelineSave => "Save project",
            Command::TimelineGenerate => {
                if self.timeline_submission_pending() {
                    "Retry submission"
                } else {
                    "Generate"
                }
            }
        }
    }
    /// Why `command` cannot run now; None when it can.
    pub(super) fn command_blocked(&self, command: Command) -> Option<Cow<'static, str>> {
        let video = self.workspace == Workspace::Video;
        let reason: Option<&'static str> = match command {
            Command::TogglePrivacy => None,
            Command::ToggleActualPixels => (video || self.desk().result.is_none())
                .then_some("Generate or choose an image first."),
            Command::Generate => self.desk_blocked().or(self.input_blocked()),
            Command::MultiplePrompts => self.desk_blocked(),
            Command::Upscale => self.desk_blocked().or(self
                .upscale_source()
                .is_none()
                .then_some("Generate or choose an image first.")),
            Command::EditImage => self.edit_image_blocked(),
            Command::Animate => self.animate_image_blocked(),
            Command::Cancel => {
                (!self.viewer_progress().can_cancel).then_some("Nothing is running.")
            }
            Command::TimelineUndo
            | Command::TimelineRedo
            | Command::TimelineDelete
            | Command::TimelineSave => self.timeline_command_blocked(command),
            Command::TimelineGenerate => return self.timeline_generate_blocked().map(Cow::Owned),
        };
        reason.map(Cow::Borrowed)
    }
    pub(super) fn run_command(&mut self, ctx: &egui::Context, command: Command) {
        if self.command_blocked(command).is_some() {
            return;
        }
        match command {
            Command::TogglePrivacy => {
                self.prefs.privacy_mode = !self.prefs.privacy_mode;
                self.save_preferences();
            }
            Command::ToggleActualPixels => {
                let desk = self.desk_mut();
                desk.actual_pixels = !desk.actual_pixels;
                desk.image_pan = egui::Vec2::ZERO;
            }
            Command::Generate => self.generate(ctx),
            Command::MultiplePrompts => {
                let restored = self.queue.parked_drafts.take();
                self.queue.drafts = Some(
                    restored.unwrap_or_else(|| vec![self.image.prompt.clone(), String::new()]),
                );
                self.queue.draft_error = None;
            }
            Command::Upscale => self.upscale(ctx),
            Command::EditImage => self.start_editing(),
            Command::Animate => self.start_animating(ctx),
            Command::Cancel => {
                if let Some((id, _)) = &self.viewer_progress().queue_job {
                    let id = id.clone();
                    self.cancel_queue_job(ctx, &id);
                } else {
                    self.cancel(ctx);
                }
            }
            Command::TimelineUndo
            | Command::TimelineRedo
            | Command::TimelineDelete
            | Command::TimelineSave
            | Command::TimelineGenerate => self.timeline_run_command(ctx, command),
        }
    }
    /// The command's icon button: name, shortcut and any disabled reason on hover.
    pub(super) fn command_button(&mut self, ui: &mut egui::Ui, command: Command) -> egui::Response {
        let label = self.command_label(command);
        let blocked = self.command_blocked(command);
        let icon = match command {
            Command::TogglePrivacy if !self.prefs.privacy_mode => Icon::Eye,
            Command::TimelineGenerate if self.timeline_submission_pending() => Icon::Retry,
            _ => command.icon(),
        };
        let response = helto::action(
            ui,
            icon,
            label,
            command.primary() || self.command_active(command),
            blocked.as_deref(),
        );
        let response = match command.shortcut() {
            Some(shortcut) => response.on_hover_text(shortcut.label()),
            None => response,
        };
        if response.clicked() {
            self.run_command(ui.ctx(), command);
        }
        response
    }
    /// Whether a toggle command is currently on.
    fn command_active(&self, command: Command) -> bool {
        match command {
            Command::TogglePrivacy => self.prefs.privacy_mode,
            Command::ToggleActualPixels => self.desk().actual_pixels,
            _ => false,
        }
    }
    /// Runs the first of `commands` whose shortcut was pressed in a focused window.
    pub(super) fn command_shortcuts(&mut self, ctx: &egui::Context, commands: &[Command]) {
        if !ctx.input(|input| input.focused) {
            return;
        }
        let typing = ctx.text_edit_focused();
        for &command in commands {
            if let Some(shortcut) = command.shortcut()
                && (shortcut.while_typing || !typing)
                && shortcut.pressed(ctx)
            {
                self.run_command(ctx, command);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn key(key: egui::Key, modifiers: egui::Modifiers) -> egui::Event {
        egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers,
        }
    }
    fn pressed(command: Command, event: egui::Event) -> bool {
        let ctx = egui::Context::default();
        let mut hit = false;
        let mut output = ctx.run_ui(
            egui::RawInput {
                events: vec![event],
                ..Default::default()
            },
            |ui| hit = command.shortcut().unwrap().pressed(ui.ctx()),
        );
        output.textures_delta.clear();
        hit
    }
    #[test]
    fn shortcuts_match_exact_modifiers_and_name_their_first_binding() {
        assert!(pressed(Command::TimelineUndo, key(egui::Key::Z, CTRL)));
        assert!(!pressed(
            Command::TimelineUndo,
            key(egui::Key::Z, CTRL_SHIFT)
        ));
        assert!(pressed(
            Command::TimelineRedo,
            key(egui::Key::Z, CTRL_SHIFT)
        ));
        assert!(pressed(Command::TimelineRedo, key(egui::Key::Y, CTRL)));
        assert!(!pressed(
            Command::ToggleActualPixels,
            key(egui::Key::Z, CTRL)
        ));
        assert_eq!(
            Command::TimelineRedo.shortcut().unwrap().label(),
            "Ctrl+Shift+Z"
        );
        assert_eq!(Command::Generate.shortcut().unwrap().label(), "Ctrl+Enter");
        assert!(Command::TimelineSave.shortcut().unwrap().while_typing);
        assert!(!Command::TimelineDelete.shortcut().unwrap().while_typing);
    }
}
