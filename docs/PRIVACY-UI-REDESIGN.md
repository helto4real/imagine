# Hide previews: canvas-based privacy

Status: **Implemented 2026-10-25.**

Date: 2026-10-25.

Hide previews today hides text and images widget by widget: each text field,
image and thumbnail reveals only while the pointer is over that one widget, and
whole windows (LoRAs, presets, models, generation settings, media health,
compatibility) are blanked even though they show only settings. This design
narrows what is hidden and makes the reveal follow the panel or window the
content sits on. The queue dock already works this way (UI.md, "one privacy
boundary"); this extends that rule to every surface.

## 1. What is private

Only these stay hidden while Hide previews is on:

| Kind | Examples |
| --- | --- |
| Images and video frames | Results, references, thumbnails, Take cards, the video player and comparison, sampling previews, cast and library images, character sheets, media bin and picker thumbnails |
| Creative writing | Image prompt and negative prompt, scene and section prompts, scene descriptions, global and project prompt, story text, dialogue, character descriptions, enhancement and authoring output, compiled plan text, prompt tool values, queued and captured prompts |
| Scene titles | Timeline bars, inspector header, story and authoring views, the first-scene title in Open private project |
| Character names | Cast and library names and aliases |
| Media file names | Media bin, pickers, Check Media / Relink, binding lists |

Everything else is always visible: models, LoRAs, presets, components, sampling
and all other settings, project folder names, file paths of models, job states,
counts, durations and error messages (error codes are mapped to fixed text; an
error that quotes creative writing is treated as creative writing).

Audio waveforms are not images and stay visible.

## 2. Canvases

A canvas is a framed surface the user sees as one area:

- **Main window panels:** every `workspace::panel` / `chrome::panel` rect — on
  the image desk the references, viewer, prompt inspector, edit viewer and
  inspector, queue dock and settings body; in Video the toolbar, viewer, Take
  strip, inspector, references bin, timeline tracks, story and authoring panels.
- **Floating windows and modals:** each `egui::Window` or `egui::Modal` is one
  canvas (Cast, Library, Library character, Media picker, Prompt tools, Project
  prompt, Plan Review, Final video, Sampling preview, Open private project, Check
  Media, Multiple prompts, and so on).
- **Popups, menus and tooltips** belong to the canvas that opened them. While the
  pointer is over one of them, its canvas counts as hovered.

### Reveal rule

All private content on a canvas is revealed together while the pointer is
anywhere inside that canvas (including its popups) **and** Imagine's window has
focus. Leaving the canvas or losing window focus hides all of it in the same
frame. Other canvases stay hidden. Hovering a single field or image is no longer
needed and no longer reveals anything by itself.

A field being typed in stays revealed while it keeps focus, as today, so text
does not vanish mid-edit when the pointer drifts off the canvas. Keyboard focus
alone (Tab, Ctrl+P, a click) still does not reveal.

### Concealed appearance (unchanged)

- Images: the opaque dark cover with "Hidden image"; layout and size kept.
- Text: glyphs, caret, selection and placeholder removed; field frames, row
  heights and scroll extents kept, so the layout never jumps on reveal.
- Accessibility labels and values of concealed content are cleared, as today.

## 3. Mechanism

One helper owns the rule; components stop computing their own reveal.

```
helto::canvas(ui, id, rect, |ui| …)   // a panel region; or for a window body
helto::canvas_window(ui, id, |ui| …)  // inside Window/Modal show closures
helto::revealed_here(ui) -> bool      // what private content asks
helto::private_text(ui, |ui| …)       // conceals the creative text drawn inside
```

- `canvas` computes `revealed` once per frame from the canvas rect (or the
  window's area rect) and the pointer, stores it with the canvas ID in egui
  context data, and pushes it on a per-frame canvas stack while its body runs.
- Popups and tooltips drawn inside the body inherit the top of the stack. The
  canvas also records the layers its popups used this frame, so next frame a
  pointer over any of them counts as inside the canvas (fixes the LoRA list
  class of bug structurally).
- `textarea`, `image_view`, thumbnails, the video player, Take cards and
  `private_text` read `revealed_here(ui)`. Outside any canvas they default to
  hidden (fail closed), which a debug assertion reports in tests.
- `private_scope` remains as the concealment primitive behind `private_text`
  but is no longer wrapped around whole windows.

## 4. Changes by surface

| Surface | Today | After |
| --- | --- | --- |
| LoRA window, LoRA info, presets window, Project generation, models library, compatibility copy, reference cache | Whole body blanked | Fully visible |
| Check Media / Relink, Open private project | Whole body blanked | Visible; media file names and scene titles hidden per canvas |
| Prompt fields, reference fields | Reveal per field | Reveal with their panel |
| Viewer images, reference images, thumbnails | Reveal per image | Reveal with their panel |
| Media bin, Cast, Library, pickers | Per image and per window | Images, names and file names reveal with the window or panel |
| Timeline scene titles | Reveal over the track canvas | Unchanged rule, now through `canvas` |
| Story, authoring, compiled plan, enhancement | Per region or whole window | Creative text hidden, controls and settings visible |
| Error strips | "Hidden error" when not hovered | Visible, unless the error quotes creative writing |
| Queue dock | Already canvas-based | Moved onto the shared helper |

## 5. Tests

- Unit: `canvas` reveal from pointer position, window focus, a popup layer owned
  by the canvas, another canvas's layer, and the typing exception.
- Mesh tests per kind (image, textarea, `private_text`) in a canvas: hidden
  without hover, revealed with the pointer anywhere in the canvas (not over the
  widget), hidden again after leaving or losing focus.
- Updated headless layout tests: the LoRA, presets, generation and models
  windows render their settings text with Hide previews on; prompts, images,
  titles, character names and media file names do not.
- Accessibility: concealed creative text never appears in AccessKit output.
- A source check that no surface calls the removed per-widget reveal helpers.

## 6. Documentation

Replace the per-field descriptions in UI.md ("Each image or text field reveals
only while the pointer is over it", the dock paragraph and similar) with one
section stating sections 1 and 2. Update the Helto privacy reference only if the
user wants the canvas rule recorded as the design-system default.

## 7. Decisions

1. **Private content:** decided — images and video frames, all creative writing,
   scene titles, character names and media file names. Project folder names,
   models, LoRAs, presets and settings are visible.
2. **Canvas:** decided — each panel and each window, with popups belonging to
   their canvas.
3. **Errors:** decided — error text that may quote creative writing is hidden
   like it. UI error strips show fixed messages mapped from error codes and are
   always visible; raw server messages on video jobs stay private.

## 8. Implementation notes

- `helto::canvas` / `helto::panel_canvas` mark main-window panels;
  `helto::window_canvas` is the first statement of every window and modal body.
  `helto::shown` replaces the per-widget `revealed` / `reveal_in_viewport`.
- A popup or tooltip counts as its canvas while that canvas was revealed the
  frame before and the popup is not itself a window canvas.
- Story, Enhance, Prompt tools and Plan Review consist almost entirely of
  creative text and keep concealing their whole body, now revealed by their
  panel or window canvas. Cast and Library character conceal only names, aliases,
  descriptions, tags, notes, voice text and images.
- In tests, `helto::outside_canvas` lists callers that asked about private
  content outside every canvas.
