MOSS Image prompt guide (adapted from OpenMOSS's MOSS-Image user guide and prompt enhancer skill, Apache-2.0).

MOSS Image is text-to-image only: the prompt is the whole input. It reads long, detailed prompts well.

Choose the prompt form:
- Natural language (default): a caption for ordinary scenes, photographs and illustrations.
- Tag sequence: for anime or illustration requests phrased as tags.
- JSON layout: only when exact text placement, posters, multi-panel designs or controlled object positions matter. A detailed image does not by itself need a layout.
Natural language and tags can be combined.

Natural-language captions:
- Write in English unless the user asks for another language; English is the most reliable.
- Write cohesive descriptive prose; paragraphs are fine. Describe the visible choices: subject, action, setting, framing, medium, lighting, materials, palette and mood. Describe spatial relations in words.
- Keep the user's subjects, counts, relationships, medium, style and exclusions. Fill small creative gaps yourself; ask only when a missing decision would change the image materially. Do not invent factual details of real events or products.
- Avoid redundant quality tags, unrequested style changes and process notes. Keep a minimalist request minimal.

Text in the image:
- Put the exact text in quotation marks, verbatim: language, punctuation, capitalization.
- Write explicit line breaks where lines must break; automatic line breaks are not reliable.
- For designed images (posters, signs, covers), state reading order and prominence, and write the real copy, never placeholders.

Tag sequences:
- Order: quality, score, meta, year, character, series, artist, general.
- Example: best quality, highres, hatsune miku, vocaloid, @foo bar, 1girl, blue hair

JSON layouts:
- The prompt is one line of JSON: {"summary": {"caption": "...", "meta": [...], "canvas": "single"}, "items": [...]}.
- summary.caption (required) describes the whole scene and its style. Optional: meta (medium and genre tags), canvas (single, multi_panel or collage), palette.
- Each item: label (object, character or text), caption (visible appearance; for text also typography and surface treatment), text (text items only: the exact visible characters, line breaks as \n) and bbox. Optional tags and palette string arrays. No other fields: no ids, children or notes.
- bbox is [y_min, x_min, y_max, x_max] in integer pixels with the origin at the image centre: x grows to the right, y grows downward. From top-left pixel coordinates, subtract half the width from x and half the height from y; on a 1024 x 1024 image the values run from -512 to 512.
- Boxes are measured on the image size the job uses (its width and height). Positions are global, even for text on an object. Boxes may overlap. A box marks placement only; it does not draw a border or panel.
- Use one box per coherent visible unit; split text into separate items only when the parts must be placed independently.
- Check every string against the requested wording. If text will not fit or read well, enlarge or recompose the layout instead of dropping copy.

Example caption: A wide view of a refrigerator beside a kitchen counter with a bowl of fruit. A yellow handwritten sticky note on the door reads "Buy milk\nand eggs". Soft morning light, photographic.
