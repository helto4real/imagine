Character sheet guide (MOSS).

A character sheet is one image: a face portrait beside a full-body turnaround of the same character seen from the front, the right side and the back. MOSS draws the whole sheet in one generation from text alone. Imagine writes the generation prompt itself. Do not send parameters.prompt, width or height; describe the character in parameters.character_sheet instead:

- character_description (required): the person's physical identity only. Age, gender presentation, build and body proportions, face shape and features, skin, hair (colour, length, style), facial hair, eyes and distinguishing marks. Plain descriptive English, one or two sentences of concrete visible traits. No clothing, setting, pose, mood or story.
- clothing_description: one coherent outfit with accessories, from head to shoes (garments, colours, materials, notable details). Headwear, glasses and jewellery also appear in the portrait. Leave it empty to let Imagine choose one outfit.

MOSS takes no reference images, so this profile has no identity references. To keep an existing character's face, use the qwen-character-sheet profile with reference images instead.

Leave character_sheet.layout at its defaults unless the user asks for a different sheet width.

For a video cast, write the same appearance and outfit you use for the character in the storyboard, and register the whole sheet as the character's identity image.

Example: {"character_sheet": {"character_description": "A man in his late sixties, tall and lean, weathered tanned face with deep lines, short white beard, pale grey eyes, close-cropped grey hair", "clothing_description": "dark navy wool fisherman's sweater, black knit watch cap, heavy grey canvas trousers, worn brown leather boots"}}
