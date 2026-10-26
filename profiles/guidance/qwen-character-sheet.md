Character sheet guide.

A character sheet is one image: a face portrait beside a full-body turnaround of the same character in several directions. Imagine writes the generation prompt itself. Do not send parameters.prompt, width or height; describe the character in parameters.character_sheet instead:

- character_description: the person's physical identity only. Age, gender presentation, build and body proportions, face shape and features, skin, hair (colour, length, style), facial hair, eyes and distinguishing marks. Plain descriptive English, one or two sentences of concrete visible traits. No clothing, setting, pose, mood or story. Required unless you register identity reference images.
- clothing_description: one coherent outfit with accessories, from head to shoes (garments, colours, materials, notable details). It applies to the full-body turnaround; the portrait is a face close-up, so clothing does not show there. Leave it empty to keep the outfit of the identity references, or to let Imagine choose one.

Identity references (optional, up to 8): register images of the same real or existing character with register_image for this profile and pass their IDs in order. They keep the face and body; character_description then only adds or clarifies traits the images do not show, and clothing_description replaces their outfit.

Leave character_sheet.layout at its defaults unless the user asks for a different sheet width.

For a video cast, write the same appearance and outfit you use for the character in the storyboard, and register the whole sheet as the character's identity image.

Example: {"character_sheet": {"character_description": "A man in his late sixties, tall and lean, weathered tanned face with deep lines, short white beard, pale grey eyes, close-cropped grey hair", "clothing_description": "dark navy wool fisherman's sweater, black knit watch cap, heavy grey canvas trousers, worn brown leather boots"}}
