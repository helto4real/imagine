MOSS image edit guide.

MOSS edits one to four reference images by a text instruction, using the Moss-Image-Edit LoRA. Register the images with register_image for this profile and pass their IDs in order. The output follows the first reference's aspect ratio, with its long side at max_side (1024 unless you set it).

Write parameters.prompt as an edit instruction in plain English: state the change, then what must stay the same. Edits of colour, outfit, lighting, season and visual style work best; single-image edits are the most reliable.

With several references, say what each image is for by its order ("the first image", "the second image"), for example which one supplies the person and which the outfit.

Example: "Change only the pink dress to bright red. Keep the character, face, hair, pose, background and composition unchanged."
