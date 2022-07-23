import numpy as np
from PIL import Image

import clip_jax

image_fn, text_fn, jax_params, jax_preprocess = clip_jax.load('ViT-B/32', "cpu")

image = np.expand_dims(jax_preprocess(Image.open("CLIP.png")), 0)
text = clip_jax.tokenize(["a diagram", "a dog", "a cat"])

image_embed = image_fn(jax_params, image)
text_embed = text_fn(jax_params, text)

print([np.linalg.norm(image_embed - text_embed[i]) for i in range(3)])