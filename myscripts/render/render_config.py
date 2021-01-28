"""Configuration for rendering with path tracing and VAE BSSRDF"""

from os import truncate


variant = "gpu_rgb"
rr_depth = 5
max_depth = 10
spp = 4
sample_per_pass = 8192

film_width = 256
film_height = 256

seed = 4

model_name = "best_model"
im_size = 63

enable_bssrdf = True
visualize_invalid_sample = False
multi_process = False
aovs = True

zoom = True
