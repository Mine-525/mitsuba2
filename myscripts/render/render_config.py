"""Configuration for rendering with path tracing and VAE BSSRDF"""

from os import truncate


variant = "gpu_rgb"
rr_depth = 5
max_depth = 10
spp = 32
sample_per_pass = 2048

film_width = 256
film_height = 256

seed = 4

model_name = "best_model"

enable_bssrdf = True
visualize_invalid_sample = False
multi_process = False

