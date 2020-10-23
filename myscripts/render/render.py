import os
import sys
import time
import mitsuba
import integrate
import render_config as config
import data_pipeline

sys.path.append("myscripts/render/dict")
from scene_dict import scene_dict
from mesh_dict import meshes_rectangle

mitsuba.set_variant(config.variant)

from mitsuba.core import Thread
from mitsuba.core.xml import load_file, load_dict

##### Setting scene #####
bdata = data_pipeline.BSSRDF_Data()
for i in range(6):
    i += 1
    bdata.register_medium(i)

meshes_rectangle().register_all_mesh(bdata)

scene_dict = bdata.add_object(scene_dict)
scene = load_dict(scene_dict)

# Rendering settings
spp = 128
sample_per_pass = 16

print("Rendering start")

start = time.time()
integrate.render(scene, spp, sample_per_pass)
process_time = time.time() - start

print(f"Rendering end (took {process_time}s)")