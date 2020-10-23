"""Basic scene file format as dict type"""

from numpy.lib.function_base import angle
import mitsuba
import render_config as config

mitsuba.set_variant(config.variant)

from mitsuba.core import ScalarTransform3f, ScalarTransform4f

scene_dict = {
    "type": "scene",
    "integrator": {
        "type": "path",
        "max_depth": 128
    },

    "sensor":{
        "type": "perspective",
        "fov": 39.3077,
        "near_clip": 0.1,
        "far_clip": 1000,
        "fov_axis": "smaller",

        "to_world": ScalarTransform4f.look_at(origin=[54, 162, 216],
                                            target=[0,0,10],
                                            up=[0,0,1]),

        "sampler": {
            "type": "independent",
            "sample_count": 128,
            "seed": 4
        },

        "film": {
            "type": "hdrfilm",
            "width": 256,
            "height": 256,
            "rfilter": {
                "type": "gaussian"
            }
        }
    },

    "emitter": {
        "type": "envmap",
        "filename": "envmap.exr",
        "to_world": ScalarTransform4f.rotate([1,0,0], angle=90)
    },

    "checker": {
        "type": "checkerboard",
        "color0": 0.4,
        "color1": 0.2,
        "to_uv": ScalarTransform4f.scale(2)
    },

    "planemat": {
        "type": "diffuse",
        "reflectance": {
            "type": "ref",
            "id": "checker"
        }
    },

    "floor": {
        "type": "rectangle",
        "to_world": ScalarTransform4f.scale(150),
        "floor_bsdf": {
            "type": "ref",
            "id": "planemat"
        }
    },

    "wall1": {
        "type": "rectangle",
        "to_world": ScalarTransform4f.translate([-150, 0, 150])
                    * ScalarTransform4f.rotate([0,1,0], angle=90)
                    * ScalarTransform4f.scale(150)
    },

    "wall2": {
        "type": "rectangle",
        "to_world": ScalarTransform4f.translate([0, -150, 150])
                    * ScalarTransform4f.rotate([1,0,0], angle=-90)
                    * ScalarTransform4f.scale(150)
    },

    # bottom
    "1": {
        "type": "rectangle",
        "to_world": ScalarTransform4f.rotate([1,0,0], angle=180)
                    * ScalarTransform4f.scale(30)
    },

    # side
    "2": {
        "type": "rectangle",
        "to_world":  ScalarTransform4f.translate([0, 30, 30])
                    * ScalarTransform4f.rotate([1,0,0], angle=-90)
                    * ScalarTransform4f.scale(30)
    },

    "3": {
        "type": "rectangle",
        "to_world":  ScalarTransform4f.translate([30, 0, 30])
                    * ScalarTransform4f.rotate([0,1,0], angle=90)
                    * ScalarTransform4f.scale(30)
    },

    "4": {
        "type": "rectangle",
        "to_world":  ScalarTransform4f.translate([0, -30, 30])
                    * ScalarTransform4f.rotate([1,0,0], angle=90)
                    * ScalarTransform4f.scale(30)
    },
    
    "5": {
        "type": "rectangle",
        "to_world":  ScalarTransform4f.translate([-30, 0, 30])
                    * ScalarTransform4f.rotate([0,1,0], angle=-90)
                    * ScalarTransform4f.scale(30)
    },

    # top
    "6": {
        "type": "rectangle",
        "to_world": ScalarTransform4f.translate([0,0,60])
                    * ScalarTransform4f.scale(30)
    }
    
}