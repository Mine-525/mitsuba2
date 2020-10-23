"""Basic scene file format of meshes for bssrdf"""

import sys
sys.path.append("myscripts/render")

import mitsuba
import render_config as config
import data_pipeline

mitsuba.set_variant(config.variant)

from mitsuba.core import ScalarTransform4f


class meshes:
    """Super class for BSSRDF meshes"""

    def __init__(self):
        self.n_mesh = 6
        self.type = [None for i in range(self.n_mesh)]
        self.filename = [None for i in range(self.n_mesh)]
        self.translate = [None for i in range(self.n_mesh)]
        self.rotate = [None for i in range(self.n_mesh)]
        self.scale = [None for i in range(self.n_mesh)]

    def register_all_mesh(self, bdata):
        """Register whole meshes in this class to BSSRDF_Data"""

        for i in range(self.n_mesh):
            bdata.register_mesh(i+1, self.type, filename=self.filename[i],
                                translate=self.translate[i], rotate=self.rotate[i],
                                scale=self.scale[i]
                                )

class meshes_rectangle(meshes):
    """
    Rectangle meshes for BSSRDF
    Mesh ID indicates each mesh below,
    1: bottom
    2: side for positive y
    3: side for positive x
    4: side for negative y
    5: side for negative x
    6: top
    """

    def __init__(self):
        super(meshes_rectangle, self).__init__()
        self.n_mesh = 6
        self.type = "rectangle"
        self.translate = [
            ScalarTransform4f(),
            ScalarTransform4f.translate([0, 30, 30]),
            ScalarTransform4f.translate([30, 0, 30]),
            ScalarTransform4f.translate([0, -30, 30]),
            ScalarTransform4f.translate([-30, 0, 30]),
            ScalarTransform4f.translate([0, 0, 60])
        ]
        self.rotate = [
            ScalarTransform4f.rotate([1,0,0], angle=180),
            ScalarTransform4f.rotate([1,0,0], angle=-90),
            ScalarTransform4f.rotate([0,1,0], angle=90),
            ScalarTransform4f.rotate([1,0,0], angle=90),
            ScalarTransform4f.rotate([0,1,0], angle=-90),
            ScalarTransform4f()
        ]
        self.scale = [
            ScalarTransform4f.scale(30),
            ScalarTransform4f.scale(30),
            ScalarTransform4f.scale(30),
            ScalarTransform4f.scale(30),
            ScalarTransform4f.scale(30),
            ScalarTransform4f.scale(30)
        ]



    