"""
Data pipeline for rendering
Handle some data, such as, mesh transform matrix for BSSRDF
and create scene format
"""

from numpy.core.fromnumeric import transpose
import mitsuba
import render_config as config

mitsuba.set_variant(config.variant)

from mitsuba.core import ScalarTransform4f


class BSSRDF_Data:
    """  Class for handling BSSRDF data """

    def __init__(self):
        self.bssrdf = {}
        self.mesh = {}

    def register_medium(self, id, ior=1.5, scale=1.0,
                        sigam_t=1.0, albedo=0.5, g=0.25):
        """
        Register medium with mesh id

        Args:
            id: Mesh id, which must start from 1
            ior, scale, sigma_t, albedo, g: Medium parameters
        """

        if id in self.bssrdf:
            print("This ID has already used. Do you want to overwrite? [y/n]")
            while(True):
                key = input()
                if key == "y":
                    print("ID " +"\"" + str(id) + "\"" " is overwritten")
                    break

                elif key == "n":
                    print("Register canceled")
                    return

                else:
                    print("Input valid key")
                    continue

        self.bssrdf[id] = {
            "type": "bssrdf",
            "int_ior": ior,
            "scale": scale,
            "sigma_t": sigam_t,
            "albedo": albedo,
            "g": g,
            "mesh_id": id
            }

    def register_mesh(self, id, mesh_type, filename=None, translate=ScalarTransform4f(),
                      rotate=ScalarTransform4f(), scale=ScalarTransform4f()):
        """
        Register mesh data
        """
        self.mesh[id] = {
            "type": mesh_type,
            "filename": filename,
            "translate": translate,
            "rotate": rotate,
            "scale": scale
        }

    def add_object(self, scene_dict):
        """
        Add registered meshes to given scene file format
        """

        if len(self.bssrdf) != len(self.mesh):
            exit("The number of registerd mesh and bssrdf are different!")

        num_obj = len(self.bssrdf)

        for i in range(num_obj):
            i += 1
            bssrdf = self.bssrdf[i]
            mesh = self.mesh[i]


            if self.mesh[i]["type"] == "rectangle":
                scene_dict[str(i)] = {
                    "type": mesh["type"],
                    "to_world": mesh["translate"]
                                * mesh["rotate"]
                                * mesh["scale"],
                }
        
            else:
                scene_dict[str(i)] = {
                    "type": self.mesh["type"],
                    "filename": mesh["filename"],
                    "to_world": mesh["translate"]
                                * mesh["rotate"]
                                * mesh["scale"]
                }

            bsdf = {
                "bsdf_" + str(i): bssrdf
            }

            scene_dict[str(i)].update(bsdf)
            

        return scene_dict

                

            


