"""
Processing trainingdata
"""
import os
import shutil
import glob
import datetime
import time
import numpy as np
import pandas as pd
import cv2
import utils
import matplotlib.pyplot as plt
from traindata_config import TrainDataConfiguration


class DataHandler:
    """
    csv data handler for post proccessing
    """

    def __init__(self, config):
        self.sample_path = config.SAMPLE_DIR
        self.list_update()
        self.train_image_dir_path = f"{config.IMAGE_DIR}"
        self.map_dir_path = f"{config.MAP_DIR}"
        self.train_sample_dir_path = f"{config.TRAIN_DIR}"
        self.debug = config.DEBUG
        self.tag = config.tag

    def list_update(self):
        self.sample_list = glob.glob(f"{self.sample_path}\\*")
        

    def generate_train_data(self, offset=0):
        """
        Generating height map w.r.t incident point and medium parameters
        Args:
            offset: ID offset
        """
        height_map = None
        id_data = 0
        height_map_list = glob.glob(f"{self.map_dir_path}\\height_map*.png")
        df_all = pd.DataFrame()

        # File operation
        key_input = None
        file_list = glob.glob(self.train_image_dir_path)
        if(os.path.exists(self.train_image_dir_path)):
            print(f"Delete sample files in {self.train_image_dir_path}? [y/n]")
            while(True):
                key_input = input()
                if(key_input == "y"):
                    shutil.rmtree(self.train_image_dir_path)
                    os.mkdir(self.train_image_dir_path)
                    break
                elif(key_input == "n"):
                    break
                print("Please input valid letter")
        else:
            os.makedirs(self.train_image_dir_path)
        
        # Process with given csv files
        for i, file_name in enumerate(self.sample_list):
            # get id number and map number
            data = pd.read_csv(file_name)
            data["id"] = data.index + id_data + offset
            df_model_id = pd.DataFrame(np.ones([len(data), 1]) * i, columns=["model_id"])
            data["model_id"] = df_model_id["model_id"]

            # join train data into one file
            df_all = df_all.append(data)

            # Get entire height map for a training data
            height_map = cv2.imread(height_map_list[i], cv2.IMREAD_GRAYSCALE)

            # Make training images directory for each height map
            if(key_input == "y"):
                os.makedirs(f"{self.train_image_dir_path}\\map_{i:03}")

            file_path = None
        
            # Process with each training data
            for row in data.itertuples():
                if(row.id % 10000 == 0):
                    file_path = f"{self.train_image_dir_path}\\map_{i:03}\\images{row.id}_{row.id + 9999}"
                    os.mkdir(file_path)

                image = gen_train_image(row, height_map, self.debug)

                # Save height map image with id
                cv2.imwrite(f"{file_path}\\train_image{row.id:08}.png", image)
                id_data += 1

                if(row.id % 1000 == 0):
                    print(f"{datetime.datetime.now()} -- Log: Processed {row.id}")

        # refine and output sampled path data
        time.sleep(2)
        print(df_all)
        self.refine_data(df_all)


    def refine_data(self, df):
        """
        Refine and Save training data.
        Refining includes select, add and scale training data

        Args:
            df: row training data
        """
        # Get new params
        df["eff_albedo"] = utils.reduced_albedo_to_effective_albedo(
            utils.get_reduced_albedo(df["albedo"], df["g"], df["sigma_t"])
            )
        df["height_max"] = df["scale_z"]
        df["sigma_n"] = utils.get_sigman(df)

        # Scale incident and outgoing position with sigma n
        df["p_in_x"] = df["p_in_x"] / df["sigma_n"]
        df["p_in_y"] = df["p_in_y"] / df["sigma_n"]
        df["p_in_z"] = df["p_in_z"] / df["sigma_n"]
        df["p_out_x"] = df["p_out_x"] / df["sigma_n"]
        df["p_out_y"] = df["p_out_y"] / df["sigma_n"]
        df["p_out_z"] = df["p_out_z"] / df["sigma_n"]

        df = df[self.tag]
        df.to_csv(f"{self.train_sample_dir_path}\\train_path.csv",
                  index=False, float_format="%.6g")


        
    def delete_sample_files(self):
        file_list = glob.glob(f"{self.sample_path}\\*")
        if(os.path.exists(self.sample_path)):
            print(f"Delete sample files in {self.sample_path}? [y/n]")
            while(True):
                key_input = input()
                if(key_input == "y"):
                    shutil.rmtree(self.sample_path)
                    time.sleep(0.5)
                    os.makedirs(self.sample_path)
                    break
                elif(key_input == "n"):
                    break
                print("Please input valid letter")
        else:
            os.makedirs(self.sample_path)

            



def join_scale_factor(path, scale):
    """
    Add scale factor of shape objects to the sampled data from given path

    Args:
    path: Path of a file which contains sampled data
    scale: Scale factor of shape objects
    """
    data = pd.read_csv(path)
    data["scale_x"] = scale["scale_x"]
    data["scale_y"] = scale["scale_y"]
    data["scale_z"] = scale["scale_z"]

    data.to_csv(path, index=False)

def join_model_id(path, model_id):
    """
    Add scale model id to the sampled data from given path

    Args:
    path: Path of a file which contains sampled data
    id: Model id number of shape objects
    """
    data = pd.read_csv(path)
    data["model_id"] = model_id["model_id"]

    data.to_csv(path, index=False)


def gen_train_image(data, height_map, debug):
    """
    Generate training image data from path sample data and entire height map.
    Generated image is centered by incident location.
    Sigma_n of the target medium defines a range of the resulting image.
    Assuming the unscaled original size of 3D model is 50 * 50 * 21 [mm]

    Args:
        data: pandas DataFrame for one path sampling data
        height_map: Entire height map for sampled 3D model
    Return:
        train_image: Generated height map centered by incident location. The size is 255*255 [pxl]
    """

    # Get medium parameters, incident loaction, and scale factors
    medium = {}
    medium["sigma_t"] = data.sigma_t
    medium["albedo"] = data.albedo
    medium["g"] = data.g

    x_in = data.p_in_x
    y_in = data.p_in_y
    z_in = data.p_in_z
    scale_x = data.scale_x
    scale_y = data.scale_y
    scale_z = data.scale_z

    # xy coordinates of u-v origin, 50 is size of 3D model assumed   
    x_min = -25 * scale_x
    y_max = 25 * scale_y

    x_range = -x_min * 2
    y_range = y_max * 2


    # Scale height map with scale factors as sampling
    map_scaled = cv2.resize(height_map, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_AREA)

    return clip_scaled_map(map_scaled, [x_in, y_in], medium, x_range, y_range, x_min, y_max)

def clip_scaled_map(map_scaled, pos_in, medium, x_range, y_range, x_min, y_max):
    height_scaled, width_scaled = map_scaled.shape

    x_in = pos_in[0]
    y_in = pos_in[1]

    # sigma_n of target meidum. sigma_n defines the range
    sigma_n = utils.get_sigman(medium)
    # Length of a pixel edge
    px_len = x_range / width_scaled
    # The number of pixels in 6 sigma_n
    r_px_range = np.ceil(6 * sigma_n / px_len).astype(np.uint32)

    u_c = int((y_max - y_in) * height_scaled / y_range)
    v_c = int((x_in - x_min) * width_scaled / x_range)

    # Clip map_full_scaled in 255*255 from incident position
    distance_u_n = np.min([r_px_range, u_c])              # distance of u for negative direction
    distance_u_p = np.max([np.min([r_px_range, height_scaled - u_c - 1]), 0]) # distance of u for positive direction
    distance_v_n = np.min([r_px_range, v_c])              # distance of v for negative direction
    distance_v_p = np.max([np.min([r_px_range, width_scaled - v_c - 1]), 0])  # distance of v for positive direction

    u_range = [u_c - distance_u_n, u_c + distance_u_p + 1]
    v_range = [v_c - distance_v_n, v_c + distance_v_p + 1]
    map_clip = map_scaled[u_range[0]: u_range[1], v_range[0]:v_range[1]]

    height_clip, width_clip = map_clip.shape

    scale_map = 255 / (12 * sigma_n / px_len)
    map_clip = cv2.resize(map_clip, None, fx=scale_map, fy=scale_map, interpolation=cv2.INTER_AREA)
    height_clip, width_clip = map_clip.shape

    u_c = int(height_clip * distance_u_n / (distance_u_n + distance_u_p))
    v_c = int(width_clip * distance_v_n / (distance_v_n + distance_v_p))

    r_px = 127

    distance_u_n = np.min([r_px, u_c])              
    distance_u_p = np.max([np.min([r_px, height_clip - u_c - 1]), 0])
    distance_v_n = np.min([r_px, v_c])
    distance_v_p = np.max([np.min([r_px, width_clip - v_c - 1]), 0])
    

    # Paste map_clip to a canvas ranging 255*255 [px]
    canvas = np.ones([2 * r_px + 1, 2 * r_px + 1], dtype="uint8") * 31
    canvas_c = r_px + 1
    canvas_size = 2 * r_px + 1

    canvas[r_px-distance_u_n:r_px+distance_u_p, r_px-distance_v_n:r_px+distance_v_p] = map_clip[u_c-distance_u_n:u_c+distance_u_p, v_c-distance_v_n:v_c+distance_v_p]

    # If pixels are out of range 6 sigma_n, set pixel value to 0 by masking
    uv_range = np.arange(1, canvas_size + 1)
    V, U = np.meshgrid(uv_range, uv_range)
    mask = (V - canvas_c)**2 + (U - canvas_c)**2 <= r_px**2
    canvas[np.logical_not(mask)] = 0

    return canvas

if __name__ == "__main__":
    config = TrainDataConfiguration()
    d_handler = DataHandler(config)
    d_handler.generate_train_data()
