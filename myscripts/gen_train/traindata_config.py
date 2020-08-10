"""
Configuration of generating train data
"""


class TrainDataConfiguration:
    def __init__(self):
        self.mode = "visual" # visualizing medium appearance
        self.mfix = False # generating fixed medium parameters

        self.OUT_DIR = "C:\\Users\\mineg\\mitsuba2\\myscripts\\csv_files"
        if (self.mode is "visual"):
            self.XML_PATH = "C:\\Users\\mineg\\mitsuba2\\myscripts\\gen_train\\scene_templates\\visual_template.xml"
            self.spp = 256
        elif(self.mode is "sample"):
            self.XML_PATH = "C:\\Users\\mineg\\mitsuba2\\myscripts\\gen_train\\scene_templates\\sample_template.xml"
            self.spp = 1024
        elif(self.mode is "test") :
            self.XML_PATH = "C:\\Users\\mineg\\mitsuba2\\myscripts\\gen_train\\scene_templates\\test_template.xml"
            self.spp = 1024

        # TO DO
        # - glob multiple serialized path in a directory
        self.SERIALIZED_PATH = "myscripts\\gen_train\\scene_templates\meshes\leather2.serialized"
