import os
import logging
from .voc_to_intermediate import voc_to_intermediate
from .intermediate_to_voc import intermediate_to_voc


class VOCProcessor:
    def __init__(self):

        self.required_dir_structure = """
├── VOCdevkit
│   └── VOC2007
│       ├── Annotations
│       │   file1.xml
│       │   file2.xml
│       │   ...
│       ├── ImageSets
│       │   └── Main
│       │       ├── test.txt (filenames without extensions)
│       │       ├── train.txt
│       │       └── val.txt
│       ├── JPEGImages
│       │   file1.jpg
│       │   file2.jpg
│       │   ...
"""

    def to_intermediate(self, data_path, write_path=None):
        """
        Takes in root directory of the dataset
        and returns a dataframe and path to the image directory.
        """

        df, image_dirs = voc_to_intermediate(
            data_path
        )  
        if write_path:
            df.to_csv(write_path)
        return df, image_dirs

    def from_intermediate(self, df, image_dirs, output_dir):
        """
        Inputs: df: dataframe or path to csv
        image_dir: path to directory containing input images
        output_dir: path to create root directory of dataset
        """
        intermediate_to_voc(df, image_dirs, output_dir)

    def check_dir_structure(self, data_path):
        data_path = os.path.join(data_path, "VOC2007")
        IMAGE_DIR = os.path.join(data_path, "JPEGImages")
        ANNOTATIONS_DIR = os.path.join(data_path, "Annotations")
        SPLIT_PATH_DIR = os.path.join(data_path, "ImageSets", "Main")

        required = [IMAGE_DIR, ANNOTATIONS_DIR, SPLIT_PATH_DIR]

        for dir_ in required:
            if not os.path.exists(dir_):
                logging.error(
                    "Please use the following structure:\n"
                    + self.required_dir_structure
                )
                return False

        return True
