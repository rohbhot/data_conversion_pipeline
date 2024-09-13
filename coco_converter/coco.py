import os
from .coco_to_intermediate import coco_to_intermediate
from .intermediate_to_coco import intermediate_to_coco
import logging


class COCOProcessor:
    def __init__(self):
        self.required_dir_structure = """
├── COCO Root
│   ├── train2017
│   |   │   file1.jpg
│   |   │   file2.jpg
│   |   │   ...
│   ├── val2017
│   |   │   file_v1.jpg
│   |   │   file_v2.jpg
│   |   │   ...
│   ├── test2017
│   |   │   file_t1.jpg
│   |   │   file_t2.jpg
│   |   │   ...
│   └── annotations
│       │   instances_train2017.json
│       │   instances_val2017.json
│       │   instances_test2017.json
"""

    def to_intermediate(self, data_path, write_path=None):
        """
        Takes in root directory of the dataset
        and returns a dataframe and path to the image directory.
        """
        df, image_dirs = coco_to_intermediate(data_path)
        if write_path:
            df.to_csv(write_path)
        return df, image_dirs

    def from_intermediate(self, df, image_dirs, output_dir):
        """
        Inputs: df: dataframe or path to csv
        image_dir: path to directory containing input images
        output_dir: path to create root directory of dataset
        """
        intermediate_to_coco(df, image_dirs, output_dir)

    def check_dir_structure(self, data_path):
        names = ["train2017", "val2017", "test2017"]
        valid = False

        for fname in names:
            if os.path.exists(
                os.path.join(data_path, "annotations", f"instances_{fname}.json")
            ) and os.path.exists(os.path.join(data_path, fname)):
                valid = True

        if not valid:
            logging.error(
                "Please use the following structure:\n" + self.required_dir_structure
            )

        return valid
