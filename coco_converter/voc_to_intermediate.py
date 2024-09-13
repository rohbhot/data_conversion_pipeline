import os
import os.path as osp
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import sys
import logging


def xml_to_csv(
    path,
):  

    xml_list = []
    IMAGE_DIR = osp.join(path, "JPEGImages")
    ANNOTATIONS_DIR = osp.join(path, "Annotations")
    SPLIT_PATH_DIR = osp.join(path, "ImageSets", "Main")
    not_found = []

    def _read_file(file_, type_):
        if osp.exists(xml_file):

            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()

                width = int(root.find("size/width").text)
                height = int(root.find("size/height").text)
                image_name = root.find("filename").text

                if osp.exists(osp.join(IMAGE_DIR, image_name)):

                    for o in root.findall("object"):
                        # create row ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'type']
                        xmin = int(o.find("bndbox/xmin").text)
                        xmax = int(o.find("bndbox/xmax").text)
                        ymin = int(o.find("bndbox/ymin").text)
                        ymax = int(o.find("bndbox/ymax").text)
                        class_name = o.find("name").text

                        value = (
                            image_name,
                            width,
                            height,
                            class_name,
                            xmin,
                            ymin,
                            xmax,
                            ymax,
                            type_,
                        )

                        xml_list.append(value)
                else:
                    not_found.append(image_name)
            except:
                pass
            else:
                not_found.append(xml_file)

    if osp.exists(SPLIT_PATH_DIR) and len(os.listdir(SPLIT_PATH_DIR)) > 1:
        for file_ in os.listdir(SPLIT_PATH_DIR):
            if "train" in file_:
                type_ = 0
            elif "val" in file_:
                type_ = 1
            elif "test" in file_:
                type_ = 2
            else:
                logging.warn(
                    f"Unknown file type{osp.join(SPLIT_PATH_DIR, file_)}, skipping..."
                )
                continue

            with open(osp.join(SPLIT_PATH_DIR, file_)) as f:
                for file_name in f.readlines():
                    xml_file = osp.join(ANNOTATIONS_DIR, f"{file_name}.xml")
                    _read_file(xml_file, type_)

    else:

        # Extract the directory name from ANNOTATIONS_DIR
        dir_name = os.path.basename(os.path.dirname(ANNOTATIONS_DIR))

        # Get the parent directory name
        parent_dir_name = os.path.basename(
            os.path.dirname(os.path.dirname(ANNOTATIONS_DIR))
        )

        # Set type_ based on the parent directory name
        if parent_dir_name == "VOCdevkit_train":
            type_ = 0
        elif parent_dir_name == "VOCdevkit_test":
            type_ = 2
        elif parent_dir_name == "VOCdevkit_val":
            type_ = 1
        else:
            # Handle any other cases
            type_ = None
            # You can choose to raise an exception, print an error message, or handle it differently based on your needs

        if type_ is not None:
            # Process XML files
            for xml_file in glob.glob(osp.join(ANNOTATIONS_DIR, "*.xml")):
                _read_file(xml_file, type_)
        else:
            print("Invalid parent directory name:", parent_dir_name)

        #         for xml_file in glob.glob(osp.join(ANNOTATIONS_DIR, '*.xml')):
        #             _read_file(xml_file, type_=0)

    column_names = [
        "filename",
        "width",
        "height",
        "class",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "type",
    ]
    xml_df = pd.DataFrame(xml_list, columns=column_names)
    nl = "\n"
    if not_found:
        logging.warning(f"Following file(s) were not found:\n{nl.join(not_found)}")
    return xml_df


def voc_to_intermediate(data_path):
    xml_df = xml_to_csv(osp.join(data_path, "VOC2007"))
    return xml_df, [osp.join(data_path, "VOC2007", "JPEGImages")]


if __name__ == "__main__":
    data_path = sys.argv[1]
    output_dir = sys.argv[2]
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    xml_df, image_dires = voc_to_intermediate(data_path)
    if xml_df is not None:
        n_rows = len(xml_df)
        print(image_dirs)
        logging.info(f"Successfully converted test xml to csv, size: {n_rows}")
        csv_path = osp.join(output_dir, "labels.csv")
        xml_df.to_csv(csv_path, index=False)
    else:
        logging.error("Could not create csv")
