import os
import os.path as osp
import pandas as pd
import json
import sys
import tempfile
import shutil
import logging


def id_mapping(image_list):
    mapping = {}
    for image_data in image_list:
        mapping[image_data["id"]] = {
            key: value for key, value in image_data.items() if key != "id"
        }
    return mapping


def get_classes(categories):
    class_map = {}
    for category in categories:
        class_map[category["id"]] = category["name"]
    return class_map


def json_to_rows(data_path, path, type_):
    if osp.exists(path):
        with open(path) as f:
            data = json.load(f)

        image_dir = ["train2017", "val2017", "test2017"][type_]

        id_to_metadata = id_mapping(data["images"])
        class_map = get_classes(data["categories"])

        rows = []
        not_found = []
        for annotation in data["annotations"]:
            file_name = id_to_metadata[annotation["image_id"]]["file_name"]
            if osp.exists(osp.join(data_path, image_dir, file_name)):
                # create row ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'type']
                rows.append(
                    [
                        file_name,
                        id_to_metadata[annotation["image_id"]]["width"],
                        id_to_metadata[annotation["image_id"]]["height"],
                        class_map[annotation["category_id"]],
                        annotation["bbox"][0],
                        annotation["bbox"][1],
                        annotation["bbox"][0] + annotation["bbox"][2],
                        annotation["bbox"][1] + annotation["bbox"][3],
                        type_,
                    ]
                )
            else:
                not_found.append(file_name)

        return {"rows": rows, "not_found": not_found}

    else:
        logging.warning(f"File {path} not found!")
        return None


def coco_to_intermediate(data_path):
    json_files = [
        "instances_train2017.json",
        "instances_val2017.json",
        "instances_test2017.json",
    ]
    image_dirs = ["train2017", "val2017", "test2017"]
    valid_image_dirs = []
    type_ = 0
    rows = []
    not_found = []
    for json_file in json_files:
        if osp.exists(osp.join(data_path, "annotations", json_file)):
            data = json_to_rows(
                data_path, osp.join(data_path, "annotations", json_file), type_
            )
            if data is not None:
                rows += data["rows"]
                not_found += data["not_found"]
                valid_image_dirs.append(osp.join(data_path, image_dirs[type_]))

        type_ += 1

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
    df = pd.DataFrame(rows, columns=column_names)
    nl = "\n"
    if not_found:
        logging.warning(f"Following file(s) were not found:\n{nl.join(not_found)}")

    return df, valid_image_dirs


if __name__ == "__main__":
    data_path = "/mnt/2tb/General/Anand/split_images/data_conversion/coco"
    output_dir = "/mnt/2tb/General/Anand/split_images/data_conversion/coco/new_coco"
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    json_df, image_dirs = coco_to_intermediate(data_path)
    if json_df is not None:
        print(image_dirs)
        n_rows = len(json_df)
        print(f"Successfully converted test xml to csv, size: {n_rows}")
        csv_path = osp.join(output_dir, "labels.csv")
        json_df.to_csv(csv_path, index=False)
    else:
        print("Could not create csv")
