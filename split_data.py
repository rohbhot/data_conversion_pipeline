import os
import random
import shutil
import argparse
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from split_voc import process_values
from coco_converter.data_converter import DataConverter
import augmentation

class ImageSplitter:
    def __init__(self):
        random.seed(42)

    def get_image_files(self, folder):
        return [
            f
            for f in os.listdir(folder)
            if f.endswith((".jpg", ".JPG", ".png", ".PNG", ".bmp", ".BMP"))
        ]

    def split_images(
        self,
        input_folder,
        output_folder,
        copy_images,
        split_percentages,
        augmentation_type,
        augment_size,
    ):

        train_percent, test_percent, val_percent = split_percentages
        split_percentages = (train_percent, test_percent, val_percent)
        total_percent = sum(split_percentages)

        if total_percent > 0:
            if total_percent > 1:
                raise ValueError("The total percentage cannot be greater than 100%.")
        else:
            raise ValueError("At least one percentage should be greater than 0.")

        augmentation_functions = []
        if augmentation_type:
            try:
                with open("config.json") as f:
                    config = json.load(f)
                augmentation_type = list(config.values())
            except Exception as e:
                print(f"Error loading config.json: {e}")

            augmentation_function_names = [
                k for k, v in config.items() if v in augmentation_type
            ]
            for augmentation_function_name in augmentation_function_names:
                augmentation_function = getattr(
                    augmentation, augmentation_function_name
                )
                augmentation_functions.append(augmentation_function)

            if not augmentation_functions:
                raise ValueError(
                    "No valid augmentation types found in the config.json file."
                )

        # Get the list of images
        images = self.get_image_files(input_folder)

        print("Total images:", len(images))  # Debug print

        # Apply the augmentation functions on the images in the input folder
        found_annotation = False
        # for _ in range(augment_size):
        for _ in range(augment_size or 1):

            for image in images:
                image_path = os.path.join(input_folder, image)
                annotation_file = os.path.splitext(image)[0] + ".xml"
                annotation_path = os.path.join(input_folder, annotation_file)

                for augmentation_function in augmentation_functions:
                    if not os.path.isfile(annotation_path):
                        print(f"No Annotation file found for {image}")
                        found_annotation = False
                        break
                    found_annotation = True
                    augmentation_function(image_path, annotation_path)
                if not found_annotation:
                    continue
                
                
        # After augmentation, update the list of images based on the updated files in the input folder
        images = self.get_image_files(input_folder)

        print("Images after augmentation:", len(images))
        num_images = len(images)
        # Calculate the number of images for each split
        num_train = int(train_percent * num_images)
        num_test = int(test_percent * num_images)
        num_val = num_images - num_train - num_test

        # Shuffle the list of images
        random.shuffle(images)

        # Initialize counters for the number of images in each split
        train_count = 0
        test_count = 0
        val_count = 0

        # Create the train, test, and validation directories
        train_dir = os.path.join(output_folder, "train")
        test_dir = os.path.join(output_folder, "test")
        val_dir = os.path.join(output_folder, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # Count the occurrences of each class
        class_counts = defaultdict(int)
        for image in images:
            annotation_file = os.path.splitext(image)[0] + ".xml"
            annotation_path = os.path.join(input_folder, annotation_file)
            if os.path.isfile(annotation_path):
                tree = ET.parse(annotation_path)
                root = tree.getroot()
                for obj in root.findall("object"):
                    class_name = obj.find("name").text
                    class_counts[class_name] += 1

        # Determine the number of images per class for each split
        class_num_train = defaultdict(int)
        class_num_test = defaultdict(int)
        class_num_val = defaultdict(int)
        for class_name, count in class_counts.items():
            num_class_train = int(train_percent * count)
            num_class_test = int(test_percent * count)
            num_class_val = count - num_class_train - num_class_test

            class_num_train[class_name] = num_class_train
            class_num_test[class_name] = num_class_test
            class_num_val[class_name] = num_class_val

        # Move or copy the images to the train, test, and validation directories
        dst_annotation_files = []
        for image in images:
            # Check if the image has an annotation file
            image_name, image_ext = os.path.splitext(image)
            annotation_file = image_name + ".xml"
            annotation_path = os.path.join(input_folder, annotation_file)

            if not os.path.isfile(annotation_path):
                print(f"No annotation file found for image: {image}")
                continue

            # Parse the annotation file to get the class names
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            class_names = []
            for obj in root.findall("object"):
                class_name = obj.find("name").text
                class_names.append(class_name)

            # Determine the destination directory based on the split and classes
            dst_annotation_file = os.path.join(val_dir, annotation_file)
            class_assigned = False
            # import pdb;pdb.set_trace();
            for class_name in class_names:
                if not class_assigned:
                    if train_count < num_train and class_num_train[class_name] > 0:
                        dst = os.path.join(train_dir, image)
                        dst_annotation_file = os.path.join(train_dir, annotation_file)
                        train_count += 1
                        class_num_train[class_name] -= 1
                        class_assigned = True
                    elif test_count < num_test and class_num_test[class_name] > 0:
                        dst = os.path.join(test_dir, image)
                        dst_annotation_file = os.path.join(test_dir, annotation_file)
                        test_count += 1
                        class_num_test[class_name] -= 1
                        class_assigned = True
                    elif val_count < num_val and class_num_val[class_name] > 0:
                        dst = os.path.join(val_dir, image)
                        dst_annotation_file = os.path.join(val_dir, annotation_file)
                        val_count += 1
                        class_num_val[class_name] -= 1
                        class_assigned = True

            # All splits are full or no more images of the class, move the image to the validation directory
            if not class_assigned:
                dst = os.path.join(val_dir, image)
                val_count += 1

            # Move or copy the image file and the corresponding annotation file to the destination directory
            if copy_images:
                shutil.copyfile(os.path.join(input_folder, image), dst)
                shutil.copyfile(annotation_path, dst_annotation_file)
            else:
                shutil.move(os.path.join(input_folder, image), dst)
                shutil.move(annotation_path, dst_annotation_file)

            dst_annotation_files.append(dst_annotation_file)

        # Print out some statistics
        print("Total images:", train_count + test_count + val_count)
        print("Training set:", train_count)
        print("Testing set:", test_count)
        print("Validation set:", val_count)
        # import pdb;pdb.set_trace()
        # Call the process_values function from split_voc.py
        process_values(train_dir, test_dir, val_dir)

        converter = DataConverter()
        voc_folder_names = ["VOCdevkit_train/", "VOCdevkit_test/", "VOCdevkit_val/"]
        data_dirs = [
            output_folder + "/" + folder_name for folder_name in voc_folder_names
        ]

        coco_folder_names = ["coco/", "coco/", "coco/"]
        output_dirs = [
            output_folder + "/" + folder_name for folder_name in coco_folder_names
        ]

        for input_path, output_path in zip(data_dirs, output_dirs):
            input_type = "voc"
            output_type = "coco"

            if input_path and output_path and input_type and output_type:
                converter.convert(
                    input_path=input_path,
                    output_path=output_path,
                    input_type=input_type,
                    output_type=output_type,
                )
            else:
                print("Please pass all the required arguments")

        return dst_annotation_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split images and move or copy them to train, test, and validation directories"
    )
    parser.add_argument(
        "--input-folder", help="path to the input folder", required=True
    )
    parser.add_argument(
        "--output-folder", help="path to the output folder", required=True
    )
    parser.add_argument(
        "--copy-images",
        help="whether to copy the images instead of moving them (default: False)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--split-percentages",
        help="percentages of images for the train, test, and validation sets (default: 0.8 0.1 0.1)",
        nargs=3,
        type=float,
        default=[0.8, 0.1, 0.1],
    )
    parser.add_argument(
        "--aug-type", help="type of augmentation (default: None)", action="store_true"
    )
    parser.add_argument(
    "--augment-size",
    type=int,
    help="Number of times to augment each image.",
    )
    

    args = parser.parse_args()

    splitter = ImageSplitter()
    dst_annotation_files = splitter.split_images(
        args.input_folder,
        args.output_folder,
        args.copy_images,
        args.split_percentages,
        args.aug_type,
        args.augment_size,
        
    )

    print("Annotation files moved/copied to the following directories:")
    for annotation_file in dst_annotation_files:
        print(annotation_file)




'''
# adding --augment-size in this original code 
import os
import random
import shutil
import argparse
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from split_voc import process_values
from coco_converter.data_converter import DataConverter
import augmentation

class ImageSplitter:
    def __init__(self):
        random.seed(42)

    def get_image_files(self, folder):
        return [
            f
            for f in os.listdir(folder)
            if f.endswith((".jpg", ".JPG", ".png", ".PNG", ".bmp", ".BMP"))
        ]

    def split_images(
        self,
        input_folder,
        output_folder,
        copy_images,
        split_percentages,
        augmentation_type,
    ):

        train_percent, test_percent, val_percent = split_percentages
        split_percentages = (train_percent, test_percent, val_percent)
        total_percent = sum(split_percentages)

        if total_percent > 0:
            if total_percent > 1:
                raise ValueError("The total percentage cannot be greater than 100%.")
        else:
            raise ValueError("At least one percentage should be greater than 0.")

        augmentation_functions = []
        if augmentation_type:
            try:
                with open("config.json") as f:
                    config = json.load(f)
                augmentation_type = list(config.values())
            except Exception as e:
                print(f"Error loading config.json: {e}")

            augmentation_function_names = [
                k for k, v in config.items() if v in augmentation_type
            ]
            for augmentation_function_name in augmentation_function_names:
                augmentation_function = getattr(
                    augmentation, augmentation_function_name
                )
                augmentation_functions.append(augmentation_function)

            if not augmentation_functions:
                raise ValueError(
                    "No valid augmentation types found in the config.json file."
                )

        # Get the list of images
        images = self.get_image_files(input_folder)

        print("Total images:", len(images))  # Debug print

        # Apply the augmentation functions on the images in the input folder
        found_annotation= False
        for image in images:
            image_path = os.path.join(input_folder, image)
            annotation_file = os.path.splitext(image)[0] + ".xml"
            annotation_path = os.path.join(input_folder, annotation_file)
        
            for augmentation_function in augmentation_functions:
                if not os.path.isfile(annotation_path):
                    print(f"No Annotation file found for{image}")
                    found_annotation = False
                    break 
                found_annotation = True
                augmentation_function(image_path, annotation_path)
            if not found_annotation:
                continue
                
                
        # After augmentation, update the list of images based on the updated files in the input folder
        images = self.get_image_files(input_folder)

        print("Images after augmentation:", len(images))
        num_images = len(images)
        # Calculate the number of images for each split
        num_train = int(train_percent * num_images)
        num_test = int(test_percent * num_images)
        num_val = num_images - num_train - num_test

        # Shuffle the list of images
        random.shuffle(images)

        # Initialize counters for the number of images in each split
        train_count = 0
        test_count = 0
        val_count = 0

        # Create the train, test, and validation directories
        train_dir = os.path.join(output_folder, "train")
        test_dir = os.path.join(output_folder, "test")
        val_dir = os.path.join(output_folder, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # Count the occurrences of each class
        class_counts = defaultdict(int)
        for image in images:
            annotation_file = os.path.splitext(image)[0] + ".xml"
            annotation_path = os.path.join(input_folder, annotation_file)
            if os.path.isfile(annotation_path):
                tree = ET.parse(annotation_path)
                root = tree.getroot()
                for obj in root.findall("object"):
                    class_name = obj.find("name").text
                    class_counts[class_name] += 1

        # Determine the number of images per class for each split
        class_num_train = defaultdict(int)
        class_num_test = defaultdict(int)
        class_num_val = defaultdict(int)
        for class_name, count in class_counts.items():
            num_class_train = int(train_percent * count)
            num_class_test = int(test_percent * count)
            num_class_val = count - num_class_train - num_class_test

            class_num_train[class_name] = num_class_train
            class_num_test[class_name] = num_class_test
            class_num_val[class_name] = num_class_val

        # Move or copy the images to the train, test, and validation directories
        dst_annotation_files = []
        for image in images:
            # Check if the image has an annotation file
            image_name, image_ext = os.path.splitext(image)
            annotation_file = image_name + ".xml"
            annotation_path = os.path.join(input_folder, annotation_file)

            if not os.path.isfile(annotation_path):
                print(f"No annotation file found for image: {image}")
                continue

            # Parse the annotation file to get the class names
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            class_names = []
            for obj in root.findall("object"):
                class_name = obj.find("name").text
                class_names.append(class_name)

            # Determine the destination directory based on the split and classes
            dst_annotation_file = os.path.join(val_dir, annotation_file)
            class_assigned = False
            # import pdb;pdb.set_trace();
            for class_name in class_names:
                if not class_assigned:
                    if train_count < num_train and class_num_train[class_name] > 0:
                        dst = os.path.join(train_dir, image)
                        dst_annotation_file = os.path.join(train_dir, annotation_file)
                        train_count += 1
                        class_num_train[class_name] -= 1
                        class_assigned = True
                    elif test_count < num_test and class_num_test[class_name] > 0:
                        dst = os.path.join(test_dir, image)
                        dst_annotation_file = os.path.join(test_dir, annotation_file)
                        test_count += 1
                        class_num_test[class_name] -= 1
                        class_assigned = True
                    elif val_count < num_val and class_num_val[class_name] > 0:
                        dst = os.path.join(val_dir, image)
                        dst_annotation_file = os.path.join(val_dir, annotation_file)
                        val_count += 1
                        class_num_val[class_name] -= 1
                        class_assigned = True

            # All splits are full or no more images of the class, move the image to the validation directory
            if not class_assigned:
                dst = os.path.join(val_dir, image)
                val_count += 1

            # Move or copy the image file and the corresponding annotation file to the destination directory
            if copy_images:
                shutil.copyfile(os.path.join(input_folder, image), dst)
                shutil.copyfile(annotation_path, dst_annotation_file)
            else:
                shutil.move(os.path.join(input_folder, image), dst)
                shutil.move(annotation_path, dst_annotation_file)

            dst_annotation_files.append(dst_annotation_file)

        # Print out some statistics
        print("Total images:", train_count + test_count + val_count)
        print("Training set:", train_count)
        print("Testing set:", test_count)
        print("Validation set:", val_count)
        import pdb;pdb.set_trace()
        # Call the process_values function from split_voc.py
        process_values(train_dir, test_dir, val_dir)

        converter = DataConverter()
        voc_folder_names = ["VOCdevkit_train/", "VOCdevkit_test/", "VOCdevkit_val/"]
        data_dirs = [
            output_folder + "/" + folder_name for folder_name in voc_folder_names
        ]

        coco_folder_names = ["coco/", "coco/", "coco/"]
        output_dirs = [
            output_folder + "/" + folder_name for folder_name in coco_folder_names
        ]

        for input_path, output_path in zip(data_dirs, output_dirs):
            input_type = "voc"
            output_type = "coco"

            if input_path and output_path and input_type and output_type:
                converter.convert(
                    input_path=input_path,
                    output_path=output_path,
                    input_type=input_type,
                    output_type=output_type,
                )
            else:
                print("Please pass all the required arguments")

        return dst_annotation_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split images and move or copy them to train, test, and validation directories"
    )
    parser.add_argument(
        "--input-folder", help="path to the input folder", required=True
    )
    parser.add_argument(
        "--output-folder", help="path to the output folder", required=True
    )
    parser.add_argument(
        "--copy-images",
        help="whether to copy the images instead of moving them (default: False)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--split-percentages",
        help="percentages of images for the train, test, and validation sets (default: 0.8 0.1 0.1)",
        nargs=3,
        type=float,
        default=[0.8, 0.1, 0.1],
    )
    parser.add_argument(
        "--aug-type", help="type of augmentation (default: None)", action="store_true"
    )

    args = parser.parse_args()

    splitter = ImageSplitter()
    dst_annotation_files = splitter.split_images(
        args.input_folder,
        args.output_folder,
        args.copy_images,
        args.split_percentages,
        args.aug_type,
    )

    print("Annotation files moved/copied to the following directories:")
    for annotation_file in dst_annotation_files:
        print(annotation_file)
'''


"""
import os
import random
import shutil
import argparse
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from split_voc import process_values
from coco_converter.data_converter import DataConverter
import augmentation

class ImageSplitter:
    def __init__(self):
        random.seed(42)

    def get_image_files(self, folder):
        return [
            f
            for f in os.listdir(folder)
            if f.endswith((".jpg", ".JPG", ".png", ".PNG", ".bmp", ".BMP"))
        ]

    def split_images(self, input_folder, output_folder, copy_images, split_percentages, augmentation_type):
       
        # Load the config.json file
        with open('config.json') as f:
            config = json.load(f)

        # Check if the augmentation type exists in the config file
        if augmentation_type in config:
            augmentation_function_name = config[augmentation_type]
            augmentation_function = getattr(augmentation, augmentation_function_name)
        else:
            raise ValueError(f"Augmentation type '{augmentation_type}' is not supported.")

        # Get the list of images
        images = self.get_image_files(input_folder)

        # Apply the augmentation function on the images in the input folder
        for image in images:
            image_path = os.path.join(input_folder, image)
            annotation_file = os.path.splitext(image)[0] + ".xml"
            annotation_path = os.path.join(input_folder, annotation_file)
            augmentation_function(image_path, annotation_path)

        # import pdb; pdb.set_trace();
        # After augmentation, update the list of images based on the updated files in the input folder
        images = self.get_image_files(input_folder)

        train_percent, test_percent, val_percent = split_percentages

        split_percentages = (train_percent, test_percent, val_percent)
        total_percent = sum(split_percentages)

        if total_percent > 0:
            if total_percent > 1:
                raise ValueError("The total percentage cannot be greater than 100%.")
        else:
            raise ValueError("At least one percentage should be greater than 0.")
      

        
        num_images = len(images)
        # Calculate the number of images for each split
        num_train = int(train_percent * num_images)
        num_test = int(test_percent * num_images)
        num_val = num_images - num_train - num_test

        # Shuffle the list of images
        random.shuffle(images)

        # Initialize counters for the number of images in each split
        train_count = 0
        test_count = 0
        val_count = 0

        # Create the train, test, and validation directories
        train_dir = os.path.join(output_folder, "train")
        test_dir = os.path.join(output_folder, "test")
        val_dir = os.path.join(output_folder, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # Count the occurrences of each class d : 2
        class_counts = defaultdict(int)#defaultdict(<class 'int'>, {'broken': 45, 'damage': 15, 'crack': 18, 'rust': 6})
        for image in images:
            annotation_file = os.path.splitext(image)[0] + ".xml"
            annotation_path = os.path.join(input_folder, annotation_file)
            if os.path.isfile(annotation_path):
                tree = ET.parse(annotation_path)
                root = tree.getroot()
                for obj in root.findall("object"):
                    class_name = obj.find("name").text
                    class_counts[class_name] += 1
        
        # Determine the number of images per class for each split
        class_num_train = defaultdict(int)
        class_num_test = defaultdict(int)
        class_num_val = defaultdict(int)
        for class_name, count in class_counts.items():
            num_class_train = int(train_percent * count)
            num_class_test = int(test_percent * count)
            num_class_val = count - num_class_train - num_class_test

            class_num_train[class_name] = num_class_train
            class_num_test[class_name] = num_class_test
            class_num_val[class_name] = num_class_val
        
        # Move or copy the images to the train, test, and validation directories
        dst_annotation_files = []
        for image in images:
            # Check if the image has an annotation file
            image_name, image_ext = os.path.splitext(image)
            annotation_file = image_name + ".xml"
            annotation_path = os.path.join(input_folder, annotation_file)

            if not os.path.isfile(annotation_path):
                print(f"No annotation file found for image: {image}")
                continue

            # Parse the annotation file to get the class names
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            class_names = []
            for obj in root.findall("object"):
                class_name = obj.find("name").text
                class_names.append(class_name)

            # Determine the destination directory based on the split and classes
            dst_annotation_file = os.path.join(val_dir, annotation_file)
            class_assigned = False
            # import pdb;pdb.set_trace();
            for class_name in class_names:
                if not class_assigned:
                    if train_count < num_train and class_num_train[class_name] > 0:
                        dst = os.path.join(train_dir, image)
                        dst_annotation_file = os.path.join(train_dir, annotation_file)
                        train_count += 1
                        class_num_train[class_name] -= 1
                        class_assigned = True
                    elif test_count < num_test and class_num_test[class_name] > 0:
                        dst = os.path.join(test_dir, image)
                        dst_annotation_file = os.path.join(test_dir, annotation_file)
                        test_count += 1
                        class_num_test[class_name] -= 1
                        class_assigned = True
                    elif val_count < num_val and class_num_val[class_name] > 0:
                        dst = os.path.join(val_dir, image)
                        dst_annotation_file = os.path.join(val_dir, annotation_file)
                        val_count += 1
                        class_num_val[class_name] -= 1
                        class_assigned = True

            # All splits are full or no more images of the class, move the image to the validation directory
            if not class_assigned:
                dst = os.path.join(val_dir, image)
                val_count += 1

            # Move or copy the image file and the corresponding annotation file to the destination directory
            if copy_images:
                shutil.copyfile(os.path.join(input_folder, image), dst)
                shutil.copyfile(annotation_path, dst_annotation_file)
            else:
                shutil.move(os.path.join(input_folder, image), dst)
                shutil.move(annotation_path, dst_annotation_file)

            dst_annotation_files.append(dst_annotation_file)

        # Print out some statistics
        print("Total images:", train_count + test_count + val_count)
        print("Training set:", train_count)
        print("Testing set:", test_count)
        print("Validation set:", val_count)
        import pdb; pdb.set_trace();
        # Call the process_values function from split_voc.py
        process_values(train_dir, test_dir, val_dir)

        converter = DataConverter()
        voc_folder_names = ["VOCdevkit_train/", "VOCdevkit_test/", "VOCdevkit_val/"]
        data_dirs = [output_folder + "/" + folder_name for folder_name in voc_folder_names]

        coco_folder_names = ["coco/", "coco/", "coco/"]
        output_dirs = [output_folder + "/" + folder_name for folder_name in coco_folder_names]

        for input_path, output_path in zip(data_dirs, output_dirs):
            input_type = "voc"
            output_type = "coco"


            if input_path and output_path and input_type and output_type:
                converter.convert(
                    input_path=input_path,
                    output_path=output_path,
                    input_type=input_type,
                    output_type=output_type,
                )
            else:
                print("Please pass all the required arguments")

        return dst_annotation_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split images and move or copy them to train, test, and validation directories"
    )
    parser.add_argument("--input-folder", help="path to the input folder", required=True)
    parser.add_argument("--output-folder", help="path to the output folder", required=True)
    parser.add_argument(
        "--copy-images",
        help="whether to copy the images instead of moving them (default: False)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--split-percentages",
        help="percentages of images for the train, test, and validation sets (default: 0.8 0.1 0.1)",
        nargs=3,
        type=float,
        default=[0.8, 0.1, 0.1],
    )
    parser.add_argument("--aug-type", help="type of augmentation (default: None)", choices=['flip_horizontal', 'flip_vertical', 'rotate'])

    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    copy_images = args.copy_images
    split_percentages = args.split_percentages
    augmentation_type = args.aug_type

    splitter = ImageSplitter()
    dst_annotation_files = splitter.split_images(input_folder, output_folder, copy_images, split_percentages, augmentation_type)

    print("Annotation files moved/copied to the following directories:")
    for annotation_file in dst_annotation_files:
        print(annotation_file)

"""


"""
import os
import random
import shutil
import argparse

from split_voc import process_values
from coco_converter.data_converter import DataConverter


class ImageSplitter:
    def __init__(self):
        random.seed(42)

    def split_images(self, input_folder, output_folder, copy_images, split_percentages):
        train_percent, test_percent, val_percent = split_percentages

        total_percent = train_percent + test_percent + val_percent
        # Unpack the split_percentages tuple
        if total_percent > 1:
            raise ValueError("The total percentage cannot be greater than 100%.")

        # Get a list of all the image files in the source directory
        images = [
            f
            for f in os.listdir(input_folder)
            if f.endswith((".jpg", ".JPG", ".png", ".PNG", ".bmp", ".BMP"))
        ]

        # Get the total number of images
        num_images = len(images)

        # Calculate the number of images for each split
        num_train = int(train_percent * num_images)
        num_test = int(test_percent * num_images)
        num_val = num_images - num_train - num_test

        # Shuffle the list of images
        random.shuffle(images)

        # Initialize counters for the number of images in each split
        train_count = 0
        test_count = 0
        val_count = 0

        # Create the train, test, and validation directories
        train_dir = os.path.join(output_folder, "train")
        test_dir = os.path.join(output_folder, "test")
        val_dir = os.path.join(output_folder, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # Move or copy the images to the train, test, and validation directories
        for image in images:
            # Check if the image has an annotation file
            image_name, image_ext = os.path.splitext(image)
            annotation_file = (
                image_name + ".xml"
                if os.path.isfile(os.path.join(input_folder, image_name + ".xml"))
                else image_name + ".txt"
                if os.path.isfile(os.path.join(input_folder, image_name + ".txt"))
                else None
            )

            if annotation_file is None:
                print(f"No annotation file found for image: {image}")
                continue

            # Determine the destination directory based on the split
            if train_count < num_train:
                dst = os.path.join(train_dir, image)
                dst_annotation_file = os.path.join(train_dir, annotation_file)
                train_count += 1
            elif test_count < num_test:
                dst = os.path.join(test_dir, image)
                dst_annotation_file = os.path.join(test_dir, annotation_file)
                test_count += 1
            elif val_count < num_val:
                dst = os.path.join(val_dir, image)
                dst_annotation_file = os.path.join(val_dir, annotation_file)
                val_count += 1
            else:
                # All splits are full, move the image to the validation directory
                dst = os.path.join(val_dir, image)
                dst_annotation_file = os.path.join(val_dir, annotation_file)
                val_count += 1

            # Move or copy the image file and the corresponding annotation file to the destination directory
            if copy_images:
                shutil.copyfile(os.path.join(input_folder, image), dst)
                shutil.copyfile(
                    os.path.join(input_folder, annotation_file), dst_annotation_file
                )
            else:
                shutil.move(os.path.join(input_folder, image), dst)
                shutil.move(
                    os.path.join(input_folder, annotation_file), dst_annotation_file
                )

        # Print out some statistics
        print("Total images:", train_count + test_count + val_count)
        print("Training set:", train_count)
        print("Testing set:", test_count)
        print("Validation set:", val_count)
        # Call the process_values function from split_voc.py
        process_values(train_dir, test_dir, val_dir)

        converter = DataConverter()
        voc_folder_names = ["VOCdevkit_train/", "VOCdevkit_test/", "VOCdevkit_val/"]
        data_dirs = [
            output_folder + "/" + folder_name for folder_name in voc_folder_names
        ]

        # coco_folder_names = ['coco_train/', 'coco_test/', 'coco_val']
        coco_folder_names = ["coco/", "coco/", "coco/"]
        output_dirs = [
            output_folder + "/" + folder_name for folder_name in coco_folder_names
        ]

        for input_path, output_path in zip(data_dirs, output_dirs):
            input_type = "voc"
            output_type = "coco"

            if input_path and output_path and input_type and output_type:
                converter.convert(
                    input_path=input_path,
                    output_path=output_path,
                    input_type=input_type,
                    output_type=output_type,
                )
            else:
                print("Please pass all the required arguments")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split images and move or copy them to train, test, and validation directories"
    )
    parser.add_argument("--input-folder", type=str, help="Path to the source directory")
    parser.add_argument(
        "--output-folder", type=str, help="Path to the destination directory"
    )
    parser.add_argument(
        "--copy", action="store_true", help="Copy images instead of moving them"
    )
    parser.add_argument(
        "--split-percentages",
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        help="Percentages of images to use for the training, testing, and validation sets (default: 0.8 0.1 0.1)",
    )
    args = parser.parse_args()
    splitter = ImageSplitter()
    splitter.split_images(
        args.input_folder, args.output_folder, args.copy, args.split_percentages
    )

"""
