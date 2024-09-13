
'''
import os
import random
import shutil
import argparse
# from split_voc import process_values


class ImageSplitter:
    def __init__(self):
        random.seed(42)
        
    def split_images(self,input_folder, output_folder, copy_images, split_percentages):
        train_percent, test_percent, val_percent = split_percentages

        total_percent = train_percent + test_percent + val_percent
        # Unpack the split_percentages tuple
        if total_percent > 1:
            raise ValueError("The total percentage cannot be greater than 100%.")

        # Get a list of all the image files in the source directory
        images = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.JPG', '.png', '.PNG', '.bmp', '.BMP'))]

        # Get the total number of images
        num_images = len(images)

        # Calculate the number of images for each split
        num_train = int(train_percent * num_images)
        num_test = int(test_percent * num_images)
        num_val = num_images - num_train - num_test
        #num_val = int(val_percent * num_images)
        #num_test = num_images - num_train - num_val

        # Shuffle the list of images
        random.shuffle(images)

        # Initialize counters for the number of images in each split
        train_count = 0
        test_count = 0
        val_count = 0

        # Create the train, test, and validation directories
        train_dir = os.path.join(output_folder, 'train')
        test_dir = os.path.join(output_folder, 'test')
        val_dir = os.path.join(output_folder, 'val')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # Move or copy the images to the train, test, and validation directories
        for image in images:
            # Check if the image has an annotation file
            image_name, image_ext = os.path.splitext(image)
            annotation_file = image_name + '.xml' if os.path.isfile(os.path.join(input_folder, image_name + '.xml')) else image_name + '.txt' if os.path.isfile(os.path.join(input_folder, image_name + '.txt')) else None

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
                shutil.copyfile(os.path.join(input_folder, annotation_file), dst_annotation_file)
            else:
                shutil.move(os.path.join(input_folder, image), dst)
                shutil.move(os.path.join(input_folder, annotation_file), dst_annotation_file)

        # Print the number of images in each split
        print("Total images:", len(images))
        print(f"Training set: {train_count}")
        print(f"Testing set: {test_count}")
        print(f"Validation set: {val_count}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split images and move or copy them to train, test, and validation directories')
    parser.add_argument('--input-folder', type=str, help='Path to the source directory')
    parser.add_argument('--output-folder', type=str, help='Path to the destination directory')
    parser.add_argument('--copy', action='store_true', help='Copy images instead of moving them')
    parser.add_argument('--split-percentages', type=float, nargs=3, default=[0.8, 0.1, 0.1], help='Percentages of images to use for the training, testing, and validation sets (default: 0.8 0.1 0.1)')
    args = parser.parse_args()
    # split_images(args.input_folder, args.output_folder, args.copy, args.split_percentages)
    splitter = ImageSplitter()
    splitter.split_images(args.input_folder, args.output_folder, args.copy, args.split_percentages)
'''