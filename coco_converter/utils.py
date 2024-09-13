import os


def file_exists(file_, image_dirs):
    file_path = None
    for image_dir in image_dirs:
        path = os.path.join(image_dir, file_)
        if os.path.exists(path):
            file_path = path
            break

    return file_path
