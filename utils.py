def extract_file_name(data_dir, image_path):
    return image_path.split(data_dir)[1].split('.')[0]