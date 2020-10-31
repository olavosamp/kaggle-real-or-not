dataset_path = "../dataset/"
target_column_name = "class_target"
experiment_path = "../results/"

def create_folder(path):
    import os
    if not os.path.isdir(path):
        os.makedirs(path)
