#################################################################################
# file_util.py
#
# Author: Nolan Lunscher
#
# All code is provided for research purposes only and without any warranty. 
# Any commercial use requires our consent. 
# When using the code in your research work, please cite the following paper:
#     @InProceedings{Lunscher_2017_ICCV_Workshops,
#     author = {Lunscher, Nolan and Zelek, John},
#     title = {Point Cloud Completion of Foot Shape From a Single Depth Map for Fit Matching Using Deep Learning View Synthesis},
#     booktitle = {The IEEE International Conference on Computer Vision (ICCV) Workshops},
#     month = {Oct},
#     year = {2017}
#     }
#################################################################################

# file utility
import sys,os
import shutil

def write_to_file(filename, text):
    with open(filename, 'w') as f:
        f.write(str(text))

def append_to_file(filename, text):
    with open(filename, 'a') as f:
        f.write(str(text) + '\n')

def append_line_to_csv(filename, text):
    with open(filename, 'a') as f:
        f.write(str(text) + ',\n')

def get_foldernames(directory = "."):
    full_dir = os.listdir(directory)
    folders = [name for name in full_dir if os.path.isdir(directory + "/" + name)]
    folders.sort()
    return folders

def get_in_dir(directory = "."):
    full_dir = os.listdir(directory)
    # full_dir.sort()
    return full_dir

def folder_exists(directory = "."):
    return os.path.isdir(directory)

def file_exists(filename):
    return os.path.isfile(filename) 

def make_directory(directory = "new_folder"):
    if not folder_exists(directory):
        os.makedirs(directory)
    return folder_exists(directory)

def get_objs_in_folder(directory = "."):
    folders = get_foldernames(directory)    
    objs = [directory + "/" + name + "/models/model_normalized.obj" for name in folders]

    objs.sort()
    return objs

def read_all_lines(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    for l in range(len(lines)):
        lines[l] = lines[l].replace("\n", "")

    return lines

def get_next_availiable_filename(filename):
    name, extension = os.path.splitext(filename)
    filename_num = 1
    while filename_num < 1000:
        try_name = name + str(filename_num) + extension
        if not file_exists(try_name):
            break
        else:
            filename_num += 1

    return try_name

def copy_as_backup(filename):
    name, extension = os.path.splitext(filename)
    backup_name = name + "_backup" + extension
    shutil.copyfile(filename, get_next_availiable_filename(backup_name))

def copy_folder(foldername, destination_folder):
    shutil.copytree(foldername, destination_folder)

def move_folder(foldername, destination_folder):
    shutil.move(foldername, destination_folder)
