#################################################################################
# pc_completion_make_data.py
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

print "============================================================"
print "Starting Program..."

import cv2
import numpy as np
# import tensorflow as tf
# from tensorflow.python.framework import dtypes

import datetime
import random

import file_util
from pc_completion_util import *
from depth_map_generator import *

image_size = 256 # downscaled to half later
scaling = 0.0001
images_per_obj = 256 # = 256 train, 64 test, 8 val # <---------------- change this for datasets

dataset_folder = "../Data/raw_data/caesar-norm-wsx-fitted-meshes_foot_mesh_BAM/"
all_objects = file_util.read_all_lines("nl_train_legs.txt") # <------- change this for datasets
all_objects.sort()
print "Total Num Obj", len(all_objects)

png_folder = dataset_folder + "../../caesar-norm-wsx_pngs/"
file_util.make_directory(png_folder)
dataset_name = "pc_completion_"
prefix = "train" # <-------------------------------------------------- change this for datasets
train_images_folder = png_folder + dataset_name + prefix + "_pngs/"
file_util.make_directory(train_images_folder)

im_generator = Depth_Generator(image_size, scaling)

start_time = datetime.datetime.now()

for i in range(len(all_objects)):
    name = [all_objects[i] + ".bam"]

    if not file_util.file_exists(dataset_folder + name[0]):
        continue

    print i, name, datetime.datetime.now() - start_time

    obj_folder = train_images_folder + "/" + name[0][:-4] + "/"
    file_util.make_directory(obj_folder)

    im_generator.load_all_objects(name, dataset_folder)

    for j in range(images_per_obj):
        d_images, d_images_jet, extrins = im_generator.generate_images(name, dataset_type = "depth_xray")

        cv2.imwrite(obj_folder + prefix + "_in_im_" + str(j) + ".png", reproject_to128(d_images[0][0]))
        save_array(obj_folder + prefix + "_in_extrin_" + str(j) + ".txt", extrinsics_to_net_rot(extrins[0][0]))
        cv2.imwrite(obj_folder + prefix + "_gt_im_" + str(j) + ".png", reproject_dm_extrinsic_to128(d_images[1][0], extrins[0][0], extrins[1][0]))

    im_generator.unload_all_objects()

print "Total Runtime:", datetime.datetime.now() - start_time

print "============================================================"
print "Ending Program..."
