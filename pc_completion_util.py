
#################################################################################
# pc_completion_util.py
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

# depthmap reader/utility

import tensorflow as tf
import numpy as np
import random
import cv2
from math import pi, sin, cos

import file_util


K1 = np.array([[192.0*2, 0.0,   64.0*2],
               [0.0,   192.0*2, 64.0*2], 
               [0.0 ,  0.0,   1.0]]);
K2 = np.array([[192.0, 0.0,   64.0],
              [0.0,   192.0, 64.0], 
              [0.0 ,  0.0,   1.0]]);
x128, y128 = np.meshgrid(np.arange(128), np.arange(128))
x128 = np.reshape(x128,128*128)
y128 = np.reshape(y128,128*128)
im_p = np.array([x128, y128, np.ones(128*128)])

x256, y256 = np.meshgrid(np.arange(256), np.arange(256))
x256 = np.reshape(x256,256*256)
y256 = np.reshape(y256,256*256)
im_p1 = np.array([x256, y256, np.ones(256*256)])

############################################ Utility Functions
def rollback_losses_csv(file_name, current_itter):
    file_util.copy_as_backup(file_name)
    
    lines = file_util.read_all_lines(file_name)
    new_file = ""
    for i in range(len(lines)):
        line_split = lines[i].split(',')
        if int(line_split[0]) < current_itter:
            new_file += lines[i] + "\n"
    file_util.write_to_file(file_name, new_file)

def read_png16(file_name):
    CV_LOAD_IMAGE_ANYDEPTH = 2 # apparently leftover from cv before cv2
    int16_d_image = cv2.imread(file_name,  CV_LOAD_IMAGE_ANYDEPTH)
    return int16_d_image

def depthmap_16bitTofloat(int16_d_image):
    float_d_image = np.array(int16_d_image, dtype=np.float32)
    float_d_image = (float_d_image  / (2**16 - 1)  - 0.5) * 1.5
    im_shape = float_d_image.shape
    return float_d_image.reshape((im_shape[0], im_shape[1], 1))

def depthmap_floatTo16bit(float_d_image):
    int16_d_image = (float_d_image / 1.5 + 0.5) * (2**16 - 1) 
    int16_d_image[np.where(int16_d_image > 60000)] = 2**16-1
    int16_d_image[np.where(int16_d_image < 0.0)] = 0
    return int16_d_image.astype(np.uint16)

def save_iteration_images(folder, id, in_im, out_gt, out_im, rot, dataset_type = "Train"):
    if dataset_type == "Train" or dataset_type == "Test":
        # only sometimes save the input
        u_in_im = depthmap_floatTo16bit(in_im)
        u_out_gt = depthmap_floatTo16bit(out_gt)
        cv2.imwrite(folder + "in_im_" + str(id) + ".png", u_in_im)
        cv2.imwrite(folder + "gt_im_" + str(id) + ".png", u_out_gt)
        save_array(folder + "out_extrin_" + str(id) + ".txt", rot)
    # always save the output
    u_out_im = depthmap_floatTo16bit(out_im)
    tile3_im = make_3tile_image(in_im, out_gt, out_im)
    cv2.imwrite(folder + "out_im_" + str(id) + ".png", u_out_im)
    cv2.imwrite(folder + "tile3_im_" + str(id) + ".png", tile3_im)

def save_array(file_name, array_list):
    string = ""
    for i in range(len(array_list)):
        string += str(array_list[i])
        if i < len(array_list) - 1:
            string += "\n"
    file_util.write_to_file(file_name, string)

def extrinsics_to_net_rot(extrinsics):
    rot = np.zeros(6) # azimuth, elevation, roll, radius, pitch, heading
    # normalize to 0 - 1
    rot[0] = extrinsics[0] / 360.0
    rot[1] = (extrinsics[1] + 90) / 180.0
    rot[2] = extrinsics[2] / 90.0
    rot[3] = extrinsics[3] / 3.0
    rot[4] = (extrinsics[4] + 5) / 10.0
    rot[5] = (extrinsics[5] + 5) / 10.0
    return rot

def net_rot_to_extrinsics(rot):
    extrinsics = np.zeros(6) # azimuth, elevation, roll, radius, pitch, heading
    # normalize to 0 - 1
    extrinsics[0] = rot[0] * 360.0
    extrinsics[1] = rot[1] * 180.0 -90
    extrinsics[2] = rot[2] * 90.0
    extrinsics[3] = rot[3] * 3.0
    extrinsics[4] = rot[4] * 10.0 - 5
    extrinsics[5] = rot[5] * 10.0 - 5
    return extrinsics

def extrinsics_to_otherside(extrinsics):
    extrin_other = np.zeros(6) # azimuth, elevation, roll, radius, pitch, heading

    extrin_other[0] = (extrinsics[0] + 180) % 360
    extrin_other[1] = extrinsics[1] * -1
    extrin_other[2] = extrinsics[2]
    extrin_other[3] = extrinsics[3]
    extrin_other[4] = extrinsics[4] # not actually doing anything
    extrin_other[5] = extrinsics[5] # not actually doing anything
    return extrin_other

def dm2points(dm):
    dm_flat = np.reshape(dm,128*128)

    p_world = np.dot(np.linalg.inv(K2), im_p) * dm_flat;
    return p_world

def surfnorm_fast(X,Y,Z):
    nz = np.empty((128,128))

    dzdy = np.zeros((128,128))
    dzdy[1:-1,1:-1] = (Z[2:,1:-1] - Z[:-2,1:-1:]) / (Y[2:,1:-1] - Y[:-2,1:-1])
    dzdx = np.zeros((128,128))
    dzdx[1:-1,1:-1] = (Z[1:-1,2:] - Z[1:-1,:-2]) / (X[1:-1,2:] - X[1:-1,:-2])

    nz = 1/np.sqrt((-dzdy)**2 + (-dzdx)**2 + 1)

    angles = np.arccos(nz)
    return nz, angles

def noise_std(z_val, theta_val):
    z_component = 0.0012 + 0.0019 * (z_val - 0.4)**2
    theta_component = (0.0001/np.sqrt(z_val)) * (theta_val**2/(np.pi/2 - theta_val)**2)

    std = z_component + theta_component
    return std

def add_noise(dm_m, Z, angles):
    angles[angles >= np.pi/2] = np.pi/2 - 0.0001 # prevent divide by 0
    std = noise_std(Z, angles)
    dm_m_noise = dm_m + np.random.normal(np.zeros((128,128)), std)

    mask = np.logical_and(Z < 2, angles < 87.5 * np.pi/180)
    dm_m_noise = dm_m_noise * mask + np.logical_not(mask) * np.max(dm_m)

    return dm_m_noise

def add_kinect_v1_noise(dm): # operates on the uint16
    scale = 0.003
    dm_m = dm.astype(float) * 0.0001 / scale / 1000

    p_world = dm2points(dm_m)
    X = np.reshape(p_world[0,:], (128,128))
    Y = np.reshape(p_world[1,:], (128,128))
    Z = np.reshape(p_world[2,:], (128,128))

    if np.min(Z) <= 0.01:
        print "ZERO?"

    nz, angles = surfnorm_fast(X,Y,Z)

    dm_m_noise = add_noise(dm_m, Z, angles)

    return (dm_m_noise * 1000 * scale / 0.0001).astype(np.uint16)

def reproject_to128(dm):
    dm_new = cv2.resize(dm, (128,128), interpolation = cv2.INTER_NEAREST)
    return dm_new.astype(np.uint16)

def reproject_dm_extrinsic_to128(dm, extrin_in, extrin_out):
    z_scale = 0.0001

    dm_new = np.zeros((128,128))

    RT1 = camera_extrinsic_to_RT(extrin_in)
    RT2 = camera_extrinsic_to_RT(extrin_out)

    im1_flat = np.reshape(dm,256*256)

    p_world_1 = np.dot(np.linalg.inv(K1), im_p1) * im1_flat * z_scale;

    p_world_1_5 = np.transpose(np.transpose(np.dot(RT1[:,:3], p_world_1)) + RT1[:,3])

    p_world_2 = np.dot(np.transpose(RT2[:,:3]), np.transpose(np.transpose(p_world_1_5) - RT2[:,3]))

    im_p2 = np.dot(K2, p_world_2)
    im_p2_z = im_p2[2,:]
    im_p2_x = im_p2[0,:] / im_p2_z
    im_p2_y = im_p2[1,:] / im_p2_z

    indx_x = np.logical_and(im_p2_x >= 0, im_p2_x < 128)
    indx_y = np.logical_and(im_p2_y >= 0, im_p2_y < 128)
    indx = np.logical_and(indx_x, indx_y)

    im_p2_z = (im_p2_z[indx] / z_scale).astype(int)
    im_p2_x = (im_p2_x[indx]).astype(int)
    im_p2_y = (im_p2_y[indx]).astype(int)

    for i in range(im_p2_z.shape[0]):
        x_2 = im_p2_x[i]
        y_2 = im_p2_y[i]
        z_2 = im_p2_z[i]
        if y_2 < 128 and y_2 >= 0 and x_2 < 128 and x_2 >= 0:
            if z_2 > dm_new[y_2,x_2]:
                dm_new[y_2,x_2] = z_2

    dm_new[np.where(dm_new <= 0)] = 2**16-1
    return dm_new.astype(np.uint16)

def depthmap_more_visible(depth_image, scaling = 0.0001):
    # make it scaled better, in a uint8 image
    
    # radius was 2 | farthest point seen is at 3, closest at 1
    farthest = (3/scaling/ (2**16 - 1)  - 0.5) * 1.5
    closest = (1/scaling/ (2**16 - 1)  - 0.5) * 1.5
    dist = farthest - closest

    d_im = depth_image - closest
    d_im[np.where(d_im >= dist)] = dist

    d_im = d_im / np.max(d_im) * 255

    return d_im.astype(np.uint8)

def diff_more_visible(diff_image, scaling = 0.0001):
    farthest = np.max(diff_image)
    closest = np.min(diff_image)

    d_im = diff_image - closest
    d_im = d_im / np.max(d_im) * 255

    return d_im.astype(np.uint8)

def depth_2jet(depth_image, scaling = 0.0001):
    # requires input to be a uint8 image
    return cv2.applyColorMap(depthmap_more_visible(depth_image,scaling), cv2.COLORMAP_JET)

def diff_2jet(diff_image, scaling = 0.0001):
    return cv2.applyColorMap(diff_more_visible(diff_image,scaling), cv2.COLORMAP_JET)

def read_dataset_objects(dataset_filename):
    # reads a file containing all the object numbers/keys 
    # returns a list of the object numbers (but as list of strings)
    csv_lines = file_util.read_all_lines(dataset_filename)
    num_files = len(csv_lines)
    print "Number of files:", num_files

    objects = []
    for line in csv_lines:
        line_split = line.split(',')
        object_num = line_split[0]
        # object_name = line_split[1]

        if object_num not in objects:
            objects.append(object_num)

    return objects

def make_3tile_image(in_im, out_gt, out_im):
    show_image = np.concatenate([depth_2jet(in_im), 
                                    depth_2jet(out_gt),
                                    depth_2jet(out_im)], axis=1)
    return show_image

def make_5tile_image(in_im, out_gt, out_im):
    in_diff = (in_im - out_gt)
    out_diff = (out_im - out_gt)
    show_image = np.concatenate([depth_2jet(in_im), diff_2jet(in_diff),
                                    depth_2jet(out_gt),
                                    depth_2jet(out_im), diff_2jet(out_diff)]
                                    , axis=1)
    return show_image

def show_3_images(in_im, out_gt, out_im, wait_time = 0, name = "Depthmaps"):
    show_image = make_3tile_image(in_im, out_gt, out_im)
    cv2.imshow(name, show_image)
    key = cv2.waitKey(wait_time)

def save_3_images(folder, id, in_im, out_gt, out_im, new_scaling):
    u_in_im = undo_preprocess(in_im, new_scaling)
    u_out_gt = undo_preprocess(out_gt, new_scaling)
    u_out_im = undo_preprocess(out_im, new_scaling)
    tile3_im = make_3tile_image(in_im, out_gt, out_im)
    cv2.imwrite(folder + "in_im_" + str(id) + ".png", u_in_im)
    cv2.imwrite(folder + "gt_im_" + str(id) + ".png", u_out_gt)
    cv2.imwrite(folder + "out_im_" + str(id) + ".png", u_out_im)
    cv2.imwrite(folder + "tile3_im_" + str(id) + ".png", tile3_im)

def undo_preprocess(image, scaling):
    original = (image / 1.5 + 0.5) * (2**16-1) / scaling
    original[np.where(original >= 40000.0)] = 2**16-1
    return original.astype(np.uint16)

def camera_extrinsic_to_RT(camera_extrinsics):
    azimuth_rad =           camera_extrinsics[0] * pi / 180
    elevation_rad =         camera_extrinsics[1] * pi / 180
    roll_rad =              camera_extrinsics[2] * pi / 180
    radius =                camera_extrinsics[3]
    pitch_offset_rad =      camera_extrinsics[4] * pi / 180
    heading_offset_rad =    camera_extrinsics[5] * pi / 180
    y = radius * sin(elevation_rad)
    x = radius * sin(azimuth_rad) * cos(elevation_rad)
    z = radius * cos(azimuth_rad) * cos(elevation_rad)

    x_rot = np.zeros((3,3))
    x_rot[0,0] = 1
    x_rot[1,1] = x_rot[2,2] = cos(-(elevation_rad - pitch_offset_rad))
    x_rot[2,1] = sin(-(elevation_rad - pitch_offset_rad))
    x_rot[1,2] = -1 * x_rot[2,1]

    y_rot = np.zeros((3,3))
    y_rot[1,1] = 1
    y_rot[0,0] = y_rot[2,2] = cos(azimuth_rad - heading_offset_rad)
    y_rot[0,2] = sin(azimuth_rad - heading_offset_rad)
    y_rot[2,0] = -1 * y_rot[0,2]

    z_rot = np.zeros((3,3))
    z_rot[2,2] = 1
    z_rot[0,0] = z_rot[1,1] = cos(roll_rad)
    z_rot[1,0] = sin(roll_rad)
    z_rot[0,1] = -1 * z_rot[1,0]

    R = np.dot(y_rot, np.dot(x_rot, z_rot))
    T = -1 * np.array([[x], [y], [z]])

    RT = np.concatenate((R,T), axis=1)

    return RT

def weight_var_nl(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_var_nl(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def conv2d_nl(input_x, in_depth, out_depth, kernel_size, stride, layer_name):
    with tf.variable_scope(layer_name):
        W = weight_var_nl([kernel_size, kernel_size, in_depth, out_depth])
        b = bias_var_nl([out_depth])
        conv = tf.nn.conv2d(input_x, W, strides=[1, stride, stride, 1], 
                            padding='SAME') + b
        print conv
    return conv

def fc_nl(input_x, in_length, out_length, layer_name):
    with tf.variable_scope(layer_name):
        W = weight_var_nl([in_length, out_length])
        b = bias_var_nl([out_length])
        fc = tf.matmul(input_x, W) + b
        print fc
    return fc

def deconv2d_nl(input_x, in_depth, out_shape, kernel_size, stride, layer_name):
    with tf.variable_scope(layer_name):
        out_depth = out_shape[3]
        W = weight_var_nl([kernel_size, kernel_size, out_depth, in_depth])
        b = bias_var_nl([out_depth])
        dconv = tf.nn.conv2d_transpose(input_x, W, strides=[1, stride, stride, 1], 
                                        padding='SAME', output_shape=out_shape) + b
        print dconv
    return dconv
