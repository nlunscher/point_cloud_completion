#################################################################################
# pc_completion.py
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
import tensorflow as tf
from tensorflow.python.framework import dtypes

import datetime
import random
import threading

import file_util
from pc_completion_util import *

class pc_completion():
    def __init__(self, sess, d_name, allow_save = True, show_images = True):
        self.sess = sess
        self.image_size = 128
        self.image_depth = 1
        self.image_shape = [self.image_size, self.image_size, self.image_depth]
        self.scaling = 0.0001 # scaling of the image format

        self.learning_rate = 5e-5
        self.batch_size = 64 

        self.iterations = 1300000
        self.global_itter = 0

        self.allow_save = allow_save
        self.show_images = show_images

        self.net_name = d_name
        self.dataset_folder = "../Data/"
        self.pngs_folder = self.dataset_folder + "caesar-norm-wsx_pngs/"
        self.info_folder = self.dataset_folder + "saved_info_dx/" + d_name
        self.cnn_model = "Initial Model"
        self.models_folder = self.info_folder + "/saved_models/"
        self.images_folder = self.info_folder + "/saved_images/"
        self.loss_csv = self.info_folder + "/cnn_losses.csv"
        self.cnn_info = self.info_folder + "/cnn_train_info.txt"

        self.in_batch = None
        self.out_batch = None
        self.STOP = False

    def initialize(self):
        self.setup_folders()

        self.train_dataset_folder = self.pngs_folder + "pc_completion_train_pngs/"
        self.train_objects = read_dataset_objects("nl_train_legs.txt")
        self.train_dataset_length = len(self.train_objects)
        self.train_data_iter = 0
        random.shuffle(self.train_objects)
        self.train_im_per_obj = 256
        self.train_im_iter = 0

        self.val_dataset_folder = self.pngs_folder + "pc_completion_val_pngs/"
        self.val_objects = read_dataset_objects("nl_val_legs.txt")
        self.val_dataset_length = len(self.val_objects)
        self.val_data_iter = 0
        self.val_im_per_obj = 8
        self.val_im_iter = 0
        self.val_iterations = self.val_dataset_length * self.val_im_per_obj / self.batch_size

        self.test_dataset_folder = self.pngs_folder + "pc_completion_test_pngs/"
        self.test_objects = read_dataset_objects("nl_test_legs.txt")
        self.test_dataset_length = len(self.test_objects)
        self.test_data_iter = 0
        self.test_im_per_obj = 64
        self.test_im_iter = 0
        self.test_iterations = self.test_dataset_length * self.test_im_per_obj / self.batch_size

        self.sess.run(tf.global_variables_initializer())
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(coord = self.coord)

    def setup_folders(self):
        file_util.make_directory(self.info_folder)
        file_util.make_directory(self.models_folder)
        file_util.make_directory(self.images_folder)
        file_util.make_directory(self.images_folder + "test")
        file_util.make_directory(self.images_folder + "train")
        file_util.make_directory(self.images_folder + "val")

    def iterate_data(self, dataset_type = "Train"):
        if dataset_type == "Train":
            self.train_data_iter += 1
            if self.train_data_iter >= self.train_dataset_length: # 1 epoch just finished
                random.shuffle(self.train_objects)
                self.train_data_iter = 0
                self.train_im_iter += 1
                if self.train_im_iter >= self.train_im_per_obj:
                    self.train_im_iter = 0
        elif dataset_type == "Test":
            self.test_im_iter += 1
            if self.test_im_iter >= self.test_im_per_obj:
                self.test_im_iter = 0
                self.test_data_iter += 1
                if self.test_data_iter >= self.test_dataset_length:
                    self.test_data_iter = 0
        elif dataset_type == "Val":
            self.val_data_iter += 1
            if self.val_data_iter >= self.val_dataset_length:
                self.val_data_iter = 0
                self.val_im_iter += 1
                if self.val_im_iter >= self.val_im_per_obj:
                    self.val_im_iter = 0
        else:
            print "************* Iterating Nothing....."

    def load_image(self, file_name, add_noise = False):
        im = read_png16(file_name)
        if add_noise:
            im = add_kinect_v1_noise(im)
        im = depthmap_16bitTofloat(im)
        return im

    def get_batch(self, dataset_type = "Train", save_batch = False):
        if dataset_type == "Train":
            in_batch = []
            out_batch = []
            for b in range(self.batch_size):
                obj_name = self.train_objects[self.train_data_iter]
                obj_folder = self.train_dataset_folder + obj_name + "/"
                in_im = self.load_image(obj_folder + "train_in_im_" + str(self.train_im_iter) + ".png", add_noise=True)
                out_im = self.load_image(obj_folder + "train_gt_im_" + str(self.train_im_iter) + ".png")
                in_batch.append(in_im.reshape((self.image_size, self.image_size, 1)))
                out_batch.append(out_im.reshape((self.image_size, self.image_size, 1)))
                self.iterate_data("Train")

        elif dataset_type == "Test":
            in_batch = []
            out_batch = []
            for b in range(self.batch_size):
                obj_name = self.test_objects[self.test_data_iter]
                obj_folder = self.test_dataset_folder + obj_name + "/"
                in_im = self.load_image(obj_folder + "test_in_im_" + str(self.test_im_iter) + ".png", add_noise=True)
                out_im = self.load_image(obj_folder + "test_gt_im_" + str(self.test_im_iter) + ".png")
                in_batch.append(in_im.reshape((self.image_size, self.image_size, 1)))
                out_batch.append(out_im.reshape((self.image_size, self.image_size, 1)))
                self.iterate_data("Test")

        elif dataset_type == "Val":
            in_batch = []
            out_batch = []
            for b in range(self.batch_size):
                obj_name = self.val_objects[self.val_data_iter]
                obj_folder = self.val_dataset_folder + obj_name + "/"
                # in_im = self.load_image(obj_folder + "val_in_im_" + str(self.val_im_iter) + ".png", add_noise=True)
                # out_im = self.load_image(obj_folder + "val_gt_im_" + str(self.val_im_iter) + ".png")
                in_im = self.load_image(obj_folder + "test_in_im_" + str(self.val_im_iter) + ".png", add_noise=True)
                out_im = self.load_image(obj_folder + "test_gt_im_" + str(self.val_im_iter) + ".png")
                in_batch.append(in_im.reshape((self.image_size, self.image_size, 1)))
                out_batch.append(out_im.reshape((self.image_size, self.image_size, 1)))
                self.iterate_data("Val")

        in_batch = np.asarray(in_batch)
        out_batch = np.asarray(out_batch)
        if save_batch:
            self.in_batch = in_batch
            self.out_batch = out_batch

        return in_batch, out_batch

    def show_num_parameters(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print "Total trainable model parameters:", total_parameters
        return total_parameters

    def restore_net(self,  restore_path = "NONE"):
        if restore_path == "NONE":
            if self.global_itter > 0:
                self.cnn_model = self.models_folder + self.net_name + "_" + str(self.global_itter) + ".ckpt"
                if file_util.file_exists(self.cnn_model + ".meta"):
                    self.saver.restore(self.sess, self.cnn_model)
                    rollback_losses_csv(self.loss_csv, self.global_itter)
                    print "**** Model Restored"
                else:
                    self.cnn_model = "Initial Model"
                    print "**** Restore File not found --> Initial Model"
            else:
                self.cnn_model = "Initial Model"
                print "**** Initial Model"
        else:
            if file_util.file_exists(restore_path + ".meta"):
                self.cnn_model = restore_path
                self.global_itter = -1
                self.saver.restore(self.sess, self.cnn_model)
                print "**** Model Restored"
            else:
                self.cnn_model = "Initial Model"
                print "**** Restore File not found --> Initial Model"

    def create_net(self, special_batch_size = -1):
        if special_batch_size != -1:
            self.batch_size = special_batch_size

        print "---------------------------------------------------------------"
        print "Model"

        self.in_im = tf.placeholder(tf.float32, name = 'input_image',
                        shape= [self.batch_size] + self.image_shape) # 128x128
        self.out_gt = tf.placeholder(tf.float32, name = 'output_ground_truth',
                        shape= [self.batch_size] + self.image_shape) # 128x128
        self.is_train = tf.placeholder(tf.bool, name = 'is_train')

        conv1 = tf.nn.relu(conv2d_nl(self.in_im, 1, 32, 5, 2, "conv1")) # 64x64
        conv2 = tf.nn.relu(conv2d_nl(conv1, 32, 32, 5, 2, "conv2")) # 32x32
        conv3 = tf.nn.relu(conv2d_nl(conv2, 32, 64, 5, 2, "conv3")) # 16x16
        conv4 = tf.nn.relu(conv2d_nl(conv3, 64, 128, 3, 2, "conv4")) # 8x8
        conv5 = tf.nn.relu(conv2d_nl(conv4, 128, 256, 3, 2, "conv5")) # 4x4

        conv5_1d = tf.reshape(conv5, [self.batch_size, 4 * 4 * 256]) # 4096
        fc1 = tf.nn.relu(fc_nl(conv5_1d, 4096, 1024, "fc1")) # 1024
        fc2 = tf.nn.relu(fc_nl(fc1, 1024, 4096, "fc2")) # 4096
        fc2_2d = tf.reshape(fc2, [self.batch_size, 4, 4, 256]) # 4x4

        dconv1 = tf.nn.relu(deconv2d_nl(fc2_2d, 256, [self.batch_size, 8, 8, 128], 3, 2, "dconv1")) # 8x8
        dconv2 = tf.nn.relu(deconv2d_nl(dconv1, 128, [self.batch_size, 16, 16, 64], 3, 2, "dconv2")) # 16x16
        dconv3 = tf.nn.relu(deconv2d_nl(dconv2, 64, [self.batch_size, 32, 32, 32], 5, 2, "dconv3")) # 32x32
        dconv4 = tf.nn.relu(deconv2d_nl(dconv3, 32, [self.batch_size, 64, 64, 32], 5, 2, "dconv4")) # 64x64
        dconv5 = tf.nn.tanh(deconv2d_nl(dconv4, 32, [self.batch_size, 128, 128, 1], 5, 2, "dconv5")) # 128x128
        self.out_im = dconv5

        self.loss = tf.reduce_mean(tf.reduce_sum(tf.abs(self.out_im - self.out_gt), 3))

        self.trainable_vars = tf.trainable_variables()
        self.show_num_parameters()

        self.saver = tf.train.Saver(max_to_keep=1000)

        print "---------------------------------------------------------------"

    def read_pairs(self, dataset_type = "Train", iterations = 100):

        self.initialize()

        cv2.namedWindow("Depthmaps", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Depthmaps", 256*2, 256)

        print "Starting"
        print "Reading dataset:", dataset_type
        print "Reading for", iterations, "iterations"
        for i in range(iterations):
            in_batch, out_batch = self.get_batch(dataset_type)

            for j in range(self.batch_size):
                print i, j, "[", np.min(in_batch[j]), np.max(in_batch[j]), "]", \
                            "[", np.min(out_batch[j]), np.max(out_batch[j]), "]", \
                            datetime.datetime.now() - start_time
                show_image = np.concatenate([depth_2jet(in_batch[j]), 
                                            depth_2jet(out_batch[j])], axis=1)
                cv2.imshow("Depthmaps", show_image)
                key = cv2.waitKey(1000/4)

        self.coord.request_stop()
        self.coord.join(self.threads)

    def thread_activity(self, dataset_type = "Train"):
        if not self.STOP:
            self.STOP = True
            self.get_batch(dataset_type, True)
            self.STOP = False

    def copy_batch(self):
        return self.in_batch[:], self.out_batch[:]

    def train(self):

        ran_val = False

        self.create_net()
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.initialize()
        self.restore_net()

        print "Max Iterations:", self.iterations
        print "Current Iteration:", self.global_itter
        print "Batch Size:", self.batch_size
        print "Learning Rate:", self.learning_rate
        # write this to file
        cnn_train_info = "\nTraining DateTime Started: " + str(start_time) + "\n"
        cnn_train_info += "Max Iterations: " + str(self.iterations) + "\n"
        cnn_train_info += "Current Iteration: " + str(self.global_itter) + "\n"
        cnn_train_info += "Batch Size: " + str(self.batch_size) + "\n"
        cnn_train_info += "Learning Rate: " + str(self.learning_rate) + "\n"
        file_util.append_to_file(self.cnn_info, cnn_train_info)
        print "---------------------------------------------------------------"

        batch_getter = threading.Thread(target=self.thread_activity, args=("Train",))
        batch_getter.start()
        for i in range(self.global_itter, self.iterations):
            self.global_itter = i

            batch_getter.join()
            if self.STOP:
                print "STOPPING TRAIN!!"
                print self.last_IO_file
                break
            in_batch, out_batch = self.copy_batch()
            batch_getter = threading.Thread(target=self.thread_activity, args=("Train",))
            batch_getter.start()

            # show info every so often
            if (i % 5 == 0 or i == self.iterations-1):
                iloss, ix, iy_gt, iy_cnn = sess.run( # forward pass
                        [self.loss, self.in_im, self.out_gt, self.out_im], 
                            feed_dict={self.in_im:in_batch, self.out_gt:out_batch, self.is_train:False})
                print i, "Loss:", iloss, "Time:", datetime.datetime.now() - start_time, "Now:", str(datetime.datetime.now())
                if self.show_images: # show images in windows
                    print "     [", np.min(ix[0]), np.max(ix[0]), "]", \
                               "[", np.min(iy_gt[0]), np.max(iy_gt[0]), "]", \
                               "[", np.min(iy_cnn[0]), np.max(iy_cnn[0]), "]"
                    show_3_images(ix[0], iy_gt[0], iy_cnn[0], wait_time=100)
                if self.allow_save and (i % 1000 == 0 or i == self.iterations-1):
                    save_iteration_images(self.images_folder + "train/", i, 
                                            ix[0], iy_gt[0], iy_cnn[0], [0,0,0,0,0,0])

            # Save model checkpoint
            if (i % 10000 == 0 or i == self.iterations-1) and True:
                # Run Validation
                val_loss = self.test("Val", False, True)
                ran_val = True

                if self.allow_save and i > 0:
                    save_path = self.saver.save(sess, self.models_folder + \
                                            self.net_name + "_" + str(i) + ".ckpt")
                    print ("Saved model as: %s" % save_path)
                print "---------------------------------------------------------------"

            # Learn
            _, iloss = self.sess.run([train_step, self.loss], 
                                        feed_dict={self.in_im:in_batch,
                                                    self.out_gt:out_batch, self.is_train:True})
            if self.allow_save:
                to_csv = str(i) + ", " + str(iloss)
                if ran_val:
                    to_csv += ", " + str(val_loss)
                    ran_val = False
                file_util.append_line_to_csv(self.loss_csv, to_csv)

        # Done Training
        total_runtime = datetime.datetime.now() - start_time
        print "Total Runtime:", total_runtime
        cnn_train_info = "Total Runtime: " + str(total_runtime) + "\n\n"
        file_util.append_to_file(self.cnn_info, cnn_train_info)

        self.coord.request_stop()
        self.coord.join(self.threads)

    def test(self, dataset_type = "Test", show_outputs = False, save_outputs = False):

        if dataset_type == "Test":
            iterations = self.test_dataset_length
            if self.allow_save and save_outputs:
                test_folder = self.images_folder + "test/"
                file_util.make_directory(test_folder)
            print "Testing on:", self.test_dataset_folder
        elif dataset_type == "Val":
            iterations = self.val_iterations
            if self.allow_save and save_outputs:
                test_folder = self.images_folder + "val/itteration_" + str(self.global_itter) + "/"
                file_util.make_directory(test_folder)
            print "Validating on:", self.val_dataset_folder

        print "---------------------------------------------------------------"
        print "Starting Test -", "Iterations:", iterations, "Batch Size:", self.batch_size

        losses = np.zeros(iterations)
        for i in range(iterations):
            in_batch, out_batch = self.get_batch(dataset_type)

            iloss, ix, iy_gt, iy_cnn = sess.run(
                        [self.loss, self.in_im, self.out_gt, self.out_im], 
                            feed_dict={self.in_im:in_batch, self.out_gt:out_batch, self.is_train:False})
            losses[i] = iloss
            if show_outputs or (dataset_type == "Test" and i % 100 == 0):
                print i, "Loss:", iloss, "Time:", datetime.datetime.now() - start_time
            for j in range(self.batch_size):
                if show_outputs:
                    if self.show_images:
                        print j, "[", np.min(ix[j]), np.max(ix[j]), "]", \
                                "[", np.min(iy_gt[j]), np.max(iy_gt[j]), "]"
                        show_3_images(ix[j], iy_gt[j], iy_cnn[j], wait_time=100)

                if self.allow_save and save_outputs:
                    if dataset_type == "Val":
                        if j == 0 and i < 100:
                            save_iteration_images(test_folder, str(i) + "-" + str(j), 
                                                    ix[j], iy_gt[j], iy_cnn[j], [0,0,0,0,0,0],
                                                    dataset_type)
                    else:
                        test_obj_folder = test_folder + "obj_" + str(i) + "/"
                        file_util.make_directory(test_obj_folder)
                        save_iteration_images(test_obj_folder, str(i) + "-" + str(j), 
                                                    ix[j], iy_gt[j], iy_cnn[j], [0,0,0,0,0,0],
                                                    dataset_type)

        total_runtime = datetime.datetime.now() - start_time
        average_loss = np.mean(losses)
        print "Average loss:", average_loss

        if dataset_type =="Test":
            print "Total Runtime:", total_runtime

            cnn_test_info = "\nDateTime Started: " + str(start_time) + "\n"
            cnn_test_info += "Test Set Results\n"
            if self.global_itter > -1:
                cnn_test_info += self.net_name + " Itteration: " + str(self.global_itter) + "\n"
            else:
                cnn_test_info += self.cnn_model + "\n"
            cnn_test_info += "Average loss: " + str(average_loss) + "\n"
            cnn_test_info += "Total Runtime: " + str(total_runtime) + "\n\n"
            file_util.append_to_file(self.cnn_info, cnn_test_info)


        return average_loss




np.set_printoptions(precision=4)

start_time = datetime.datetime.now()
with tf.Session() as sess:
    d_name = "foot_pc_completion"
    pc = pc_completion(sess, d_name, 
        allow_save = True, show_images = True)

    run_option = 0


    # ========== Read From the Dataset ==========
    if run_option == 0:
        print "Running: Read From the Dataset"
        pc.read_pairs("Train", 100)

    # ========== Train the Model ================
    elif run_option == 1:
        print "Running: Train the Model"
        pc.train()

    # ========== Validate the Model =================
    elif run_option == 2:
        print "Running: Validate the Model"
        pc.create_net()
        pc.initialize()
        pc.restore_net('model1300000/pc_completion_1300000.ckpt')
        pc.test("Val", show_outputs = True)

    # ========== Test the Model =================
    elif run_option == 3:
        print "Running: Test the Model"
        pc.create_net()
        pc.initialize()
        pc.restore_net('model1300000/pc_completion_1300000.ckpt')
        pc.test("Test", show_outputs = False, save_outputs = True)

    else:
        print "A Valid Run Option was not selected"


print "============================================================"
print "Ending Program..."

