#################################################################################
# depth_map_generator.py
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

# renders random views of an object

import cv2
import numpy as np
from math import pi, sin, cos
import random
import sys,os
import time, datetime

import file_util
from pc_completion_util import *

from panda3d.core import *
from direct.showbase.ShowBase import ShowBase
from direct.task import Task

class Depth_Generator(ShowBase):
    def __init__(self, image_size = 128, scaling = 0.0001):

        self.im_size = image_size

        # loadPrcFileData("", "window-type none")
        loadPrcFileData("", "window-type offscreen")
        # loadPrcFileData("", "Generate Depthmaps")
        # loadPrcFileData("", "fullscreen 0") # Set to 1 for fullscreen
        # loadPrcFileData("", "win-origin 100 100")
        loadPrcFileData("", "win-size " + str(image_size) + " " + str(image_size))

        ShowBase.__init__(self)
        # base.setFrameRateMeter(True)

        self.scaling = scaling # ever value 1 (in png) represents scaling distance in game engine
        self.azimuth = 0
        self.elevation = 20
        self.roll = 0
        self.radius = 2
        self.pitch_offset = 0
        self.heading_offset = 0

        self.obj_scale = 0.003
        a_range = 20
        self.az_opt = np.concatenate([np.arange(90-a_range,90+a_range+.1), np.arange(270-a_range,270+a_range+.1)],axis =0)
        self.el_opt = np.arange(-a_range, a_range +0.1)
        self.radius_options = np.arange(1.9, 2.1 +0.05, 0.05)
        self.r_options = np.arange(-5, 5 +.1, 1)

        self.max_16bit = 2**16-1
        self.near_plane = 1.0
        self.far_plane = 10.0
        self.focal_length = 1.5
        self.cam.node().getLens().setNear(self.near_plane)
        self.cam.node().getLens().setFar(self.far_plane)
        self.cam.node().getLens().setFocalLength(self.focal_length)
        self.K = self.get_calib_mat()

        self.object_names_to_index = {}
        self.all_objects = []
        self.object = None
        self.current_object_index = -1

        self.scene = NodePath("Scene")
        self.scene.reparentTo(self.render)
        self.scene.setScale(1, 1, 1)
        self.scene.setTwoSided(True) # important to avoid alot of holes
        self.scene.setPos(0, 0, 0)
        self.scene.setHpr(0, 0, 0)

        self.taskMgr.add(self.everyFrameTask, "EveryFrameTask")
        self.frame = 0
        self.roll_direction = self.elevation_direction = self.radius_direction = 1
        self.rotation_count = 0

    ## source: https://gist.github.com/alexlee-gk/b28fb962c9b2da586d1591bac8888f1f
    ## {
        self.dr = self.camNode.getDisplayRegion(0)

        winprops = WindowProperties.size(self.win.getXSize(), self.win.getYSize())
        fbprops = FrameBufferProperties()
        fbprops.setDepthBits(32)
        self.depthBuffer = self.graphicsEngine.makeOutput(
            self.pipe, "depth buffer", -2,
            fbprops, winprops,
            GraphicsPipe.BFRefuseWindow,
            self.win.getGsg(), self.win)
        self.depthTex = Texture()
        self.depthTex.setFormat(Texture.FDepthComponent)
        self.depthBuffer.addRenderTexture(self.depthTex,
            GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPDepth)
        lens = self.cam.node().getLens()

        self.depthCam = self.makeCamera(self.depthBuffer,
            lens=lens,
            scene=render)
        self.depthCam.reparentTo(self.cam)

    def get_camera_image(self):
        tex = self.dr.getScreenshot()
        data = tex.getRamImage()
        image = np.frombuffer(data.get_data(), np.uint8)
        image.shape = (tex.getYSize(), tex.getXSize(), tex.getNumComponents())
        image = np.flipud(image)
        return image

    def get_camera_depth_image(self):
        # values between 0.0 and 1.0.
        data = self.depthTex.getRamImage()
        depth_image = np.frombuffer(data.get_data(), np.float32)
        # depth_image.shape = (self.depthTex.getYSize(), self.depthTex.getXSize(), self.depthTex.getNumComponents())
        depth_image.shape = (self.depthTex.getYSize(), self.depthTex.getXSize(), 1)
        depth_image = np.flipud(depth_image)
        return depth_image
    ## }

    def close(self):
        self.unload_all_objects()
        base.destroy()

    def linearize_depth(self, d_image):
        # depth function: d = b1/(z + b2) + b3
        b1 = - self.far_plane / (self.far_plane - self.near_plane)
        b2 = 0.0
        b3 = self.far_plane / (self.far_plane - self.near_plane)

        l_d_image = np.round((b1 / (d_image[:,:,0] - b3) - b2) / self.scaling)

        l_d_image[l_d_image >= np.max([self.max_16bit, self.far_plane/self.scaling])] = self.max_16bit
        l_d_image[l_d_image <= 0] = self.max_16bit

        return l_d_image.astype(np.uint16)

    def get_depthmap(self):
        return self.linearize_depth(self.get_camera_depth_image())

    def unload_all_objects(self):
        self.remove_viewable_object()
        for obj in self.all_objects:
            self.loader.unloadModel(obj)
            obj.removeNode()
            obj = None
        self.all_objects = []

        TransformState.garbageCollect()
        RenderState.garbageCollect()

    def load_all_objects(self, object_names, objects_path = ""):
        self.object_names_to_index = {}
        index = 0
        full_object_paths = []
        for name in object_names:
            full_path = objects_path + name 
            full_object_paths.append(full_path)

            self.object_names_to_index[name] = index
            index += 1

        self.all_objects = self.loader.loadModel(full_object_paths)
        # print "Loaded", len(self.all_objects), "objects"

    def pick_viewable_object(self, object_index):
        # print self.all_objects, object_index
        self.object = self.all_objects[object_index]
        self.object.reparentTo(self.scene)
        self.object.setScale(self.obj_scale, self.obj_scale, self.obj_scale)
        self.object.setPos(0,0,0)# - feet
        self.object.setHpr(0,0,0)# - feet
        self.current_object_index = object_index

    def remove_viewable_object(self):
        if self.object is not None:
            self.object.detachNode()
            self.object = None
            self.current_object_index = -1

    def switch_viewable_object(self, object_index):
        self.remove_viewable_object()
        self.pick_viewable_object(object_index)

    def switch_viewable_object_by_name(self, object_name):
        self.switch_viewable_object(self.object_names_to_index[object_name])

    def camera_angles_to_coord(self):
        az_rad = self.azimuth * pi / 180 # input is in degrees
        el_rad = self.elevation * pi / 180

        z = self.radius * sin(el_rad)
        x = self.radius * sin(az_rad) * cos(el_rad)
        y = self.radius * cos(az_rad) * cos(el_rad)

        return x, y, z # z = forward, x = sideways, y = updown

    def camera_angles_to_direction(self):
        h = -(self.azimuth + 180) + self.heading_offset
        p = -(self.elevation) + self.pitch_offset
        r = self.roll
        return h, p, r # yaw/heading, pitch, roll

    def get_calib_mat(self):
        film_size = self.cam.node().getLens().getFilmSize()

        K = np.zeros((3,3))
        K[0,0] = K[1,1] = self.focal_length * self.im_size * film_size[0]
        K[0,2] = self.im_size/2.0
        K[1,2] = self.im_size/2.0
        K[2,2] = 1

        return K

    def update_camera_position(self):
        cam_x, cam_y, cam_z = self.camera_angles_to_coord()
        cam_h, cam_p, cam_r = self.camera_angles_to_direction()
        self.camera.setPos(cam_x, cam_y, cam_z)
        self.camera.setHpr(cam_h, cam_p, cam_r)
        # print self.camera.getPos()

    def depth_2jet(self, depth_image):
        farthest = 4.5/self.scaling
        closest = 1/self.scaling
        dist = farthest - closest

        d_im = depth_image - closest
        d_im[np.where(d_im > dist)] = dist

        d_im = d_im / (np.max(d_im) + 1e-8) * 255
        d_im = d_im.astype(np.uint8)
        return cv2.applyColorMap(d_im, cv2.COLORMAP_JET)

    def generate_image(self, object_name, camera_extrinsics):
        self.switch_viewable_object_by_name(object_name)
        self.azimuth = camera_extrinsics[0]
        self.elevation = camera_extrinsics[1]
        self.roll = camera_extrinsics[2]
        self.radius = camera_extrinsics[3]
        self.pitch_offset = camera_extrinsics[4]
        self.heading_offset = camera_extrinsics[5]

        # print camera_extrinsics

        self.update_camera_position()
        base.graphicsEngine.renderFrame()
        d_image = self.get_depthmap()

        TransformState.garbageCollect()
        RenderState.garbageCollect()

        return d_image.reshape((self.im_size, self.im_size, 1))

    def generate_random_image(self, object_name, image_type = "input", in_extrin = None):
        if image_type == "input":
            az = random.choice(self.az_opt)
            el = random.choice(self.el_opt)
            rad = random.choice(self.radius_options)
            roll = random.choice(self.r_options)
            pit_o = head_o = 0.0
        else:
            az, el, roll, rad, pit_o, head_o = extrinsics_to_otherside(in_extrin)
        # print image_type, [az, el, roll, rad, pit_o, head_o]

        camera_extrinsics = [az, el, roll, rad, pit_o, head_o]
        d_image = self.generate_image(object_name, camera_extrinsics)
        return d_image, camera_extrinsics

    def generate_images(self, object_names):
        d_images = [[], []]
        d_images_jet = [[], []]
        extrins = [[], []]
        for name in object_names:
            d_image, extrin = self.generate_random_image(name, "input")
            d_images[0].append(d_image)
            d_images_jet[0].append(self.depth_2jet(d_image))
            extrins[0].append(extrin)

            d_image, extrin = self.generate_random_image(name, "output", extrin)
            d_images[1].append(d_image)
            d_images_jet[1].append(self.depth_2jet(d_image))
            extrins[1].append(extrin)

        return d_images, d_images_jet, extrins

    def generate_all_outputs_images(self, object_name):
        d_images = []
        extrins = []
        for rad in self.radius_options_out:
            for el in self.elevation_options_out:
                for az in self.azimuth_options_out:
                    extrin = [az, el, 0, rad, 0, 0]
                    d_image = self.generate_image(object_name, extrin)
                    d_images.append(d_image)
                    extrins.append(extrin)

        return d_images, extrins

    def everyFrameTask(self, task):
        if self.frame > 2:
            self.update_camera_position()

            d_image = self.get_depthmap()
            # print np.min(d_image), np.max(d_image)
            cv2.imshow("Images", self.depth_2jet(d_image))
            key = cv2.waitKey(100)

            #### setup for next view
            self.azimuth += 5
            if self.azimuth >= 360:
                self.azimuth = 0
                self.rotation_count += 1
                print "Full Rotation", self.rotation_count

            if self.rotation_count >= 2:
                next_object = self.current_object_index + 1
                if next_object >= len(self.all_objects):
                    next_object = 0
                self.switch_viewable_object(next_object)
                self.rotation_count = 0
            self.elevation += 0.2 * self.elevation_direction * 5
            if self.elevation > 30 or self.elevation < -30:
                self.elevation_direction *= -1
            self.roll += 0.1 * self.roll_direction * 5
            if self.roll > 10 or self.roll < -10:
                self.roll_direction *= -1
            self.radius += 0.01 * self.radius_direction * 5
            if self.radius > 4 or self.radius < 1.5:
                self.radius_direction *= -1            

        self.frame += 1
        # time.sleep(1)
        return Task.cont

