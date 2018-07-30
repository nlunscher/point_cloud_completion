# point_cloud_completion

[**Point Cloud Completion of Foot Shape From a Single Depth Map for Fit Matching Using Deep Learning View Synthesis**](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w32/Lunscher_Point_Cloud_Completion_ICCV_2017_paper.pdf)

Nolan Lunscher and John Zelek

![alt tag](https://github.com/nlunscher/pc_completion0/blob/master/io_setup.png)

This code was used for my paper presented in the Computer Vision for Fashion Workshop in ICCV 2017. If you find it useful in your works, please site the original paper.

This project and parts of its code were inspired by "Multi-view 3D Models from Single Images with a Convolutional Network", which you may find useful as well: https://github.com/lmb-freiburg/mv3d

## Dependencies
This program has the below dependencies but will likely work with newer versions as well:
- Ubuntu 16.04
- Python 2.7
- Tensorflow 0.12
- Panda3D 1.9.3
- MatLab 2016b
- Opencv 3.4

## Usage
1. Download: *caesar-norm-wsx-fitted-meshes* (http://humanshape.mpi-inf.mpg.de/)
    - place in "../Data"
2. Run: **mat_model_mesh_centerfoot.m** from inside *dataset_utils*
    - make sure *mat_folder* is set correctly
3. Run: **convert_objs2bam.sh** from inside *dataset_utils*
    - make sure *obj_folder* and *bam_folder* are set correctly
4. Run: **pc_completion_make_data.py**
    - Run for train, test and val
    - make sure to set the appropriate values in the file for each set
5. Run: **pc_completion.py** - train
    - use *run_option* = 1
    - make sure *self.dataset_folder* is set correctly
6. Run: **pc_completion.py** - test depth map
    - use *run_option* = 3
    - make sure *self.dataset_folder* is set correctly
    - this will provide the depth map error
7. Run: **pc_errors.m** from inside *dataset_utils* - test point cloud
    - make sure that *folder*, *obj_nums* and *im_nums* as set correctly


## License and Citation
All code is provided for research purposes only and without any warranty. Any commercial use requires our consent. When using the code in your research work, please cite the following paper:

    @InProceedings{Lunscher_2017_ICCV_Workshops,
    author = {Lunscher, Nolan and Zelek, John},
    title = {Point Cloud Completion of Foot Shape From a Single Depth Map for Fit Matching Using Deep Learning View Synthesis},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV) Workshops},
    month = {Oct},
    year = {2017}
    }
