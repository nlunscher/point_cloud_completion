%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% transformPointCloud.m
%
% Author: Nolan Lunscher
%
% All code is provided for research purposes only and without any warranty. 
% Any commercial use requires our consent. 
% When using the code in your research work, please cite the following paper:
%     @InProceedings{Lunscher_2017_ICCV_Workshops,
%     author = {Lunscher, Nolan and Zelek, John},
%     title = {Point Cloud Completion of Foot Shape From a Single Depth Map for Fit Matching Using Deep Learning View Synthesis},
%     booktitle = {The IEEE International Conference on Computer Vision (ICCV) Workshops},
%     month = {Oct},
%     year = {2017}
%     }
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% XYZ - point cloud to be transformed 3xn
% Rt - a 3x4 matrix containing the Rotation and Transformation
%       matricies [R|T]

function cloud_transformed = transformPointCloud(cloud, RT)
    cloud_transformed = RT(1:3,1:3) * cloud + ...
                   repmat(RT(1:3,4), 1, size(cloud,2));
end