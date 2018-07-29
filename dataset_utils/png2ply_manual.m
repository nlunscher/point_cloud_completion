%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% png2ply_manual.m
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

clear;

folder = '../../Data/caesar-norm-wsx_pngs/pc_completion_train_pngs/CSR0002A_L-foot/'
file_name = 'train_in_im_0.png'
im_file = strcat(folder, file_name);

RT = [eye(3,3), zeros(3,1)];

im2 = imread(im_file);
im = im2double(im2(:,:,1))*(2^16-1);
im(im > (2^16-10000)) = 0;

im_size =  size(im);
n_points = size(im,1) * size(im,2);

z_scale = 0.0001;
im = im * z_scale;

K = [
192.0 0.0 64.0 
0.0 192.0 64.0 
0.0 0.0 1.0 
];

points = zeros(n_points, 2);
points(:,3) = 1;

for y = 1:size(im,1)
   for x = 1:size(im,2)
       
       points((x-1)*size(im,1) + y, 1) = x - 1;
       points((x-1)*size(im,1) + y, 2) = y - 1;
       
   end
end

points_world = (inv(K) * points')';
depth = repmat(reshape(im, [n_points,1]), 1,3);
zs = repmat(points(:,3), 1,3);
points_world = (points_world.*depth)./zs;

ply_points = points_world / 0.003;

header = strcat('ply \n', ...
'format ascii 1.0 \n', ...
'comment Author: Nolan Lunscher \n', ...
['element vertex ', num2str(size(ply_points,1)), ' \n'], ...
'property float x \n', ...
'property float y \n', ...
'property float z \n', ...
'end_header \n');
fileID = fopen(strcat(im_file,'_3D_point_cloud.ply'),'w');
nbytes = fprintf(fileID, header);
nbytes = fprintf(fileID, '%12.8f %12.8f %12.8f \n', ply_points');
fclose(fileID);