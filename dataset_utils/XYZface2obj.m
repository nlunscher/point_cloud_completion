%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% XYZface2ply.m
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

function XYZface2ply(fileName, points, faces)

num_points = length(points);
num_faces = length(faces);

faces2 = faces +1;

header = strcat('# obj \n', ...
'# comment Author: Nolan Lunscher \n', ...
'\n');
fileID = fopen(fileName,'w');
nbytes = fprintf(fileID, header);
nbytes = fprintf(fileID, 'v %12.8f %12.8f %12.8f \n', points');
nbytes = fprintf(fileID, 'f %d %d %d \n', faces2');
fclose(fileID);