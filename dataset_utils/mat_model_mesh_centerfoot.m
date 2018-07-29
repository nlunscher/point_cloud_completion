%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mat_model_mesh_centerfoot.m
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

% reads the mat files and cuts off the feet
clear

load('body_faces.mat')
load('foot_indexes.mat')
mat_folder = '../../Data/raw_data/caesar-norm-wsx-fitted-meshes';

ply_folder = strcat(mat_folder, '_mesh_PLY');
obj_foot_folder = strcat(mat_folder, '_foot_mesh_OBJ')
files = dir(mat_folder);

mkdir(obj_foot_folder)

lf_center_idx1 = 4387;
lf_center_idx2 = 6447;
rf_center_idx1 = 3306;
rf_center_idx2 = 1253;

want_axis = [0 1 0];

for i = 3:10+3 %:size(files,1)

   mat_name = files(i).name;
   disp(mat_name);
   load(strcat(mat_folder, '/', mat_name));
   
   % LEFT
   lf_center1 = points(lf_center_idx1,:);
   lf_center2 = points(lf_center_idx2,:);
   lf_center = [mean([lf_center1(1),lf_center2(1)]),
                mean([lf_center1(2),lf_center2(2)]),
                mean([lf_center1(3),lf_center2(3)])];
   lf_vec = (lf_center1 - lf_center2)/norm(lf_center1 - lf_center2);
   lf_vecz = lf_vec;
   lf_vecz(3) = 0;
   lf_vecx = lf_vec;
   lf_vecx(1) = 0;
   thetaz = atan(lf_vecz(2)/lf_vecz(1)) + atan(want_axis(2)/want_axis(1));
   thetax = atan(lf_vecx(2)/lf_vecx(3)) - atan(want_axis(2)/want_axis(3));
   lf_rotationz = [
       cos(thetaz) -sin(thetaz) 0;
       sin(thetaz)  cos(thetaz) 0;
       0            0         1
   ]; % rotates so that the y axis points with the foot
   lf_rotationx = [
       1            0           0;
       0            cos(thetax) -sin(thetax);
       0            sin(thetax) cos(thetax)
   ];

   lf_t = eye(4);
   lf_t(4,1:3) = -lf_center;
   lf_pc = pctransform(pointCloud(points),affine3d(lf_t));
   lf_t = eye(4);
   lf_t(1:3,1:3) = lf_rotationz;
   lf_pc = pctransform(lf_pc,affine3d(lf_t));
   lf_t = eye(4);
   lf_t(1:3,1:3) = lf_rotationx;
   lf_pc = pctransform(lf_pc,affine3d(lf_t));
   if mean(lf_pc.Location(:,3)) < 0
       lf_t = eye(4);
       lf_t(1:3,1:3) = [
                   1            0           0;
                   0            cos(pi) -sin(pi);
                   0            sin(pi) cos(pi)
               ];
       lf_pc = pctransform(lf_pc, affine3d(lf_t));
   end
   if mean(lf_pc.Location(lf_pc.Location(:,3) < 500, 2)) > 0 % if leg is behind
       lf_t = eye(4);
       lf_t(1:3,1:3) = [
                   cos(pi) -sin(pi) 0;
                   sin(pi)  cos(pi) 0;
                   0            0         1
               ];
       lf_pc = pctransform(lf_pc, affine3d(lf_t));
   end
   
   % RIGHT
   rf_center1 = points(rf_center_idx1,:);
   rf_center2 = points(rf_center_idx2,:);
   rf_center = [mean([rf_center1(1),rf_center2(1)]),
                mean([rf_center1(2),rf_center2(2)]),
                mean([rf_center1(3),rf_center2(3)])];
   rf_vec = (rf_center1 - rf_center2)/norm(rf_center1 - rf_center2);
   rf_vecz = rf_vec;
   rf_vecz(3) = 0;
   rf_vecx = rf_vec;
   rf_vecx(1) = 0;
   thetaz = atan(rf_vecz(2)/rf_vecz(1)) + atan(want_axis(2)/want_axis(1));
   thetax = atan(rf_vecx(2)/rf_vecx(3)) - atan(want_axis(2)/want_axis(3));
   rf_rotationz = [
       cos(thetaz) -sin(thetaz) 0;
       sin(thetaz)  cos(thetaz) 0;
       0            0           1]; 
   rf_rotationx = [
       1            0           0;
       0            cos(thetax) -sin(thetax);
       0            sin(thetax) cos(thetax)
   ]; % rotates so that the y axis points with the foot
           
   rf_t = eye(4);
   rf_t(4,1:3) = -rf_center;
   rf_pc = pctransform(pointCloud(points),affine3d(rf_t));
   rf_t = eye(4);
   rf_t(1:3,1:3) = rf_rotationz;
   rf_pc = pctransform(rf_pc,affine3d(rf_t));
   rf_t = eye(4);
   rf_t(1:3,1:3) = rf_rotationx;
   rf_pc = pctransform(rf_pc,affine3d(rf_t));
   if mean(rf_pc.Location(:,3)) < 0
       rf_t = eye(4);
       rf_t(1:3,1:3) = [
                   1            0           0;
                   0            cos(pi) -sin(pi);
                   0            sin(pi) cos(pi)
               ];
       rf_pc = pctransform(rf_pc, affine3d(rf_t));
   end
   
   n_len = size(mat_name,2);
   
   % separate the feet
   % LEFT
   lf_pc2 = lf_pc.Location;
   lf_pc2(lf_pc2(:,3) > lf_pc2(239,3),:) = 10000;
   left_heal = lf_pc2(lf_center_idx2,1:2);
   right_heal = lf_pc2(rf_center_idx1,1:2);
   for j = 1:length(lf_pc2)
       if norm(lf_pc2(j,1:2) - left_heal) > norm(lf_pc2(j,1:2) - right_heal)
           lf_pc2(j,:) = 10000;
       end
   end
   lf_faces = [];
   for j = 1:length(faces)
       add_face = 1;
       for k = 1:3
           point_idx = faces(j,k) + 1;
           if lf_pc2(point_idx,1) > 6000 | lf_pc2(point_idx,2) > 6000 | lf_pc2(point_idx,3) > 6000
               add_face = 0;
           end
       end
       if add_face
          lf_faces = [lf_faces; faces(j,:)]; 
       end
   end
   lf_pc2(lf_pc2(:,3) > 6000,:) = 0;
   
   %RIGHT
   rf_pc2 = rf_pc.Location;
   rf_pc2(rf_pc2(:,3) > rf_pc2(239,3),:) = 10000;
   left_heal = rf_pc2(lf_center_idx2,1:2);
   right_heal = rf_pc2(rf_center_idx1,1:2);
   for j = 1:length(rf_pc2)
       if norm(rf_pc2(j,1:2) - left_heal) < norm(rf_pc2(j,1:2) - right_heal)
           rf_pc2(j,:) = 10000;
       end
   end
   rf_faces = [];
   for j = 1:length(faces)
       add_face = 1;
       for k = 1:3
           point_idx = faces(j,k) + 1;
           if rf_pc2(point_idx,1) > 6000 | rf_pc2(point_idx,2) > 6000 | rf_pc2(point_idx,3) > 6000
               add_face = 0;
           end
       end
       if add_face
          rf_faces = [rf_faces; faces(j,:)]; 
       end
   end
   rf_pc2(rf_pc2(:,3) > 6000,:) = 0;
   
   l_obj_name = strcat(obj_foot_folder, '/', mat_name(1:n_len-4), '_L-foot.obj');
   XYZface2obj(l_obj_name, lf_pc2, lf_faces);
   r_obj_name = strcat(obj_foot_folder, '/', mat_name(1:n_len-4), '_R-foot.obj');
   XYZface2obj(r_obj_name, rf_pc2, rf_faces);
   
end




% LEFT
% x = 189.7
% y = 60.3
% z = -982.1
% a = points(:,3) <= z+0.1 & points(:,3) >= z-0.1 & points(:,1) <= x+0.1 & points(:,1) >= x-0.1;
% [m, idx] = max(a) % -> idx = 4387
% points(idx,:)

% x = 113.3
% y = -121.3
% z = -1004
% a = points(:,3) <= z+0.1 & points(:,3) >= z-0.1 & points(:,1) <= x+0.1 & points(:,1) >= x-0.1;
% [m, idx] = max(a) % -> 6447
% points(idx,:)

% KNEE
% x = -105.1
% y = 108
% z = 589.2
% a = points(:,3) <= z+0.1 & points(:,3) >= z-0.1 & points(:,1) <= x+0.1 & points(:,1) >= x-0.1;
% [m, idx] = max(a) % -> 1721
% points(idx,:)

% RIGHT
% x = -162.7
% y = 75.03
% z = -992
% a = points(:,3) <= z+0.9 & points(:,3) >= z-0.9 & points(:,1) <= x+0.9 & points(:,1) >= x-0.9;
% [m, idx] = max(a) % -> idx = 3306
% points(idx,:)

% x =  79.6
% y =  207.8
% z =  -994.2
% a = points(:,3) <= z+0.9 & points(:,3) >= z-0.9 & points(:,1) <= x+0.9 & points(:,1) >= x-0.9;
% [m, idx] = max(a) % -> idx = 1253
% points(idx,:)




