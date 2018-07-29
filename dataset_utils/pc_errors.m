%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pc_errors.m
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

% calculates the error of a depthmap as a point cloud

clear;

folder = '../Data/saved_info_dx/foot_pc_completion/saved_images/test/';

obj_nums = 0:3; %0:1719;
im_nums = 0:7; %0:63;

errors = zeros(length(obj_nums), length(im_nums));
errors_o2g = zeros(length(obj_nums), length(im_nums));
errors_g2o = zeros(length(obj_nums), length(im_nums));

for i = 1:length(obj_nums)
    obj_num = obj_nums(i);
    disp(obj_num);
    for j = 1:length(im_nums)
        im_num = im_nums(j);
        
        [err, err_o2g, err_g2o] = pc_error(folder, obj_num, im_num);
        errors(i, j) = err;
        errors_o2g(i, j) = err_o2g;
        errors_g2o(i, j) = err_g2o;
    end
end

errors = reshape(errors, length(obj_nums)* length(im_nums), 1);
mean_err = mean(errors)
std_err = std(errors)

errors_o2g = reshape(errors_o2g, length(obj_nums)* length(im_nums), 1);
mean_err_o2g = mean(errors_o2g)
std_err_o2g = std(errors_o2g)

errors_g2o = reshape(errors_g2o, length(obj_nums)* length(im_nums), 1);
mean_err_g2o = mean(errors_g2o)
std_err_g2o = std(errors_g2o)


function [err, err_o2g, err_g2o] = pc_error(folder, obj_num, im_num)
    gt_file = strcat(folder, 'obj_',num2str(obj_num),'/gt_im_',num2str(obj_num),'-',num2str(im_num),'.png');
    out_file = strcat(folder, 'obj_',num2str(obj_num),'/out_im_',num2str(obj_num),'-',num2str(im_num),'.png');

    gt_png = imread(gt_file);
    out_png = imread(out_file);
    
    gt_pc = png2pc(gt_png);
    out_pc = png2pc(out_png);

    gt_pc = pc_crop(gt_pc);
    out_pc = pc_clean(out_pc);

%     pcshow(out_pc, 'MarkerSize', 50); hold on;
%     pcshow(gt_pc); hold off;

    [err, err_o2g, err_g2o] = nn_error(gt_pc, out_pc);
end

function pc = png2pc(png)
    im = double(png(:,:,1));
    im(im > (2^16-10000)) = 0;

    im_size =  size(im);
    n_points = size(im,1) * size(im,2);

    z_scale = 0.0001;
    im = im * z_scale;

    K = [192.0 0.0 64.0 
         0.0 192.0 64.0 
         0.0 0.0 1.0];
    
    points = zeros(n_points, 2);
    points(:,3) = 1;
    for y = 1:size(im,1)
       for x = 1:size(im,2)
           points((x-1)*size(im,1) + y, 1) = x - 1;
           points((x-1)*size(im,1) + y, 2) = y - 1;
       end
    end
     
    pc = (inv(K) * points')';
    depth = repmat(reshape(im, [n_points,1]), 1,3);
    zs = repmat(points(:,3), 1,3);
    pc = (pc.*depth)./zs;
    pc = pointCloud(pc / 0.003);
end

function pc_new = pc_crop(pc)
    ps = pc.Location;
    thresh = 1000;
    ps = ps(ps(:,3) > 0, :);
    ps = ps(ps(:,3) < thresh, :);
    pc_new = pointCloud(ps);
end

function pc_new = pc_clean(pc)
    pc_new = pc_crop(pc);

%     basic network
    [pc_new, inlierIndices, outlierIndices] = ...
            pcdenoise(pc_new, 'NumNeighbors', 3, 'Threshold', .1);
    [pc_new, inlierIndices, outlierIndices] = ...
            pcdenoise(pc_new, 'NumNeighbors', 3, 'Threshold', .5);
end

function [err, err_o2g, err_g2o] = nn_error(gt_pc, out_pc)
    p_gt = gt_pc.Location;
    p_out = out_pc.Location;

    [n_gt, k_gt] = size(p_gt);
    [n_out, k_out] = size(p_out);

    Mdl_gt = KDTreeSearcher(p_gt);
    Mdl_out = KDTreeSearcher(p_out);

    errors_o2g = zeros(n_out, 1);    
    nn_o2g = knnsearch(Mdl_gt, p_out,'K',1);
    for i = 1:n_out
       errors_o2g(i) = sqrt((p_out(i,1) - p_gt(nn_o2g(i),1))^2 + ...
                            (p_out(i,2) - p_gt(nn_o2g(i),2))^2 + ...
                            (p_out(i,3) - p_gt(nn_o2g(i),3))^2);
    end

    errors_g2o = zeros(n_gt, 1);
    nn_g2o = knnsearch(Mdl_out, p_gt,'K',1);
    for i = 1:n_gt
       errors_g2o(i) = sqrt((p_gt(i,1) - p_out(nn_g2o(i),1))^2 + ...
                            (p_gt(i,2) - p_out(nn_g2o(i),2))^2 + ...
                            (p_gt(i,3) - p_out(nn_g2o(i),3))^2);
    end

    mean_error_o2g = mean(errors_o2g);

    mean_error_g2o = mean(errors_g2o);

    mean_error = mean([mean_error_o2g, mean_error_g2o]);
    
    err_o2g = mean_error_o2g;
    err_g2o = mean_error_g2o;
    err = mean_error;
end

