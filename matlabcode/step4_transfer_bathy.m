clear all;clc;
InFileName='E:\xcsdata\xisha\controlpoints\20200310\step4_control\envi_20200310_bathymetry_msk_mean.tif';   %反演得到的水深图
model_matfile='E:\xcsdata\xisha\controlpoints\20200310\step4_control\envi_20200310_bathymetry_msk_mean.mat';    %转成tif格式
transfer_tif_mat(InFileName,model_matfile);
    
InFileName='E:\xcsdata\virgin\lidar\subset_2_of_lidar.tif';   %反演得到的水深图
saveFilename = 'E:\xcsdata\virgin\lidar\subset_2_of_lidar.mat';  %转成tif格式
transfer_tif_mat(InFileName,saveFilename)

    