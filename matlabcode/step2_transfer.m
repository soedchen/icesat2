clc
clear all
close all
Path = 'C:\Users\Administrator\Desktop\代码打包\controlpoints\step2_outlier\';                   % 待数据存放的文件夹路径
savePath ='C:\Users\Administrator\Desktop\代码打包\controlpoints\step3_grid\';     %处理结束数据存放文件夹路径
addpath(Path)

%% 合并outlier
File = dir(fullfile(Path,'*.csv'));  % 显示文件夹下所有符合后缀名为.csv文件的完整信息
FileNames = {File.name}';
LengthFiles = length(FileNames);

pre_dat=[];
for i = 1 : LengthFiles
    FileName= FileNames{i};
    %计算部分
    title_name = FileName(4:19);
    all_data = load(FileName);
    tmp=all_data;
    pre_dat=[pre_dat;tmp];
end
savename = [savePath,'pregrid_all', '.csv'];
writematrix(pre_dat, savename);

%% 转tif
InFileName='E:\xcsdata\virgin\sentinel\subset_reprojection\201901115_virgin.tif';
saveFilename='E:\xcsdata\virgin\sentinel\subset_reprojection\201901115_virgin.mat';
transfer_tif_mat(InFileName,saveFilename);