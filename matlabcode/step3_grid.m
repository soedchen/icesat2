clc;clear all;
savePath ='C:\Users\Administrator\Desktop\代码打包\controlpoints\step3_grid\';     %处理结束数据存放文件夹路径
savename = [savePath,'pregrid_all', '.csv'];
saveFilename='E:\xcsdata\virgin\sentinel\subset_reprojection\201901115_virgin.mat';

%% grid to sentinel
load(savename);
load(saveFilename);
[sv_tb] = grid_sentinel(LON, LAT,pregrid_all);   
len = length(sv_tb);
%chose 80% poins to train empirical model and the left to test
rand_ind = randperm(len);
rand_num = floor(rand_ind*0.8);
point_train = sv_tb(rand_ind(1:rand_num),:);
point_test = sv_tb(rand_ind(rand_num+1:end),:);

savePath1 ='E:\xcsdata\virgin\controlpoints\step4_control\';
%所有的控制点
savename = [savePath1,'control_points', '.csv'];
savename_txt = [savePath1,'control_points', '.txt'];
writematrix(sv_tb, savename);
writematrix(sv_tb, savename_txt);
%train and test points
savename = [savePath1,'point_train', '.csv'];
savename_txt = [savePath1,'point_train', '.txt'];
writematrix(point_train, savename);
writematrix(point_train, savename_txt);
savename = [savePath1,'point_test', '.csv'];
savename_txt = [savePath1,'point_test', '.txt'];
writematrix(point_test, savename);
writematrix(point_test, savename_txt);