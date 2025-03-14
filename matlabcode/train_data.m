clc
close all
msk_path =['H:\codes\image\msk\'];
msk_name = [msk_path,'huaguangjiao_msk.mat'];
load(msk_name);

allpath = ['H:\codes\controlpoints\'];
contropoint_file = [allpath,'step4\control_points.csv'];
csv_data = csvread(contropoint_file);
%该文件共 5层 B_1--NIR B_2--land mask B_3--B2 blue B_4--B3 green B_5--B4 red
[sv_tb,B] = grid_sentinel_all(LON, LAT,csv_data,A);  %
y = sv_tb(:,3);
train=[B(:,5),B(:,4),B(:,3),B(:,1),B(:,1),B(:,1),y];
tmp=train(:,1);
Inx=isnan(tmp);
n=numel(tmp(Inx));
train(Inx,:)=repmat(train(423,:),[n,1]);
train(1594:1600,:)=repmat(train(1593,:),[7,1]);
save('train.txt','train','-ascii');