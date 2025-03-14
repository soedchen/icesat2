clc
clear all
close all
Path = 'C:\Users\Administrator\Desktop\代码打包\controlpoints\step1_ori\';          % 待数据存放的文件夹路径 
savePath = 'C:\Users\Administrator\Desktop\代码打包\controlpoints\step2_outlier\';     %处理结束数据存放文件夹路径
addpath(Path)

File = dir(fullfile(Path,'*.csv'));  % 显示文件夹下所有符合后缀名为.csv文件的完整信息
FileNames = {File.name}';
LengthFiles = length(FileNames);
marker_size1 = 4; %图标大小
marker_size = 4; %图标大小

for i = 1 : LengthFiles
    FileName= FileNames{i};
    %计算部分
    all_data = csvread(FileName);
    latdata = all_data(:,1);
    londata = all_data(:,2);
    xdata = all_data(:,3);
    ydata = all_data(:,4);
    %小波滤波
    y = ydata;
    wname = 'sym8';
    lev = 3;
    [ydata1,c3,l3,threshold_DJ] = wden(y,'sqtwolog','h','mln',lev,wname);
    
    %沿alongtrack轴去除outlier
    med_y = ydata1;
    med_x = xdata;
    med_lat = latdata;
    med_lon = londata;
    X = [med_y,med_x];
    seg_n = 3; %设置数据处理的分段数
    opts = statset('Display','iter');
    [idx,C,sumd,d,midx,info] = kmedoids(X,seg_n,'Distance','cityblock','Options',opts);

    med_siglat = [];
    med_siglon = [];
    med_sigx = [];
    med_sigy = [];
    for j = 1:seg_n
        tem_lat = med_lat((idx==j));
        tem_lon = med_lon((idx==j));
        tem_x = med_x(idx==j);
        tem_y = med_y(idx==j);

        [tem_x1,outlierIndices] = rmoutliers(tem_x,'movmedian',100);
        tem_y1 = tem_y(~outlierIndices);
        tem_lat1 = tem_lat(~outlierIndices);
        tem_lon1 = tem_lon(~outlierIndices);
        med_siglat = [med_siglat;tem_lat1];
        med_siglon = [med_siglon;tem_lon1];
        med_sigx = [med_sigx;tem_x1];
        med_sigy = [med_sigy;tem_y1];
    end
    
    %% 离群值定义为与中位数相差超过三倍换算 MAD 的元素剔除异常值
    med_y = med_sigy;
    med_x = med_sigx;
    med_lat = med_siglat;
    med_lon = med_siglon;
    [xdata_n, ind]=sort(med_x);
    med_y = med_y(ind);
    med_lat = med_lat(ind);
    med_lon = med_lon(ind);
    med_x = xdata_n;
    [med_sigy1,outlierIndices] = rmoutliers(med_y,'movmedian',50);
    med_sigx1 = med_x(~outlierIndices);
    med_siglat1 = med_lat(~outlierIndices);
    med_siglon1 = med_lon(~outlierIndices);
    
    title_name = FileName(8:19);
    sv_tb = [med_siglat1,med_siglon1,med_sigx1,med_sigy1];
    savename = [savePath,'med_',title_name,'_',num2str(i), '.csv'];
    writematrix(sv_tb, savename);
    
   figure
    subplot(211)
    plot(xdata,ydata,'Marker','.','Color','r','MarkerSize',marker_size, 'LineStyle','none');
    hold on;
    plot(med_sigx,med_sigy,'Marker','o','Color','g','MarkerSize',marker_size1, 'LineStyle','none')
    hold on
    xlabel('AlongTrack')
    ylabel('Depth')
    legend('Original Signal','Outliered Signal','Location','best')
    title([title_name,' X axis Median’s Method '])
    subplot(212)
    plot(xdata,ydata,'Marker','.','Color','r','MarkerSize',marker_size, 'LineStyle','none');
    hold on;
    plot(med_sigx1,med_sigy1,'Marker','o','Color','g','MarkerSize',marker_size1, 'LineStyle','none')
    hold on
    xlabel('AlongTrack')
    ylabel('Depth')
    legend('Original Signal','Outliered Signal','Location','best')

end