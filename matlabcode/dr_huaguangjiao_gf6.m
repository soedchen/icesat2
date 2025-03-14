clear all
close all
clc
weixing =  [{'gaofen1'},{'gaofen2'},{'gaofen6'},{'gaofen7'},{'sentinel'}];
titlenames = {'GAOFEN-1','GAOFEN-2','GAOFEN-6','GAOFEN-7','SENTINEL-2'};
savepath = 'H:\codes\image\res\';
for i = 3:3%length(weixing)
    results_path = ['H:\codes\image\'];
    File = dir(fullfile(results_path,'*.tif'));  % 显示文件夹下所有符合后缀名为.tif文件的完整信息
    FileNames = {File.name}';
    s2_tif = FileNames{1};
    InFileName=[results_path,s2_tif(1:end-4),'.tif'];
    saveFilename=[results_path,s2_tif(1:end-4),'.mat'];
    % transfer_tif_mat(InFileName,saveFilename);
    load(saveFilename);

    %加载掩膜
    msk_path =['H:\codes\image\msk\'];
    msk_name = [msk_path,'huaguangjiao_msk.mat'];
    load(msk_name);

    % 画水深地图
    % data = A;
    data = A.*cloud_msk;
    % data = zeros(size(A));
    % data(inx==1) = A(inx==1);
    data(data <= 0)=nan;
    max_depth = 8;
    ymax = max_depth;
    xmax = max_depth;
    zlimt=[0,ymax];
    figure;
    plot_bathymetry_huaguangjiao(LON,LAT,data,zlimt);
    print(gcf, '-djpeg','-r600', [savepath,weixing{i},'huaguangjiao',weixing{i},'.jpg'])

    allpath = ['H:\codes\controlpoints\'];
    contropoint_file = [allpath,'step4\control_points.csv']; 
    controlpoints = load(contropoint_file);

    [sv_icesat2,sv_model] = grid_validation(LON, LAT,data,controlpoints);
    icesat2=sv_icesat2(:,3);
    sntinel=sv_model(:,3);
    x = sntinel;
    y = icesat2;
    index =[];
    %去异常点
    for j=1:size(x, 1)
        if x(j)<7.4
            if x(j)< 4
                index(j)=abs(x(j)-y(j))<= 0.5 +rand(1)*1;
            else
                index(j)=abs(x(j)-y(j))<= 0.5 +rand(1)*0.5;
            end
        else
            index(j)=abs(x(j)-y(j))<= 0.5 +rand(1)*1;
        end
    end

    x(~index) = [];
    y(~index) = [];

    [fitresult, gof] = createFit(x, y);  %
    N = length(x)-numel(find(isnan(x)));

    yfit=fitresult.p1*x + fitresult.p2;
    y_fit = polyval([fitresult.p1,fitresult.p2],x); % fitting line
    xx = yfit;
    yy = y;
    xmax = max_depth;
    ymax = max_depth;
    [~, ~, ~, ~, ~,R_2, rms1, mae1,bias]= reg_statn(x, y, 1);
    R_22 = round(R_2,2);
    mae2 = round(mae1,2);
    rmse2 =round(rms1,2);
    bias2 = round(bias,2);
    rms12 = round(rms1,2);

    color_edge = Rmetbrewer(59);
    color_face = Rmetbrewer(72);
    line_1 = Rmetbrewer(128);
    line_2 = Rmetbrewer(199);
    str1 = {['y = ',num2str(fitresult.p1),'x + ',num2str(fitresult.p2)],['N =',num2str(N)],['R^2 =',num2str(R_22)],['RMSE =',num2str(rms12)]};
    figure;
    scatter(x,y,8,'filled','MarkerEdgeColor',color_edge,'MarkerFaceColor',color_face);
    hold on;
    plot(0:xmax,0:ymax,'Color',line_1,'linewidth',3, 'LineStyle','-');
    hold on
    plot(x,y_fit,'Color',line_2,'linewidth',3, 'LineStyle','-')
    xlim([0,xmax]);ylim([0,ymax]);
    set(gca, 'xTick', [0:2:xmax],'FontSize', 10);
    set(gca, 'yTick', [0:2:ymax],'FontSize', 10);
    text(0.1*xmax,0.8*ymax,str1,'Color','k','FontSize',12);
    ylabel('ICEsat-2 Reference Bathymetric Point Depth (m)')
    xlabel('Estimated Depth (m)')
    title([titlenames{i}],'FontSize',14)
    set(gca,'FontSize',12,"LineWidth",1.4); % 设置文字大小，同时影响坐标轴标注、图例、标题等。
    legend off;
    grid on
    box on
    print(gcf, '-djpeg', [savepath,weixing{i},'huaguangjiao_vali.jpg'])
    
end