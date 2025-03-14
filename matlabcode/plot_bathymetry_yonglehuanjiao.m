function h = plot_bathymetry_yonglehuanjiao(lon,lat,data,zlimt)
% 绘图
% LATLIMS=[min(min(lat)) max(max(lat))];
% LONLIMS=[min(min(lon)) max(max(lon))];
% min_lat=min(min(lat));
% max_lat= max(max(lat));
% min_lon=min(min(lon));
% max_lon=max(max(lon));
LATLIMS=[16.413628145086350,16.621748340000000];
LONLIMS=[111.467363970000	111.828802404573];
min_lon=min(min(LONLIMS));
max_lon=max(max(LONLIMS));
min_lat=min(min(LATLIMS));
max_lat= max(max(LATLIMS));
m_proj('lambert','lon',LONLIMS,'lat',LATLIMS);
m_pcolor(lon,lat,data);shading flat;
m_grid('box','fancy','xtick',2,'ytick',4,'linewi',1,'fontsize',14);

%% 颜色定义
map = slanCM('parula');
map = flipud(map);
% colormap(flipud(m_colmap('jet')));
% colormap(flipud('jet'));
colormap(map);
%%定义color bar
h = colorbar('fontsize',12);
set(get(h,'Title'),'string','m','fontsize',11);
set( h,'ticklabels',{'0',(2:2:8),['>',num2str(zlimt(2))]});%设置色度条边上的刻度值
m_ruler([.05 0.15],.85,2,'tickdir','out','ticklen',[.007 .007],'color','w','fontsize',11);
m_northarrow(min_lon+0.25*(max_lon-min_lon), ...
    min_lat+0.85*(max_lat-min_lat),1/50,'type',3,'aspect',1.5);
caxis(zlimt);
% hXLabel = xlabel('Longitude');
% hYLabel = ylabel('Latitude');
% set(gca, 'FontName', 'Helvetica')
% set([hXLabel, hYLabel], 'FontName', 'AvantGarde')
% set(gca, 'FontSize', 10)
% set([hXLabel, hYLabel], 'FontSize', 12)
% 背景颜色
set(gcf,'Color',[1 1 1])
end