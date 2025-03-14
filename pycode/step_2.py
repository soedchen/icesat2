import numpy as np
import matplotlib.pyplot as plt
import bisect
import math
import sympy
import collections
import pandas as pd
from getAtlMeasuredSwath_auto import getAtlMeasuredSwath
from getAtlTruthSwath_auto import getAtlTruthSwath
from getMeasurementError_auto import getMeasurementError, offsetsStruct
from sklearn.cluster import DBSCAN
from online_dbscan1 import Adapter_DBSCAN

atl03FilePath = 'G:\\xcsdata\\virgin\\ICEsat2\\ATL03_20201213060025_12230901_004_01.h5'
outFilePath = 'G:\\xcsdata\\virgin\\ICEsat2\\' # 文件存放位置
gtNum = 'gt1l' #处理的轨道数
daytime = atl03FilePath[-33:-25]
n_time = outFilePath+'it' + atl03FilePath[11:15] + daytime + str(gtNum) + '.npy'
n_at =  outFilePath+'iat' + atl03FilePath[11:15] + daytime + str(gtNum) + '.npy'
n_height =  outFilePath+'ih' + atl03FilePath[11:15] + daytime + str(gtNum) + '.npy'
n_sc =  outFilePath+'isc' + atl03FilePath[11:15] + daytime + str(gtNum) + '.npy'
n_lon = outFilePath+ 'ilon' + atl03FilePath[11:15] + daytime + str(gtNum) + '.npy'
n_lat =  outFilePath+'ilat' + atl03FilePath[11:15] + daytime + str(gtNum) + '.npy'


time = np.load(n_time)
alo_tr = np.load(n_at)
yData = np.load(n_height)
signalConf = np.load(n_sc)
lon = np.load(n_lon)
lat = np.load(n_lat)

#初始化参数
min_time = 285.5
max_time = 135
min_alo = 20.325e5
max_alo = 0
max_y = -40
min_y = -80
seg_num = 9000 #each segment length
seg_size = 4
seg_totalnum = seg_size*seg_num + 1000
seg_totalnum1 = seg_size*seg_num + 1000
range_gate = 5  #越大检测细微信号点效果越好


min_xlim1 = bisect.bisect(alo_tr, min_alo)  #有序序列的查找。

#alongtrack data segement
seg_start1 = min_xlim1
# total participate cal
x_Data2 = alo_tr[seg_start1:seg_totalnum1+seg_start1]
y_Data2 = yData[seg_start1:seg_totalnum1+seg_start1]
time_d = time[seg_start1:seg_totalnum1+seg_start1]
ocean_signal_conf_ph2 = signalConf[seg_start1:seg_totalnum1+seg_start1]
lon_Data2 = lon[seg_start1:seg_totalnum1+seg_start1]
lat_Data2 = lat[seg_start1:seg_totalnum1+seg_start1]
y_Data3 = [y_Data2 for y_Data2 in y_Data2[0:] if y_Data2 > min_y and y_Data2 < max_y]
# y index with alongtrack
y_index1 = np.where(np.logical_and(y_Data2 > min_y, y_Data2 < max_y) )
time_d3 = time_d[y_index1]
y_Data33 = y_Data2[y_index1]
x_Data3 = x_Data2[y_index1]  # y limitation
ocean_signal_conf_ph3 = ocean_signal_conf_ph2[y_index1]
lon_Data3 = lon_Data2[y_index1]
lat_Data3 = lat_Data2[y_index1]

N_loop = int(math.floor( len(y_Data3) )/ seg_num)
M = int((max_y - min_y)/ range_gate)
time_Data4 = time_d3[1:N_loop*seg_num+1]
x_Data4 = x_Data3[1:N_loop*seg_num+1]
y_Data4 = y_Data3[1:N_loop*seg_num+1]
y_Data44 = y_Data33[1:N_loop*seg_num+1]
ocean_signal_conf_ph4 = ocean_signal_conf_ph3[1:N_loop*seg_num+1]
lon_Data4 = lon_Data3[1:N_loop*seg_num+1]
lat_Data4 = lat_Data3[1:N_loop*seg_num+1]
font2 = {'family' : 'Times New Roman','weight' : 'normal','size': 15,}
plt.subplot(211)
f1 = plt.plot(time_Data4,y_Data44,'.',color='k', markersize=4)
plt.xlabel('Time(sec)', font2)
plt.subplot(212)
f2 = plt.plot(x_Data4,y_Data44,'.',color='k', markersize=4)
plt.xlabel('Alongtrack(m)', font2)
plt.ylabel('Height(m)', font2)
#plt.savefig('along_height.jpg')


times = (max(x_Data3) - min(x_Data3)) / (max(y_Data3) - min(y_Data3))#alongtrack轴缩小的倍数

core_samples_mask1 = []
labels1 = []

# 海面以上光子
up_xt1 = []
up_yt1 = []
# 海面光子
sur_xt1 = []
sur_yt1 = []
 #海底光子
dep_xt1 = []
lat_t1 = []
lon_t1 = []
dep_yt1 = []
cor_dep_yt1 = []
cor_d_yt1 = []

for i in range(0 , N_loop):
    y_temp = y_Data3[i * seg_num + 1:i * seg_num + seg_num+1]
    x_temp = x_Data3[i * seg_num + 1:i * seg_num + seg_num+1]
    lon_temp = lon_Data3[i * seg_num + 1:i * seg_num + seg_num+1]
    lat_temp = lat_Data3[i * seg_num + 1:i * seg_num + seg_num+1]
    signal_conf_temp = ocean_signal_conf_ph3[i * seg_num + 1:i * seg_num + seg_num+1]
    x_temp = x_temp/times
    clouds = np.concatenate((np.array(x_temp[:, np.newaxis]), np.array(y_temp)), axis=1)
    l = max(x_temp) - min(x_temp)# the along track range of each segment
    choose_signal_conf_ph = np.where(signal_conf_temp == 4)
    y_temp = np.array(y_temp)
    y_temp2 = y_temp[choose_signal_conf_ph]
    first_edge, last_edge = y_temp2.min(), y_temp2.max()
    bin_num = round((last_edge - first_edge) / 0.1)
    # return frequency, bin boundary
    hist, bin_edges = np.histogram(y_temp2, bins=bin_num, range=None, weights=None, density=False)
    index_sur = np.argmax(hist)
    y_d = bin_edges[index_sur]
    y_u = bin_edges[index_sur + 1]
    y_suf = (y_d + y_u) / 2
    print('histogram 海表下层%f - 上层 %f,海平面 %f' % (y_d, y_u, y_suf))
    y_up = y_suf + 1
    y_down = y_suf - 1

    y_higher = [y_temp for y_temp in y_temp[0:] if y_temp >= y_down]
    print('海表以上的光子数量： %d' %len(y_higher))
    N_tot = len(y_temp) - len(y_higher)# the number of photons in total
    y_mean = (N_tot*range_gate / ((max_y - min_y)-(y_up-y_down)))
    y_c1 = 0
    y_c2 = 0
    n_s = 0
    n_n = 0
#Batch name data and assign value
    for ii in range(1,M+1):
        y_h = []
        y_h = [y_temp for y_temp in y_temp[0:] if y_temp >= min_y+range_gate * (ii -1) and y_temp < min_y+ range_gate*ii]
        y_num = len(y_h)
        if y_num >= y_mean:
            y_c1 = y_c1 + y_num
            n_s = n_s + 1
        else:
            y_c2 = y_c2 + y_num
            n_n = n_n +1
    N1 = y_c1
    print('N1=%d N_s= %d' %(N1,n_s))
    N2 = y_c2
    print("N2=%d N_n = %d" % (N2,n_n))
    DB = Adapter_DBSCAN()
    DB.fit(clouds, N1, N2, range_gate, l, n_s, n_n)
    # 输出最优解
    ops_K, opt_eps, opt_minpts = DB.do_multi_dbscan(clouds)
    print('ops_K = %f,opt_eps = %f,opt_minpts = %d' % (ops_K, opt_eps,opt_minpts))
    MinPts = max (3, opt_minpts)
    print('Minpts = %d' %MinPts)
    db = DBSCAN(opt_eps, MinPts).fit(clouds)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    # 拼接结果
    core_samples_mask1 = np.concatenate((core_samples_mask1, core_samples_mask), axis=0)
    labels = db.labels_
    # 拼接结果
    labels1 = np.concatenate((labels1, labels), axis=0)

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]

    lat_t = []
    lon_t = []
    up_xt = []
    up_yt = []
    dep_xt = []
    dep_yt = []
    cor_dep_yt = []
    cor_d_yt = []
    sur_xt = []
    sur_yt = []
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # 聚类结果为-1的样本为离散点
            # 使用黑色绘制离散点
            col = [0, 0, 0, 1]
        # 将所有属于该聚类的样本位置置为true
        class_member_mask = (labels == k)

        # 提取海面以上的点
        xy = clouds[class_member_mask & core_samples_mask]
        xy_y = xy[:, 1]
        xy_x = xy[:, 0]
        xy_index = np.where(xy_y > y_up)
        xy_x1 = xy_x[xy_index]
        up_xt = np.concatenate((up_xt, xy_x1))
        xy_y1 = np.array([xy_y for xy_y in xy_y[0:] if xy_y > y_up])
        up_yt = np.concatenate((up_yt, xy_y1))
        # 绘制海面以上光子
        plt.plot(xy_x1 * times, xy_y1, 'o', markerfacecolor='#FF1493', markeredgecolor='#FF1493', markersize=1)

        # 提取海面的点
        xy = clouds[class_member_mask & core_samples_mask]
        xy_fy = xy[:, 1]
        xy_fx = xy[:, 0]
        xy_findex = np.where(np.logical_and( xy_fy <= y_up, xy_fy >= y_down))
        xy_x2 = xy_fx[xy_findex] # y limitation
        xy_y2 = np.array([xy_fy for xy_fy in xy_fy[0:] if xy_fy >= y_down and xy_fy <= y_up])
        sur_yt = np.concatenate((sur_yt, xy_y2))
        sur_xt = np.concatenate((sur_xt, xy_x2))
        # 绘制海面光子
        plt.plot(xy_x2 * times, xy_y2, 'o', markerfacecolor='b', markeredgecolor='b', markersize=1)

        #提取海底的点
        xy = clouds[class_member_mask & core_samples_mask]
        lon1 = lon_temp[class_member_mask & core_samples_mask]
        lat1 = lat_temp[class_member_mask & core_samples_mask]
        xy_y = xy[:, 1]
        xy_x = xy[:, 0]
        xy_index = np.where(xy_y < y_down)
        xy_x3 = xy_x[xy_index]
        lon1 = lon1[xy_index]
        lat1 = lat1[xy_index]
        xy_y3 = np.array([xy_y for xy_y in xy_y[0:] if xy_y < y_down])

        #水底校正
        cor_xy_y3 = xy_y3 + 0.25416 * (y_suf - xy_y3 )
        cor_xy_y4 = y_suf - cor_xy_y3
        dep_xt = np.concatenate((dep_xt, xy_x3))
        dep_yt = np.concatenate((dep_yt, xy_y3))
        cor_dep_yt = np.concatenate((cor_dep_yt, cor_xy_y3))
        cor_d_yt = np.concatenate((cor_d_yt, cor_xy_y4))
        #lat- lon - depth
        lat_t = np.concatenate((lat_t, lat1))
        lon_t = np.concatenate((lon_t, lon1))
        # 绘制海底光子
        plt.plot(xy_x3 * times , xy_y3, 'o', markerfacecolor='r', markeredgecolor='r', markersize=1)
        # 绘制矫正后海底光子
        #plt.plot(xy_x3 * times, cor_xy_y3, 'o', markerfacecolor='#FFFF00', markeredgecolor='#FFFF00', markersize=1)
        # 绘制噪声光子
        xy = clouds[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0] * times , xy[:, 1], 'o', markerfacecolor='k', markeredgecolor='k', markersize=1)

    plt.xlabel('Alongtrack(m)', font2)
    plt.ylabel('Height(m)', font2)
    plt.title('Mintpoints: %d,Eps:%f' % (MinPts, opt_eps))
    

    # 拼接海面以上光子位置
    up_xt1 = np.concatenate((up_xt1, up_xt))
    up_yt1 = np.concatenate((up_yt1, up_yt))
    # 拼接海面光子位置
    sur_xt1 = np.concatenate((sur_xt1, sur_xt))
    sur_yt1 = np.concatenate((sur_yt1, sur_yt))
    # 拼接海底光子位置
    dep_xt1 = np.concatenate((dep_xt1, dep_xt))
    dep_yt1 = np.concatenate((dep_yt1, dep_yt))
    cor_dep_yt1 = np.concatenate((cor_dep_yt1, cor_dep_yt))
    cor_d_yt1 = np.concatenate((cor_d_yt1, cor_d_yt))
    lat_t1 = np.concatenate((lat_t1, lat_t))
    lon_t1 = np.concatenate((lon_t1, lon_t))

clouds1 = np.concatenate((np.array(x_Data4[:, np.newaxis]), np.array(y_Data4)), axis=1)
n_clusters_ = len(set(labels1)) - (1 if -1 in labels1 else 0)
n_noise_ = list(labels1).count(-1)

# Black removed and is used for noise instead.
unique_labels = set(labels1)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # 聚类结果为-1的样本为离散点
        # 使用黑色绘制离散点
        col = [0, 0, 0, 1]
        # 将所有属于该聚类的样本位置置为true
    class_member_mask = (labels1 == k)
    core_samples_mask1 = (core_samples_mask1 > 0.5)
    xy1 = clouds1[class_member_mask & ~core_samples_mask1]
    # 将所有属于该类的非核心样本取出，使用小图标绘制
    plt.plot(xy1[:, 0], xy1[:, 1], 'o', markerfacecolor='#6495ED', markeredgecolor='#6495ED', markersize=1)


up_xt1 = (up_xt1.reshape(-1,1))*times
up_yt1 = up_yt1.reshape(-1,1)
sur_xt1 = (sur_xt1.reshape(-1,1))*times
sur_yt1 = sur_yt1.reshape(-1,1)
dep_xt1 = (dep_xt1.reshape(-1,1))*times
dep_yt1 = dep_yt1.reshape(-1,1)
cor_dep_yt1 = cor_dep_yt1.reshape(-1,1)
cor_d_yt1 = cor_d_yt1.reshape(-1,1)

# 绘制海面以上光子
plt.plot(up_xt1, up_yt1, 'o', markerfacecolor='#EE82EE', markeredgecolor='#EE82EE', markersize=1)
# 绘制海面光子
plt.plot(sur_xt1 , sur_yt1, 'o', markerfacecolor='#0000CD', markeredgecolor='#0000CD', markersize=1)
# 绘制海底光子
plt.plot(dep_xt1, dep_yt1, 'o', markerfacecolor='#DC143C', markeredgecolor='#DC143C', markersize=1)
# 绘制矫正后海底光子
plt.plot(dep_xt1, cor_dep_yt1, 'o', markerfacecolor='#FFA500', markeredgecolor='#FFA500', markersize=1)
# 绘制矫正后海底光子深度
#plt.plot(dep_xt1*times, cor_d_yt1, 'o', markerfacecolor='#FFFF00', markeredgecolor='#FFFF00', markersize=1)
font2 = {'family' : 'SimHei','weight' : 'normal','size': 15}
font3 = {'family' : 'SimHei','weight' : 'normal','size': 18}
plt.ylabel('Height(m)', font2)
title_name = atl03FilePath[-39:-3]
plt.title(title_name, font3)
plt.show()
"""
fig_path = 'E:\\xcsdata\\论文配图\\fig18\\'
dt_name = fig_path + atl03FilePath[8:12] +daytime + str(gtNums[0])+'.csv'
#save origin data
origin_Data = (time_Data4,x_Data4,y_Data44)
origin_Data = pd.DataFrame(origin_Data)
origin_Data.to_csv(dt_name,index=False,header=0)

up_data =  pd.DataFrame(np.concatenate((up_xt1,up_yt1),axis = 1))
surface_data = pd.DataFrame( np.concatenate((sur_xt1,sur_yt1),axis = 1))
floor_data = pd.DataFrame(np.concatenate((dep_xt1,dep_yt1,cor_dep_yt1,cor_d_yt1),axis = 1))
up_name = fig_path + 'up_' + atl03FilePath[8:12] + daytime + str(gtNums[0]) +'.csv'
sur_name = fig_path + 'sur_' + atl03FilePath[8:12] + daytime + str(gtNums[0]) +'.csv'
floor_name = fig_path + 'floor_' + atl03FilePath[8:12] + daytime + str(gtNums[0]) +'.csv'
up_data.to_csv(up_name,index=False,header=0)
surface_data.to_csv(sur_name,index=False,header=0)
floor_data.to_csv(floor_name,index=False,header=0)
"""
lat_t1 = lat_t1.reshape(-1,1)
lon_t1 = lon_t1.reshape(-1,1)
cor_d_yt1 = cor_d_yt1.reshape(-1,1)
dep_xt1 = dep_xt1.reshape(-1,1)
lalode = np.concatenate((lat_t1,lon_t1,dep_xt1,cor_d_yt1),axis = 1)
save_csv = pd.DataFrame(lalode)
dt_name = 'C:\\Users\\Administrator\\Desktop\\代码打包\\pycode\\d_t' + atl03FilePath[8:12] + daytime + str(gtNum[0]) +'.csv'
save_csv.to_csv(dt_name,index=False,header=0)
print('end')