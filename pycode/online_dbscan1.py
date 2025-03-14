import math
import copy
import numpy as np
from sklearn.cluster import DBSCAN
import sklearn.metrics.pairwise as pairwise
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

class Adapter_DBSCAN():

    def returnEpsCandidate(self, dataSet,N1,N2):
        """
        :param dataSet: 数据集
        :return: eps候选集合
        """
        # self.DistMatrix = self.CalculateDistMatrix(dataSet)
        self.DistMatrix = pairwise.euclidean_distances(dataSet)
        tmp_matrix = copy.deepcopy(self.DistMatrix)
        for i in range(len(tmp_matrix)):
            tmp_matrix[i].sort()
        EpsCandidate = []
        for k in range(1, len(dataSet)):
            # Dk = self.returnDk(tmp_matrix,k)
            Dk = tmp_matrix[:, k]
            # DkAverage = self.returnDkAverage(Dk)
            # 快160+倍
            DkAverage = np.mean(Dk)
            EpsCandidate.append(DkAverage)
        #初始化eps
        if round(N1/N2) > 20:
            EpsCandidate = [x for x in EpsCandidate if x >= 0.2]
        else:
            EpsCandidate = [x for x in EpsCandidate if x >= 0.5]
        return EpsCandidate

    def returnMinptsCandidate(self,N1,N2,range_gate,l,n_s,n_n):

        # return: Minpts候选列表
        MinptsCandidates = []
        for i in range(len(self.EpsCandidate)):
            Ra = self.EpsCandidate[i]
            SN1 = (math.pi * (float(Ra) ** 2) * float(N1)) / (range_gate * l * n_s)
            #print('SN1 = ',SN1)
            SN2 = (math.pi * (float(Ra) ** 2) * float(N2)) / (range_gate * l * n_n)
            #print('SN2 =',SN2)
        # MinPts = round((( SN1 - SN2) + math.log(M))/math.log( SN1 / SN2 ))
            MinptsCandidates.append(round(((SN1 - SN2)) / math.log(SN1 / SN2)))

        return MinptsCandidates

    def fit(self, X, N1, N2, range_gate, l, n_s, n_n):
        self.EpsCandidate = self.returnEpsCandidate(X, N1, N2)
        self.MinptsCandidate = self.returnMinptsCandidate(N1,N2,range_gate,l,n_s,n_n)
        # self.do_multi_dbscan(X)
    def fit1(self, X, N1, N2, range_gate, l, n_s, n_n):
        self.EpsCandidate = self.returnEpsCandidate1(X,N1, N2)
        self.MinptsCandidate = self.returnMinptsCandidate(N1,N2,range_gate,l,n_s,n_n)
    def all_dbscan(self, X):
        self.all_num_clusters = []

        for i in range(1, len(self.EpsCandidate)):
            eps = self.EpsCandidate[i]
            minpts = self.MinptsCandidate[i]
            db = DBSCAN(eps=eps, min_samples=minpts).fit(X)
            num_clusters = max(db.labels_) + 1
            # 统计符合范围的聚类情况
            self.all_num_clusters.append(num_clusters)
        return self.all_num_clusters

    def pl_nums(self,X):
        Epss = self.EpsCandidate
        all_num_clusters = np.array(self.all_dbscan(X))
        len_Epss = len(Epss)
        fig, axe = plt.subplots(figsize=[8, 6])
        font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20, }
        axe.plot(Epss[1:len_Epss], all_num_clusters, 'k--')
        plt.xlabel('Eps', font2)
        plt.xscale('log')
        plt.ylabel('Cluster Number', font2)
        plt.show()

    def do_multi_dbscan(self, X):
        flag = 1
        last_N = 0
        K = int(1)
        last_K = 1
        last_eps = 0
        last_minpts = 0
        jud = 3
        len_E = int(np.array(len(self.EpsCandidate)))
        while K <= len_E:
            eps = self.EpsCandidate[K]
            minpts = self.MinptsCandidate[K]
            db = DBSCAN(eps=eps, min_samples=minpts).fit(X)
            num_clusters = max(db.labels_) + 1
            # 统计符合范围的聚类情况
            if num_clusters == last_N:
                flag = flag + 1
                if flag == jud:
                    while (num_clusters == last_N):
                        K = last_K + 1
                        if K < len_E:
                          
                            num_clusters = max(db.labels_) + 1
                            if num_clusters != last_N:
                                break
                        else:
                            print("wrong!")
                            last_K = K - 1
                            last_eps = 0.4
                            last_minpts = 3
                            break
                        last_K = K
                        last_eps = eps
                        last_minpts = minpts
                    break
                else:
                    K = int((K + last_K) / 2)
                    last_K = K
            else:
                last_K = K
                K = int(K + 2)
                flag = 1
                last_N = num_clusters
        opt_K = last_K
        opt_eps = last_eps
        opt_minpts = last_minpts
        return opt_K, opt_eps,opt_minpts

'''
if __name__ == '__main__':
    start = time.time()
    cloud = np.load("11223.npy")
    cloud1 = cloud[0:1000, 0:2]
    DB = Adapter_DBSCAN()
    N1 = 9803
    N2 = 197
    range_gate = 10
    l = 12
    n_s = 3
    n_n = 3
    # initial parameters
    DB.fit(cloud1, N1, N2, range_gate, l, n_s, n_n)
    # 输出最优解
    ops_K, opt_eps,opt_minpts = DB.do_multi_dbscan(cloud1)
    end = time.time()
    print("optimise times = ", end - start)
    print('ops_K = %f,opt_eps = %f,opt_minpts = %d' % (ops_K, opt_eps,opt_minpts))

    DB.pl_nums(cloud1)
    end1 = time.time()
    print("times = ", end1 - end)
'''