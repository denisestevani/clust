# # -*- coding: utf-8 -*-
# """
# Created on Thu Dec 23 11:21:18 2021

# @author: Denise
# """
# import numpy as np
# import pandas as pd
# # import pandas as pd
# # import numpy as np
# import random
# import functools
# import time
# from numpy import linalg as LA
# from numpy import nan


# def eucledian(raw_p, raw_q):
#     p = np.array(raw_p)
#     q = np.array(raw_q)

#     return np.sqrt(np.sum((p-q)**2))

# # def eucledian(raw_p, raw_q):
# #     p = np.array(raw_p)
# #     q = np.array(raw_q)

# #     return np.sqrt(np.dot(p, p) - 2 * np.dot(p, q) + np.dot(q, q))


# def manhattan(raw_p, raw_q):
#     x = np.array(raw_p)
#     y = np.array(raw_q)

#     return np.abs(x-y).sum()

# def cosine(raw_p, raw_q):
#     a = np.array(raw_p)
#     b = np.array(raw_q)
#     result = np.dot(a, b)/(LA.norm(a)*LA.norm(b))
#     return result

# class KMeans():
#     def __init__(self, data: pd.DataFrame, k):
#         self.result = None
#         self.data = data
#         self.distance = eucledian
#         self.k = k
#         self.iteration =3
#         #self.Centroids = data

#     @classmethod
#     # def eucledian(cls, data: pd.DataFrame, k=3):
#     #     cluster = cls(data, k)
#     #     cluster.distance = eucledian
#     #     return cluster
    
#     def eucledian(cls, data: pd.DataFrame, k=3):
#         cluster = cls(data, k)
#         cluster.distance = eucledian
#         return cluster
    
#     @classmethod
#     def manhattan(cls, data: pd.DataFrame, k=5):
#         cluster = cls(data, k)
#         cluster.distance = manhattan
#         return cluster

#     @classmethod
#     def cosine(cls, data: pd.DataFrame, k=3):
#         cluster = cls(data, k)
#         cluster.distance = cosine
#         return cluster

#     def run(self):

#         Ny = self.data.shape[0]  # Data rows m
#         Nx = self.data.shape[1]  # n data points

#         Centroids = np.array([]).reshape(Nx, 0)

#         for i in range(self.k):
#             rand = random.randint(0, Ny-1)
#             # Pick random points
#             Centroids = np.c_[Centroids, self.data[rand]]
            
#         clusters = {}
#         clustersIdx = {}

#         #print('Start KMeans using k = ', self.k)
#         #print('Start KMeans using k = ', self.iteration)
#         for i in range(self.iteration):
#             #print('Kmeans Iteration', ' ', i)
#             Distance = np.array([]).reshape(Ny, 0)
#             for k in range(self.k):
#                 #print('Kmeans Iteration(distance) = ', ' ', k)
#                 tempDistance = []
#                 for item in self.data:
#                     tempDistance.append(self.distance(item, Centroids[:, k]))
#                 Distance = np.c_[Distance, np.array(tempDistance)]

#             C = np.argmin(Distance, axis=1)+1

#             Y = {}
#             I = {}
#             for k in range(self.k):
#                 Y[k+1] = np.array([]).reshape(Nx, 0)
#                 I[k+1] = []
#             for i in range(Ny):
#                 Y[C[i]] = np.c_[Y[C[i]], self.data[i]]
#                 I[C[i]].append(i)

#             for k in range(self.k):
#                 Y[k+1] = Y[k+1].T

#             for k in range(self.k):
#                 Centroids[:, k] = np.mean(Y[k+1], axis=0)
#                 #print(Centroids)
                
#             clustersIdx = I
#         self.result = clustersIdx
        
#         #self.result = Centroids
#         #newlist = [x for x in self.result() if np.isnan(x) == False]
#         #self.result = Centroids.astype(int)
# def main():
#   # Simpan seed random agar hasil sama kalau diulang lagi
#   random.seed(1)
#   # Baca data
#   data = pd.read_csv('dummyData.csv')

#   # Label data
#   data_labels = data.values[:,:1].flatten()
#   # Data matriks
#   values = data.values[:, 1:]

#   start_time = time.time()
#   # Inisialisasi K-Means dengan k = 4 dan menggunakan Euclidean
#   euclidean_kmean = KMeans.eucledian(values, k=3)
#   # Jalankan algoritma
#   euclidean_kmean.run()

#   #print(euclidean_kmean.result)

#   for cluster in euclidean_kmean.result:
#     print([data_labels[item] for item in euclidean_kmean.result[cluster]])
#     #print([data_labels[item] for item in euclidean_kmean.result[cluster]])
    
    


import time
import pandas as pd
import numpy as np

import methods as m
from sklearn.decomposition import PCA


class Hierarchical:
    def __init__(self, datapoints: np.array, label, indexes, child: tuple = None, distance=0):
        self.datapoints = datapoints
        self.label = label
        self.distance = distance
        self.indexes = indexes
        self.child = child

    @staticmethod
    def castObject(obj: 'Hierarchical'):
        return obj


def flatten(*n):
    return (e for a in n for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))


class Clusters:
    def __init__(self, data: pd.DataFrame):
        label = data.values[:, [0]].flatten()
        values = data.values[:, 1:]
        pca = PCA(n_components=3)
        values = pca.fit_transform(values)
        values = pd.DataFrame(values)
        # values.columns = ['P1', 'P2', 'P3']
        values = values.values[:, :]
        N = len(values)
        self.clusters = np.array(
            [Hierarchical(values[i], label[i], [i]) for i in range(N)])
        self.distance = m.cosine

    @staticmethod
    def joinClusters(cluster1: Hierarchical, cluster2: Hierarchical, distance):
        return Hierarchical((
            cluster1.datapoints+cluster2.datapoints)/2,
            [list(flatten(cluster1.label)), list(flatten(cluster2.label))],
            cluster1.indexes + cluster2.indexes,
            (cluster1, cluster2),
            distance
        )

    @staticmethod
    def splitClusters(cluster: Hierarchical, k):
        result = [cluster]

        curr = k
        while curr > 1:
            find = [item for item in result if item.distance > 0]
            item = find[0]
            result.remove(item)
            result += list(item.child)
            curr -= 1
        return result

    def run(self):
        if len(self.clusters) < 1:
            return
        tempClusters = []

        print(f'First scan: {len(self.clusters)}')

        start_time = time.time()
        Clusters.joinClusters(
            self.clusters[0], self.clusters[1],
            self.distance(self.clusters[0].datapoints,
                          self.clusters[1].datapoints)
        )
        end_time = time.time() - start_time
        N = len(self.clusters)
        print(f'Estimated done in {end_time*N*(N+1)/2} s')
        tempClusters = np.array([
            Clusters.joinClusters(
                self.clusters[i],
                self.clusters[j],
                self.distance(
                    self.clusters[i].datapoints, self.clusters[j].datapoints)
            ) for i in range(N-1) for j in range(i+1, N)
        ])
        start_time = time.time()
        while len(self.clusters) > 1:
            print('Len of cluster: ' + str(len(self.clusters)))
            distances = [item.distance for item in tempClusters]
            smallest = tempClusters[distances.index(min(distances))]
            smallest = Hierarchical.castObject(smallest)

            for child in smallest.child:
                self.clusters = np.delete(
                    self.clusters, np.where(self.clusters == child))
                tempClusters = np.delete(
                    tempClusters, [child in item.child for item in tempClusters])
            self.clusters = np.append(self.clusters, smallest)
            new_joins = np.array([
                Clusters.joinClusters(smallest, item, self.distance(
                    smallest.datapoints, item.datapoints))
                for item in self.clusters[:-1]])
            tempClusters = np.concatenate((tempClusters, new_joins))
            # return tempClusters
        print(f'Took {time.time()-start_time}s')


data = pd.read_csv('dummyData.csv')

label = data.values[:, [0]].flatten()
values = data.values[:,1:]
start_time = time.time()
pca = PCA(n_components=3)
values = pca.fit_transform(values)
values = pd.DataFrame(values)
values.columns = ['P1', 'P2', 'P3']

values_sum = np.sum(values, axis=0)/len(values)
# print(data.head())
clusts = Clusters(data)
clusts.run()
print(f'Took {time.time()-start_time}s')
clusters = Hierarchical.castObject(clusts.clusters[0])
# for i in range(1, len(values)):
#   j = 0
#   print(f'Splitting cluster into {i}')
#   for item in Clusters.splitClusters(clusters, i):
#     j+=1
#     print(f'Cluster {j}: Range = {np.sqrt(m.eucledian(values_sum, item.datapoints))}')
#     print(item.label)
  # print(item.label,  file=open("outputHirar_test.txt", "a"))

n_cluster=3
j=0
for item in Clusters.splitClusters(clusters, n_cluster):
  # lines=['Cluster {j+1} : \n {item.label}']
  # print(lines)
  # print(item.label)
  with open('hasil3.txt', 'a') as f:
    f.writelines('Cluster {} \n {} \n'.format(j, item.label))
  j +=1
