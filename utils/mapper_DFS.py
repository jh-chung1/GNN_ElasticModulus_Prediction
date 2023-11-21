"""
Code for Mapper algorithm to covert porous medium to graph data


# Identify clusters in 3D grid
# Based on the implementation by Seunghwa Ryu in MD++
#  IsingFrame::calcrystalorder and IsingFrame::DFS
@article{ryu2010numerical,
  title={Numerical tests of nucleation theories for the Ising models},
  author={Ryu, Seunghwa and Cai, Wei},
  journal={Physical Review E},
  volume={82},
  number={1},
  pages={011603},
  year={2010},
  publisher={APS}
}
"""


from numba import jit, prange
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import networkx as nx
import atomman as am
from torch_geometric.data import Data
import torch
import argparse
import pickle
from multiprocessing import Pool, cpu_count
import math
from networkx.exception import PowerIterationFailedConvergence

# ----------- functions for Depth First Search (DFS) algorithm -------------------------
def load_data_from_file(datafile, subcube_size = 75):
    data = np.fromfile(datafile, dtype='uint8').reshape(subcube_size,subcube_size,subcube_size)
    return data

@jit(nopython=True, cache=True, parallel=True)
def find_cluster_3d_nopbc(data, x_range=None, cid_start=0):
    if x_range is None:
        x_range = (0, data.shape[0])

    cids = (-1) * np.ones(data.shape, dtype='int32')
    my_cid = cid_start
    clusters = []

    for i in range(x_range[0], x_range[1]): 
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                if data[i, j, k] > 0 and cids[i, j, k] == -1:
                    index_list = [(i, j, k)]
                    index_list, cids = DFS_3d_nopbc(data, x_range, index_list, cids, (i, j, k), my_cid)
                    clusters.append(index_list)
                    my_cid += 1

    return clusters, cids


@jit(nopython=True, cache=True)
def DFS_3d_nopbc(data, x_range, index_list, cids, ind, my_cid):
    y_range = (0, data.shape[1])
    z_range = (0, data.shape[2])
                
    stack = [ind]
    while stack:
        i, j, k = stack.pop()
        if data[i, j, k] > 0 and cids[i, j, k] == -1:
            cids[i, j, k] = my_cid
            index_list.append((i, j, k))
            if i+1 <  x_range[1]: stack.append((i+1, j, k))
            if i-1 >= x_range[0]: stack.append((i-1, j, k))
            if j+1 <  y_range[1]: stack.append((i, j+1, k))
            if j-1 >= y_range[0]: stack.append((i, j-1, k))
            if k+1 <  z_range[1]: stack.append((i, j, k+1))
            if k-1 >= z_range[0]: stack.append((i, j, k-1))
    
    return index_list, cids

def find_clusters_list(data, x_range_list, threshold):
    clusters_list = []
    cids_list = []
    cid_start = 0
    for x_range in x_range_list:
        clusters, cids = find_cluster_3d_nopbc(data, x_range = x_range, cid_start = cid_start)
        #clusters, cids = find_cluster_3d_nopbc(data, threshold, x_range = x_range, cid_start = cid_start)
        cid_start += len(clusters)
        clusters_list.append(clusters)
        cids_list.append(cids)
        #print(f'x_range = {x_range}, # of clusters = {len(clusters)}')
    return clusters_list, cids_list

# produce the same result as find_overlaps_from_cids but slower
def find_overlaps_from_clusters(clusters_list, cids_list):
    overlaps = []
    for layer in range(len(clusters_list)-1):
        for ia in range(len(clusters_list[layer])):
            for ib in range(len(clusters_list[layer+1])):
                if not set(clusters_list[layer][ia]).isdisjoint(set(clusters_list[layer+1][ib])):
                    ida = cids_list[layer][clusters_list[layer][ia][0]]
                    idb = cids_list[layer+1][clusters_list[layer+1][ib][0]]
                    overlaps.append((ida, idb))
    return overlaps

def find_overlaps_from_cids(x_range_list, clusters_list, cids_list, overlap_thickness = 2):
    overlaps = []
    for layer in range(len(clusters_list)-1):
        i = x_range_list[layer][1] - overlap_thickness
        cida = cids_list[layer][i:i+overlap_thickness,:,:].flatten()
        cidb = cids_list[layer+1][i:i+overlap_thickness,:,:].flatten()
        overlap_set = set([ (cida[i], cidb[i]) for i in range(len(cida))])
        if (-1, -1) in overlap_set:
            overlap_set.remove((-1, -1))        
        overlaps.extend(list(overlap_set))
    return overlaps

