# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 02:05:17 2022

@author: sanaalamgeer
"""
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import networkx as nx
from stellargraph import StellarGraph
import itertools
#%%
def skeleton_image_to_graph_nx(skeIm, connectivity=2):
	#skeIm = img_data
    assert(len(skeIm.shape) == 2)
    skeImPos = np.stack(np.where(skeIm))
    skeImPosIm = np.zeros_like(skeIm, dtype=np.int)
    skeImPosIm[skeImPos[0], skeImPos[1]] = np.arange(0, skeImPos.shape[1])
    g = nx.DiGraph()
    if connectivity == 1:
        neigh = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    elif connectivity == 2:
        #neigh = np.array([[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]])
        neigh = np.array([[0, 1], [0, -1], [1, 0], [-1, 0], [1, -1], [-1, 1]])
    else:
        raise ValueError(f'unsupported connectivity {connectivity}')
    for idx in range(skeImPos[0].shape[0]):
        for neighIdx in range(neigh.shape[0]):
            curNeighPos = skeImPos[:, idx] + neigh[neighIdx]
            if np.any(curNeighPos<0) or np.any(curNeighPos>=skeIm.shape):
                continue
            if skeIm[curNeighPos[0], curNeighPos[1]] > 0:
                g.add_edge(skeImPosIm[skeImPos[0, idx], skeImPos[1, idx]], skeImPosIm[curNeighPos[0], curNeighPos[1]], weight=np.linalg.norm(neigh[neighIdx]))
    g.graph['physicalPos'] = skeImPos.T    
    #g = compute_node_features(g)
    bb = nx.betweenness_centrality(g)
    nx.set_node_attributes(g, bb, "feature")    
    return g

def skeleton_image_to_graph_nx_3D(skeIm, connectivity=3):
	#skeIm = skeleton
    assert(len(skeIm.shape) == 3)
    skeImPos = np.stack(np.where(skeIm))
    skeImPosIm = np.zeros_like(skeIm, dtype=np.int)
    skeImPosIm[skeImPos[0], skeImPos[1], skeImPos[2]] = np.arange(0, skeImPos.shape[1])
    g = nx.Graph()
    if connectivity == 1:
        neigh = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    elif connectivity == 2:
        neigh = np.array([[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]])
    elif connectivity == 3:
        neigh = generate_negihbors([-1, 0, 1])
    else:
        raise ValueError(f'unsupported connectivity {connectivity}')
    for idx in range(skeImPos[0].shape[0]):
        for neighIdx in range(neigh.shape[0]):
            curNeighPos = skeImPos[:, idx] + neigh[neighIdx]
            if np.any(curNeighPos<0) or np.any(curNeighPos>=skeIm.shape):
                continue
            if skeIm[curNeighPos[0], curNeighPos[1], curNeighPos[2]] > 0:
                g.add_edge(skeImPosIm[skeImPos[0, idx], skeImPos[1, idx], skeImPos[2, idx]], skeImPosIm[curNeighPos[0], curNeighPos[1], curNeighPos[2]], weight=np.linalg.norm(neigh[neighIdx]))
    g.graph['physicalPos'] = skeImPos.T	
    #g = compute_node_features(g)
    bb = nx.betweenness_centrality(g)
    nx.set_node_attributes(g, bb, "feature")
    return g

def compute_node_features(g):
	bb = nx.betweenness_centrality(g)
	pr = nx.pagerank(g, alpha=0.9)
	eg = nx.eigenvector_centrality_numpy(g)
	degr = nx.degree_centrality(g)
	dw = nx.closeness_centrality(g)	
	comb = mergeDictionary(bb, pr, eg, degr, dw)
	nx.set_node_attributes(g, comb, "feature")
	
	return g

def mergeDictionary(dict_1, dict_2, dict_3, dict_4, dict_5):
   dict_6 = {**dict_1, **dict_2}
   for key, value in dict_6.items():
       if key in dict_1 and key in dict_2:
               dict_6[key] = [value , dict_1[key]]
			   
   dict_7 = {**dict_6, **dict_3}
   for key, value in dict_6.items():
       if key in dict_6 and key in dict_3:
               dict_7[key] = [value[0], value[1] , dict_3[key]]
   
   dict_8 = {**dict_7, **dict_4}
   for key, value in dict_7.items():
       if key in dict_7 and key in dict_4:
               dict_8[key] = [value[0], value[1], value[2], dict_4[key]]
			   
   dict_9 = {**dict_8, **dict_5}
   for key, value in dict_8.items():
       if key in dict_8 and key in dict_5:
               dict_9[key] = [value[0], value[1], value[2], value[3], dict_5[key]]

   return dict_9

def generate_negihbors(arr):	
	a = list(itertools.combinations_with_replacement(arr, r=3))
	neigh = []
	for i in a:
		neigh.append(np.asarray(i))
	neigh = np.asarray(neigh)
	return neigh


def get_graphs_data(root_dir):
	#root_dir = "D:/Projects/7/Image Graphs/variables/win5lid/"
	#root_dir = root_dir
	files = [f for f in listdir(root_dir) if isfile(join(root_dir, f))]
	files.sort()
	
	graphs = []
	graph_labels = []
	for i in range(len(files))[:]: #10:110 = win5lid
		print(str(i), files[i])
		npz_file = np.load(os.path.join(root_dir, files[i]))
		img_data = npz_file["img_data"]
		#g = skeleton_image_to_graph_nx_3D(img_data, connectivity=3)
		g = skeleton_image_to_graph_nx(img_data, connectivity=2)
		#g_feature_attr = g.copy()
		for node_id, node_data in g.nodes(data=True):
			#print(round(float(str(node_data['feature']).replace('e-', '')), 5))
			node_data["feature"] = [node_data['feature']]
			
		square = StellarGraph.from_networkx(g, node_features="feature")
		
		mos = npz_file["mos"]
			
		graphs.append(square)
		graph_labels.append(mos)
		
	graph_labels = np.asarray(graph_labels)
	graph_labels = pd.Series(graph_labels) 
	return graphs, graph_labels
########################################################################
