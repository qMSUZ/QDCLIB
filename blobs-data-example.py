#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#/***************************************************************************
# *   Copyright (C) 2022 -- 2023 by Marek Sawerwain                         *
# *                                  <M.Sawerwain@gmail.com>                *
# *                                  <M.Sawerwain@issi.uz.zgora.pl>         *
# *                                                                         *
# *                              by Joanna Wiśniewska                       *
# *                                  <Joanna.Wisniewska@wat.edu.pl>         *
# *                                                                         *
# *   Part of the Quantum Distance Classifier:                              *
# *         https://github.com/qMSUZ/QDCLIB                                 *
# *                                                                         *
# *   Licensed under the EUPL-1.2-or-later, see LICENSE file.               *
# *                                                                         *
# *   Licensed under the EUPL, Version 1.2 or - as soon they will be        *
# *   approved by the European Commission - subsequent versions of the      *
# *   EUPL (the "Licence");                                                 *
# *                                                                         *
# *   You may not use this work except in compliance with the Licence.      *
# *   You may obtain a copy of the Licence at:                              *
# *                                                                         *
# *   https://joinup.ec.europa.eu/software/page/eupl                        *
# *                                                                         *
# *   Unless required by applicable law or agreed to in writing,            *
# *   software distributed under the Licence is distributed on an           *
# *   "AS IS" basis, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,          *
# *   either express or implied. See the Licence for the specific           *
# *   language governing permissions and limitations under the Licence.     *
# *                                                                         *
# ***************************************************************************/

import qdclib as qdcl
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from sklearn import decomposition
from sklearn.datasets import make_blobs

def blobs_example():
    centers = [
                (-5,  5), (0,  5), (5,  5),
                (-5,  0), (0,  0), (5,  0),
                (-5, -5), (0, -5), (5, -5),
              ]
    d, labels = make_blobs(n_samples=200, centers=centers, cluster_std=0.5, shuffle=False, random_state=1234)

    #X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
        
    f = qdcl.create_scatter_plot_for_2d_data(d, _limits=[-7.0, 7.0, -7.0, 7.0])    
    
    # n_clusters = 2
    # labels, centers = qdcl.kmeans_spherical( d, n_clusters, 128, qdcl.cosine_distance)
    # f=qdcl.create_circle_plot_with_centers_for_2d_data( d, n_clusters, centers, labels )
    # f.show()
    
def quantum_kmeans_example():

    _n_samples = 20
    _n_clusters = 2
    
    centers = [
                (-5,  5), (5, -5),
              ]
    
    d, org_labels = make_blobs( n_samples=_n_samples, centers=centers, cluster_std=0.5, shuffle=False, random_state=1234 )
    _qdX=d
    
    dnrm = qdcl.encode_probes_by_normalisation(d)
    
    # f = qdcl.create_scatter_plot_for_2d_data(dnrm, _limits=[-7.0, 7.0, -7.0, 7.0])
    f = qdcl.create_circle_plot_for_2d_data( dnrm )
    
    # lab_kmeans, cent_kmeans= qdcl.kmeans_quantum_states( d, _n_clusters, _func_distance=qdcl.MANHATTAN_DISTANCE )
    # lab_kmedoids, cent_kmedoids = qdcl.kmedoids_quantum_states( d, _n_clusters, _func_distance=qdcl.MANHATTAN_DISTANCE )
    
    
    # ck = qdcl.create_ck_table_zero_filled( _n_samples ) 
    # ck[:10]=0
    # ck[10:]=1
    
    ck = qdcl.random_assign_clusters_to_ck(_n_samples, _n_clusters)
    
    # centroids = np.zeros( shape=(_n_clusters, _qdX.shape[1]) )
    
    
    centroids = qdcl.create_initial_centroids(dnrm, _n_samples, _n_clusters)
    
    distance_table, ck = qdcl.quantum_kmeans_clusters_assignment(dnrm,
                                             centroids, 
                                             _n_samples, _n_clusters,
                                             _func_distance = qdcl.cosine_distance )
    
    centroids = qdcl.quantum_kmeans_update_centroids( dnrm, ck, _n_samples, _n_clusters )
    
    labels = qdcl.quantum_kmeans_assign_labels( dnrm, centroids, 
                                               _n_samples,  _n_clusters, 
                                               _func_distance = qdcl.cosine_distance)
    
    f = qdcl.create_circle_plot_with_centers_for_2d_data( dnrm, _n_clusters, centroids, labels)
    
    #
    # create centroids for verification
    # order of labels is not preserved
    #
    
    d_for_k0=np.zeros( shape = _qdX.shape  )
    d_for_k1=np.zeros( shape = _qdX.shape  )
    
    for idx in qdcl.get_indices_for_cluster_k(ck, 0):
        d_for_k0[idx] = d[ idx ]
    
    for idx in qdcl.get_indices_for_cluster_k(ck, 1):
        d_for_k1[idx] = d[ idx ]
        
    center_k0 = np.sum(d[qdcl.get_indices_for_cluster_k(ck, 0)], axis=0) / qdcl.number_of_probes_in_cluster(ck, 0)
    center_k1 = np.sum(d[qdcl.get_indices_for_cluster_k(ck, 1)], axis=0) / qdcl.number_of_probes_in_cluster(ck, 1)


def spectral_clustering():
    centers = [
                (-5,  5), (5, -5),
              ]
    d, labels = make_blobs( n_samples=200, centers=centers, cluster_std=0.5, shuffle=False, random_state=1234 )
    
    f = qdcl.create_scatter_plot_for_2d_data( d, _limits=[-7.0, 7.0, -7.0, 7.0] )
    
    adj_matrix = qdcl.create_adjacency_matrix( d, 2.0, qdcl.euclidean_distance_with_sqrt )

    lap_matrix = qdcl.create_laplacian_matrix( adj_matrix )

    fig, ax = plt.subplots()
    im = ax.imshow( adj_matrix )

    fig, ax = plt.subplots()
    im = ax.imshow( lap_matrix )


# blobs_example()
spectral_clustering()
# quantum_kmeans_example()
