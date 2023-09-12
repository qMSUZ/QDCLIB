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


def classical_spectral_clustering_example():
    _n_samples = 10
    _n_clusters = 2
    _threshold = 2.0
    
    centers = [
                (-5,  5), (5, -5),
              ]
    d, labels = make_blobs( n_samples=_n_samples, centers=centers, cluster_std=0.5, shuffle=False, random_state=1234 )
    
    f = qdcl.create_scatter_plot_for_2d_data( d, _limits=[-7.0, 7.0, -7.0, 7.0] )
    
    labels, A = qdcl.classic_spectral_clustering( d, _n_samples, _n_clusters, _threshold, _func_distance = qdcl.euclidean_distance_with_sqrt )

    adj_matrix = qdcl.create_adjacency_matrix( d, _threshold, _func_distance = qdcl.euclidean_distance_with_sqrt )
    lap_matrix = qdcl.create_laplacian_matrix( adj_matrix )

    fig, ax = plt.subplots()
    im = ax.imshow( adj_matrix )

    fig, ax = plt.subplots()
    im = ax.imshow( lap_matrix )


# simulation of quantum spectral clustering
def quantum_spectral_clustering_example():
    _n_samples = 10
    _n_clusters = 2
    _threshold = 2.0
    
    centers = [
                (-5,  5), (5, -5),
              ]
    d, labels = make_blobs( n_samples=_n_samples, centers=centers, cluster_std=0.5, shuffle=False, random_state=1234 )
    
    f = qdcl.create_scatter_plot_for_2d_data( d, _limits=[-7.0, 7.0, -7.0, 7.0] )
    _func_distance = qdcl.euclidean_distance_with_sqrt

    _qdX = d
    adj_matrix = qdcl.create_adjacency_matrix( _qdX, _threshold, _func_distance )
   
    lap_matrix = qdcl.create_laplacian_matrix( adj_matrix )
   
    evalues, evectors = np.linalg.eig( lap_matrix )
    
    A = np.zeros( shape=(_n_samples, _n_clusters) )
   
    indicies = np.argpartition(evalues,_n_clusters)[:_n_clusters]
    idx=0
    for i in indicies:
        A[:, idx] = evectors[:, i]
        idx = idx + 1

    rho = (1.0/_n_clusters) * (A @ A.T)

    prj_evals, prj_evectors = np.linalg.eig( rho )

    # max eigenvalues
    ev1=prj_evals[1]
    ev2=prj_evals[6]
    
    evec1=prj_evectors[:, 1].reshape(_n_samples, 1)
    evec2=prj_evectors[:, 6].reshape(_n_samples, 1)
    
    prj_evec1 = evec1 @ evec1.T
    prj_evec2 = evec2 @ evec2.T

    rho1=(prj_evec1 @ rho @ prj_evec1)/np.trace(rho @ prj_evec2)
    rho2=(prj_evec2 @ rho @ prj_evec2)/np.trace(rho @ prj_evec2)

    rho1 @ d
    rho2 @ d
    

    fig, ax = plt.subplots()
    im = ax.imshow( rho )


# blobs_example()
classical_spectral_clustering_example()
# quantum_kmeans_example()
