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