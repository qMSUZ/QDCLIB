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

import matplotlib.pyplot as plt
import numpy as np

import qdclib as qdcl

def example1():

    _n_samples = 10

    print("\n\nExample No. 1\n\n")

    print(f"Create {_n_samples} normalized points on a unit circle.")
    print("Data can be regared as qubits if both probability")
    print("amplitudes are real.")
    print("To create clusters, we use modified spherical variant of");
    print("the kmeans algorithm.")
    
    
    np.random.seed( 1234 )

    d = qdcl.create_spherical_probes(_n_samples, 2)
    
    print("Norms of each point in d:")
    print(" " * 4,np.linalg.norm(d, axis=1))
    
    #
    # Scatter plot of 2D data 
    #
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    circle = plt.Circle( (0,0), 1,  color='r', fill=False)
    ax.scatter( d[:,0], d[:,1])
    ax.add_patch(circle)
    fig.show()
    
    n_clusters = 4
    labels, centers = qdcl.kmeans_quantum_states( d, n_clusters, _func_distance=qdcl.COSINE_DISTANCE )
    # labels, centers = qdcl.kmeans_quantum_states( d, n_clusters, _func_distance=qdcl.MANHATTAN_DISTANCE )
    # labels, centers = qdcl.kmeans_quantum_states( d, n_clusters, _func_distance=qdcl.DOT_DISTANCE )
    # labels, centers = qdcl.kmeans_quantum_states( d, n_clusters, _func_distance=qdcl.FIDELITY_DISTANCE )
    # labels, centers = qdcl.kmeans_quantum_states( d, n_clusters, _func_distance=qdcl.TRACE_DISTANCE )
    
    print("Norms of each point in centers:")
    print(" " * 4,np.linalg.norm(centers, axis=1))
    
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    circle = plt.Circle( (0,0), 1,  color='r', fill=False)
    ax.scatter( d[:,0], d[:,1], c=labels)
    ax.scatter(centers[:, 0], centers[:, 1], marker='x', color='g')
    for idx in range(n_clusters):
        ax.annotate("", xy=(centers[idx, 0], centers[idx, 1]), xytext=(0, 0), arrowprops=dict(arrowstyle="->"))
    ax.add_patch(circle)
    fig.show()
    
    t = qdcl.create_distance_table( d, centers, labels, n_clusters, qdcl.cosine_distance )
    print("")
    print(t)

def example2():

    _n_samples = 30
    _n_clusters = 3
    
    print("\n\nExample 2\n\n")
    
    print(f"Similiar to the Example 1, but we create {_n_samples}")
    print(f"with {_n_clusters} clusters. The applied API is also easier to use.")
    print("To create clusters, we use modified spherical variant of");
    print("the kmeans algorithm.")

    np.random.seed( 1234 )
    
    d = qdcl.create_focused_circle_probes(_n_samples, _n_clusters, _width_of_cluster=0.5)
    labels, centers = qdcl.kmeans_quantum_states( d, _n_clusters, _func_distance=qdcl.COSINE_DISTANCE )
    
    f=qdcl.create_circle_plot_with_centers_for_2d_data( d, _n_clusters, centers, labels )
    f.show()

def example3():

    _n_samples = 40
    _n_clusters = 5    

    print("\n\nExample 3\n\n")

    print(f"Similiar to the Examples 1 and 2, but we create {_n_samples}")
    print(f"with {_n_clusters} clusters. The applied API is also easier to use.")
    print("To create clusters, we perform the kmedoids algorithm.")

    np.random.seed( 1234 )

    
    d = qdcl.create_focused_circle_probes(_n_samples, _n_clusters, _width_of_cluster=0.175)
    labels, centers = qdcl.kmedoids( d, _n_clusters, _max_iterations=128, _func_distance=qdcl.cosine_distance )
    
    f=qdcl.create_circle_plot_with_centers_for_2d_data( d, _n_clusters, centers, labels )
    f.show()

    dt = qdcl.create_distance_table(d, centers, labels, _n_clusters, _func_distance=qdcl.fidelity_as_distance)
    for i in range( _n_clusters ):
        print( "distance for cluster", i )
        print( qdcl.get_distances_for_cluster( dt, i ) )



example1()
example2()
example3()

