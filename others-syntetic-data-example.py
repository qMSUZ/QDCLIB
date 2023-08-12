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
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons

def circles_example():
    
    verbose = 0
    
    d, org_labels = make_circles(n_samples=100, factor=0.3, noise=0.05, random_state=0)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
        
    f = qdcl.create_circle_plot_for_2d_data(d)    
       
    #class0 = d[ org_labels==0 ]
    class0 = qdcl.get_data_for_class(d, org_labels, 0)
    #class1 = d[ org_labels==1 ]
    class1 = qdcl.get_data_for_class(d, org_labels, 1)
    
    # convert data to Bloch's vectors
    # and then to quantum states
    
    ps_d = np.empty((0,2), dtype=complex)
    for probe in d:
        v0 = probe[0]
        v1 = probe[1] 
        v2 = probe[0]**2 + probe[1]**2
        
        q=qdcl.convert_bloch_vector_to_pure_state(v0, v1, v2)
        ps_d = np.append(ps_d, [[ q[0], q[1] ]], axis=0)
    
    class0_d = np.empty((0,2), dtype=complex)
    for probe in class0:
        v0 = probe[0]
        v1 = probe[1] 
        v2 = probe[0]**2 + probe[1]**2
        
        q=qdcl.convert_bloch_vector_to_pure_state(v0, v1, v2)
        class0_d = np.append(class0_d, [[ q[0], q[1] ]], axis=0)
    
    class1_d = np.empty((0,2), dtype=complex)
    for probe in class1:
        v0 = probe[0]
        v1 = probe[1] 
        v2 = probe[0]**2 + probe[1]**2
        
        q=qdcl.convert_bloch_vector_to_pure_state(v0, v1, v2)
        class1_d = np.append(class1_d, [[ q[0], q[1] ]], axis=0)
      
    
    #distance_constant=qdcl.FIDELITY_DISTANCE
    #func_distance = qdcl.fidelity_as_distance
    
    #distance_constant=qdcl.COSINE_DISTANCE
    #func_distance = qdcl.cosine_distance

    distance_constant=qdcl.P_CQA_DISTANCE
    func_distance = qdcl.probability_as_distance_case_qubit_alpha

    #distance_constant=qdcl.P_CQB_DISTANCE
    #func_distance = qdcl.probability_as_distance_case_qubit_beta

    #distance_constant=qdcl.SWAP_TEST_DISTANCE
    #func_distance = qdcl.swap_test_as_distance_p1

    n_clusters = 2
    labels, centers = qdcl.kmedoids_quantum_states( ps_d, n_clusters, distance_constant)
    
    b = qdcl.BlochVisualization()
    b.set_view(-20, 15)
    b.set_title("Bloch Vector Points")
    
    b.clear_points()
    b.clear_vectors()
    b.enable_single_batch_draw()
    
    b.set_pure_states( ps_d, "green") 
    b.set_pure_states_as_vectors( centers, "red" )
    
    f=b.make_figure()
    f.show()

    if verbose > 0:
        t = qdcl.create_distance_table( ps_d, centers, org_labels, n_clusters, func_distance )
        print("Distance between probes and centers for each class")
        print("data are sorted by class")
        print(t)


        idx=0
        distance_table=np.zeros( shape=(ps_d.shape[0], 4) )
        for e in ps_d:
            distance_table[idx, 0] = func_distance(e, centers[0])
            distance_table[idx, 1] = func_distance(e, centers[1])
            distance_table[idx, 2] = labels[idx]
            distance_table[idx, 3] = org_labels[idx]
            idx=idx+1
        print("Distance between probes and centers for each class and kmedoids label, oryginal label")
        print("smaller value at two first columns points out the class")
        print(distance_table)

def moon_example():
    d, org_labels = make_moons( n_samples = 100,  noise=0.05, )
    #X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

    f = qdcl.create_plot_for_2d_data(d)
   
    class0 = d[ org_labels==0 ]
    class1 = d[ org_labels==1 ]    
   
    ps_d = np.empty((0,2), dtype=complex)
    for probe in d:
        v0 = probe[0]
        v1 = probe[1] 
        v2 = probe[0]**2 + probe[1]**2
        
        q=qdcl.convert_bloch_vector_to_pure_state(v0, v1, v2)
        ps_d = np.append(ps_d, [[ q[0], q[1] ]], axis=0)
        
   
    # n_clusters = 2
    # labels, centers = qdcl.kmeans_spherical( d, n_clusters, 128, qdcl.cosine_distance)
    # f=qdcl.create_circle_plot_with_centers_for_2d_data( d, n_clusters, centers, labels )
    # f.show()

    b = qdcl.BlochVisualization()
    #b.set_view(-20, 15)
    b.set_title("Bloch Vector Points")
    
    b.clear_points()
    b.clear_vectors()
    b.enable_single_batch_draw()
    
    b.set_pure_states( ps_d, "green") 
    # b.set_pure_states_as_vectors( centers, "red" )
    
    f=b.make_figure()
    f.show()
    
circles_example()
