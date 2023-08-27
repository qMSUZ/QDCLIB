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

def circles_example( _verbose = 0 ):
    
    
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

    if _verbose > 0:
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

def moon_example( _verbose = 0 ):
    d, org_labels = make_moons( _n_samples = 100,  noise=0.05, )
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


def example_simple_2d_blob( _verbose = 0 ):
    
    blob_data = qdcl.create_blob_2d( _n_samples = 500 )
    
    limits_blob_data = [ np.min(blob_data[:,0]), np.max(blob_data[:,0]), np.min(blob_data[:,1]), np.max(blob_data[:,1]) ]   
    limits_blob_data = output_list = [v * 1.25 for v in limits_blob_data]
    
    f = qdcl.create_scatter_plot_for_2d_data( blob_data, _limits=limits_blob_data )
    
   
def example_linearly_separable_data_2d(  _verbose = 0 ):
    centers=[[0,3],[3,0]]
    line1x, line2x, label1, label2 = qdcl.create_data_separated_by_line ( _centers=centers )
    
    train_d, train_labels, test_d, test_labels = qdcl.split_data_and_labels(line1x, label1, line2x, label2, 0.30)
    
    n_samples = train_d.shape[0]
    n_samples_test = test_d.shape[0]
    
    line_data=qdcl.data_vertical_stack( line1x, line2x )
    limits_line_data = [ np.min(line_data[:,0]), np.max(line_data[:,0]), np.min(line_data[:,1]), np.max(line_data[:,1]) ]
   
    f = qdcl.create_scatter_plot_for_2d_data( line_data, _limits=limits_line_data )
    f = qdcl.create_scatter_plot_for_2d_data( train_d, _limits=limits_line_data )
    f = qdcl.create_scatter_plot_for_2d_data( test_d, _limits=limits_line_data )
    
    # classic SVM with QuantumSVM class
    
    objsvm=qdcl.QuantumSVM()
    objsvm.set_data(train_d, train_labels)
    objsvm.classic_fit()
    labels_predict = objsvm.classic_predict( test_d )
    
    # and quantum version of SVM with QuantumSVM class
    
    q_train_d = np.empty((0,2), dtype=complex)
    for d in train_d:
        q=qdcl.encode_probe(d)
        q_train_d = np.append(q_train_d, [[ q[0], q[1] ]], axis=0)

    q_test_d = np.empty((0,2), dtype=complex)
    for d in test_d:
        q=qdcl.encode_probe(d)
        q_test_d = np.append(q_test_d, [[ q[0], q[1] ]], axis=0)



    objqsvm=qdcl.QuantumSVM()    
    objqsvm.set_data(train_d, train_labels)
    
    K = qdcl.create_kernel_matrix_for_training_data( q_train_d, 0.1, n_samples )
    # Kinv=np.linalg.inv(K)
    # Id= qdcl.chop_and_round_for_array( Kinv @ K )
   
    b_alpha_vector = qdcl.create_right_b_alpha_vector( K, train_labels, n_samples )
    
    b, C, alphas = qdcl.create_b_c_and_alphas( b_alpha_vector, n_samples)
    
    Nu = qdcl.create_nu_coefficent(train_d, b, alphas, n_samples )
    Nx = qdcl.create_nx_coefficent(train_d[0], n_samples ) 
    
    print("Training data")
    for idx in range(n_samples):
        P = 0.5 * ( 1.0 - qdcl.create_dot_ux_for_classification( Nu, Nx, b, alphas, q_train_d, q_train_d[idx], n_samples))
        print("P=",P," Label = ", 1 if P <= 0.5 else -1, "org label", train_labels[idx])
        #print("P < 1/2 we classify probe as +1, otherwise −1")

    print("Test data")
    for idx in range(n_samples_test):
        P = 0.5 * ( 1.0 - qdcl.create_dot_ux_for_classification( Nu, Nx, b, alphas, q_train_d, q_test_d[idx], n_samples))
        print("P=",P," Label = ", 1 if P <= 0.5 else -1, "org label", test_labels[idx])
        #print("P < 1/2 we classify probe as +1, otherwise −1")
    

def example_non_linearly_separable_data_2d(  _verbose = 0 ):    
   
    line_data = qdcl.create_data_non_line_separated( 100 )
    limits_line_data = [ np.min(line_data[:,0]), np.max(line_data[:,0]), np.min(line_data[:,1]), np.max(line_data[:,1]) ]
    limits_line_data = output_list = [v * 1.25 for v in limits_line_data]
    
    f = qdcl.create_scatter_plot_for_2d_data( line_data, _limits=limits_line_data )
            
# circles_example( 1 )
# moon_example()
# example_simple_2d_blob()
example_linearly_separable_data_2d()
# example_non_linearly_separable_data_2d()
