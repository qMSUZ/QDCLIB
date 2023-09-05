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


# data preparation

centers=[[0,3],
         [3,0]]

line1x, line2x, label1, label2 = qdcl.create_data_separated_by_line ( _centers=centers )

train_d, train_labels, test_d, test_labels = qdcl.split_data_and_labels(line1x, label1, line2x, label2, 0.30)

n_samples = train_d.shape[0]
n_samples_test = test_d.shape[0]

# unused variables removing
del( centers )
del( line1x, line2x, label1, label2 )

# classic SVM with QuantumSVM class

def classic_svm():
    objsvm=qdcl.QuantumSVM()
    objsvm.set_data(train_d, train_labels)
    objsvm.classic_fit()
    labels_predict = objsvm.classic_predict( test_d )
    
    print("Labels: labels_predict - test_labels")
    print(labels_predict - test_labels)

# Quantum version of SVM with QuantumSVM class

def quantum_svm_direct_api():
    
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
    
def quantum_svm():
    objsvm=qdcl.QuantumSVM()
    objsvm.set_data(train_d, train_labels, _is_it_quantum=True)
    objsvm.prepare_quantum_objects()
    objsvm.quantum_fit()
    labels_predict = objsvm.quantum_predict( test_d )

    print("Labels: labels_predict - test_labels")
    print(labels_predict - test_labels)

    print("Test data")
    for idx in range(n_samples_test):
        print("idx=",idx," Label = ", labels_predict[idx], "org label", test_labels[idx])
        #print("P < 1/2 we classify probe as +1, otherwise −1")

    
# classic_svm()
# quantum_svm_direct_api()
quantum_svm()
