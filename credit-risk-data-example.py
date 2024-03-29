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

import qcs
import scipy

import pandas as pd
import numpy as np

from sklearn import decomposition

def read_data(_fname):
    df = pd.read_excel ( _fname )
    
    j=1
    K=np.ndarray(shape=(80,8))
    while(j<5):
        x_pom=df["X"+str(j)]
        min1=x_pom[0]
        max1=x_pom[0]
        for i in range(80):
            if x_pom[i] < min1:
                min1=x_pom[i]
            if x_pom[i] > max1:
                max1=x_pom[i]
        zakres=max1-min1
        for i in range(80):
            K[i,(j-1)]=(x_pom[i]-min1)/zakres
        j+=1
    x5_pom=df["X5"]
    x6_pom=df["X6"]
    x7_pom=df["X7"]
    x8_pom=df["X8"]
    for i in range(80):
        K[i,4]=x5_pom[i]
        K[i,5]=x6_pom[i]
        K[i,6]=x7_pom[i]
        K[i,7]=x8_pom[i]
    
    Q=np.ndarray(shape=(80,8))
    labels_for_Q = np.ndarray(shape=(80,))
    Q0=np.ndarray(shape=(1,8))
    Q1=np.ndarray(shape=(1,8))
    for i in range(80):
        sum_all=0
        for j in range(8):
            sum_all+=K[i,j]
        for j in range(8):
            Q[i,j]=np.sqrt(K[i,j]/sum_all)
    for i in range(80):
        suma=0
        for j in range(8):
            suma+=Q[i,j]*Q[i,j]
    for i in range(80):  
        labels_for_Q[i]=df.Y[i]
        if df.Y[i] == 0:
            Q0=np.vstack((Q0, Q[i]));
        if df.Y[i] == 1:
            Q1=np.vstack((Q1, Q[i]));
    
    Q0 = np.delete(Q0, (0), axis=0)
    Q1 = np.delete(Q1, (0), axis=0)
     
    return df, Q, labels_for_Q, Q0, Q1



def prepare_data():
    pass

def perfom_kmedoids( _n_centers):
    pass

def bloch_sphere_representation():
    pass





df, Q, labels_for_Q, Q0, Q1 = read_data( r'data/credit-risk-train-data.xlsx' )

# Pearson Correlation Coefficient  (PCC)
pcc_rho = np.corrcoef( Q[:,0:7].transpose() )

print("Pearson Correlation Coefficients")
print(pcc_rho)


n_features = 3
pca = decomposition.PCA( n_features )

Q0_r = pca.fit(Q0).transform(Q0)
Q1_r = pca.fit(Q1).transform(Q1)


for idx in range(Q0_r.shape[0]):
    p = Q0_r[idx]
    p = p / np.linalg.norm(p)
    Q0_r[idx] = p

for idx in range(Q1_r.shape[0]):
    p = Q1_r[idx]
    p = p / np.linalg.norm(p)
    Q1_r[idx] = p


plt.figure( figsize = (12,12) )
fig, ax = plt.subplots()
ax.scatter(Q0_r[:, 0], 
            Q0_r[:, 1], 
            s=14, marker="o", color="blue")
#ax.scatter(cluster_for_Q0_r.cluster_centers_  [:, 0], 
            #cluster_for_Q0_r.cluster_centers_  [:, 1], 
#            s=50, marker="o", alpha=0.5, color="blue")
# ax.scatter(Q0_r[Q0_cluster2_idx, 0], 
#            Q0_r[Q0_cluster2_idx, 1], 
#            s=14, marker="o", color="blue")
# ax.scatter(cluster_for_Q0_r.cluster_centers_  [:, 0], 
#           cluster_for_Q0_r.cluster_centers_  [:, 1], 
#           s=50, marker="o", alpha=0.5, color="blue")
#for idx in range(0,40):
#    ax.annotate( str(idx), (Q0_r[idx, 0], Q0_r[idx, 1]) )
ax.scatter(Q1_r[:, 0], 
            Q1_r[:, 1], 
            s=10, marker="^", color="red")
#ax.scatter(cluster_for_Q1_r.cluster_centers_  [:, 0], 
#            cluster_for_Q1_r.cluster_centers_  [:, 1], 
#            s=50, marker="^", alpha=0.5, color="red")
#for idx in range(0,40):
#    ax.annotate( str(idx), (Q1_r[idx, 0], Q1_r[idx, 1]) )
plt.title("Principal component analysis for data encoded as quantum states ")
plt.xlabel("Values of first feature")
plt.ylabel("Values of second feature")
# plt.savefig("pca-figure.png")
# plt.savefig("pca-figure.eps")
fig.show()


Q0_r_labels, Q0_r_centers = qdcl.kmedoids_quantum_states( Q0_r, 8, _func_distance=qdcl.COSINE_DISTANCE )
Q1_r_labels, Q1_r_centers = qdcl.kmedoids_quantum_states( Q1_r, 8, _func_distance=qdcl.COSINE_DISTANCE )

b = qdcl.BlochVisualization()
b.set_view(15, 30)
#b.set_view(0, 0)

b.set_title("Bloch Vector Points for first three features")

b.clear_points()
b.add_points( Q0_r, "red", "+")
b.add_points( Q1_r, "blue", ".")

b.add_vectors( Q0_r_centers, "red")
b.add_vectors( Q1_r_centers, "blue" )

b.enable_multi_batch_draw()

f=b.make_figure()
f.show()


# VQE class test

Q0_labels, Q0_centers = qdcl.kmedoids_quantum_states( Q0, 8, _func_distance=qdcl.COSINE_DISTANCE )
Q1_labels, Q1_centers = qdcl.kmedoids_quantum_states( Q1, 8, _func_distance=qdcl.COSINE_DISTANCE )


_circuit_type = 0
_n_layers = 2
_n_qubits = 3
_qubits = [0, 1, 2]
_n_centers = 8


vqeQ0 = qdcl.VQEClassification()
vqeQ0.set_qubits_table( _qubits )
vqeQ0.set_number_of_qubits( _n_qubits )
vqeQ0.create_n_centers( _n_centers )

for idx in range( 0, _n_centers ):
    vqeQ0.set_center( idx, Q0_centers[idx] )


vqeQ1 = qdcl.VQEClassification()
vqeQ1.set_qubits_table( _qubits )
vqeQ1.set_number_of_qubits( _n_qubits )
vqeQ1.create_n_centers( _n_centers )


for idx in range( 0, _n_centers ):
    vqeQ1.set_center( idx, Q1_centers[idx] )

if _circuit_type == 0:
    paramsQ0 = [1] * (9*_n_layers) 

if _circuit_type == 1:
    paramsQ0 = [1] * (6*_n_layers) 

if _circuit_type == 2:
    paramsQ0 = [1] * (6*_n_layers) # for type 2
 
if _circuit_type == 3:
    paramsQ0 = [1] * (9*_n_layers) # for type 3

if _circuit_type == 4:
    paramsQ0 = [1] * ((3 + (6*_n_layers) + 3)) # for type 4

if _circuit_type == 5:
    paramsQ0 = [1] * ((3 + (3*_n_layers) + 3)) # for type 5

paramsQ1 = paramsQ0
    
# 
# for class 0 and  Q0_centers[0]
#
paramsQ0C0 = paramsQ0

_num_qubits=_n_qubits
centers = vqeQ0.get_centers()
_n_center=0

def objective_function( _parameters, *args):
    cost_value = 0.0
    
    _qubits, _n_center, _circuit_type, _n_layers = args 

    #q = vqeQ0.perform_variational_circuit( _qubits, _parameters, _circuit_type, _n_layers )
    #_state = q.ToNumpyArray()
    _state = Q0_centers[0]
    
    cost_value = sum( abs( centers[i, _n_center] - _state[i]) 
                     for i in range(2 ** _num_qubits) )

    return cost_value

result = scipy.optimize.minimize(
            fun=objective_function,
            x0=paramsQ0C0,
            args=(_qubits, _n_center, _circuit_type, _n_layers),
            method='COBYLA' )