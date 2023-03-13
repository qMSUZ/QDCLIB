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
# *                                                                         *
# *   Part of the VQEClassification:                                              *
# *         https://github.com/qMSUZ/VQEClassification                            *
# *                                                                         *
# *   This program is free software; you can redistribute it and/or modify  *
# *   it under the terms of the GNU General Public License as published by  *
# *   the Free Software Foundation; either version 3 of the License, or     *
# *   (at your option) any later version.                                   *
# *                                                                         *
# *   This program is distributed in the hope that it will be useful,       *
# *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
# *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
# *   GNU General Public License for more details.                          *
# *                                                                         *
# *   You should have received a copy of the GNU General Public License     *
# *   along with this program; if not, write to the                         *
# *   Free Software Foundation, Inc.,                                       *
# *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
# ***************************************************************************/

import numpy as np
import pandas as pd
import time


from sympy import sqrt


from qiskit.utils import algorithm_globals

from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import RealAmplitudes

from qiskit.algorithms.optimizers import COBYLA
#from qiskit.algorithms.optimizers import SPSA

from qiskit.primitives import Sampler

from qiskit_machine_learning.algorithms.classifiers import VQC


from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def read_data():
    df = pd.read_excel (r'data/train-data.xlsx')
    
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
            Q[i,j]=sqrt(K[i,j]/sum_all)
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


def callback_graph(weights, obj_func_eval):
    train_obj_func_vals.append(obj_func_eval)   

print("Read data: credit values")
df, Q, labels_for_Q, Q0, Q1 = read_data()


algorithm_globals.random_seed = 111111

svc = SVC(kernel="poly", C=10, probability=True, random_state=0)
svc = svc.fit(Q, labels_for_Q)  # suppress printing the return value


train_score = svc.score(Q, labels_for_Q) * 100

print(f"Classical SVC on the training dataset: {train_score:.2f}%")


num_max_iter = 100
num_features = Q.shape[1]


feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2)
feature_map.decompose().draw(output="mpl", fold=20)

ansatz = RealAmplitudes(num_qubits=num_features, reps=2)
ansatz.decompose().draw(output="mpl", fold=20)



optimizer = COBYLA(maxiter=num_max_iter)
#optimizer = SPSA(maxiter=num_max_iter)
sampler = Sampler()

train_obj_func_vals = []

vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=callback_graph,
)

objective_func_vals = []

start = time.time()
vqc.fit(Q, labels_for_Q)
elapsed = time.time() - start

print(f"Training time: {round(elapsed)} seconds")

train_score_q = vqc.score(Q, labels_for_Q) * 100
#test_score_q = vqc.score(Q, df.Y) * 100

print(f"Quantum VQC on the training dataset: {train_score_q:.2f}%")
#print(f"Quantum VQC on the test dataset:     {test_score_q:.2f}%")

print("Objective values:", train_obj_func_vals)
