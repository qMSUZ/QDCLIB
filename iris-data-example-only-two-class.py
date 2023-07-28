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

def read_iris_data( fname ):
    
    df = pd.read_csv( fname )
    
    #classical normalization - 4 variables
    j=1
    K=np.ndarray(shape=(150,4))
    Kraw=np.ndarray(shape=(150,4))
    while(j<5):
        x_pom=df["X"+str(j)]
        min1=x_pom[0]
        max1=x_pom[0]
        for i in range(80):
            if x_pom[i] < min1:
                min1=x_pom[i]
            if x_pom[i] > max1:
                max1=x_pom[i]
        interval=max1-min1
        #normalized data saved in a numpy array K
        for i in range(150):
            K[i,(j-1)]=(x_pom[i]-min1)/interval
            Kraw[i,(j-1)]=x_pom[i]
        j+=1
    #print(K)
 
    #quantum normalization - final data saved in a numpy array Q
    Q=np.ndarray(shape=(150,4))
    QPrime=np.ndarray(shape=(150,4))
    Q0=np.ndarray(shape=(50,4))
    Q1=np.ndarray(shape=(50,4))
    Q2=np.ndarray(shape=(50,4))
    for i in range(150):
        QPrime[i]=Kraw[i]/np.linalg.norm(Kraw[i])
        sum_all=0
        for j in range(4):
            sum_all+=K[i,j]
        for j in range(4):
            # IRIS data contains only real data
            Q[i,j]=np.sqrt(K[i,j]/sum_all)
    
    
    Y=np.ndarray(shape=(150,1))
    idx0=0
    idx1=0
    idx2=0
    for i in range(150):
        if df['class'][i] == 'Iris-setosa':
            Y[i]=0
            Q0[idx0]=Q[i]
            idx0 = idx0 + 1
        if df['class'][i] == 'Iris-versicolor':
            Y[i]=1
            Q1[idx1]=Q[i]
            idx1 = idx1 + 1
        if df['class'][i] == 'Iris-virginica':
            Y[i]=2
            Q2[idx2]=Q[i]
            idx2 = idx2 + 1
    return df.values, Q, QPrime, Y,  Q0, Q1, Q2


org_iris_data, d,  dprime, Y, d0, d1, d2 = read_iris_data( 'data/iris_data.txt')

fig = plt.figure( figsize = (12,12) )
ax = fig.add_subplot( )
ax.scatter( org_iris_data[0:49, 0],    org_iris_data[0:49, 1],    color="red")
#ax.scatter( org_iris_data[50:99, 0],   org_iris_data[50:99, 1],   color="green")
ax.scatter( org_iris_data[100:149, 0], org_iris_data[100:149, 1], color="blue")
ax.set_title("Oryginal Iris data two first features")
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
plt.show()


fig = plt.figure( figsize = (12,12) )
ax = fig.add_subplot( )
ax.scatter( dprime[0:49, 0],    dprime[0:49, 1],    color="red")
#ax.scatter( dprime[50:99, 0],   dprime[50:99, 1],   color="green")
ax.scatter( dprime[100:149, 0], dprime[100:149, 1], color="blue")
ax.set_title("Iris data after normalizatoion two first features")
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
plt.show()
   
fig = plt.figure( figsize = (12,12) )
ax = fig.add_subplot( )
ax.scatter( d[0:49, 0],    d[0:49, 1],    color="red")
#ax.scatter( d[50:99, 0],   d[50:99, 1],   color="green")
ax.scatter( d[100:149, 0], d[100:149, 1], color="blue")
ax.set_title("Iris data after scale and normalizatoion two first features")
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
plt.show()

fig = plt.figure( figsize = (12,12) )
ax = fig.add_subplot( projection = '3d' )
ax.scatter( d[0:49, 0],    d[0:49, 1],    d[0:49, 2],    color="red")
#ax.scatter( d_r[50:99, 0],   d_r[50:99, 1],   d_r[50:99, 2],   color="green")
ax.scatter( d[100:149, 0], d[100:149, 1], d[100:149, 2], color="blue")
ax.set_title("Iris data after scale and normalizatoion three first features")
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

class1=d[    0:49, 0:3 ]
#class2=d[  50:99, : ]
class3=d[ 100:149, 0:3 ]


# Find centers
dataset = np.vstack( (class1, class3) )
#labels, centers = qdcl.kmeans_quantum_states( dataset, 2, _func_distance=qdcl.FIDELITY_DISTANCE )
#labels, centers = qdcl.kmedoids_quantum_states( dataset, 2, _func_distance=qdcl.MANHATTAN_DISTANCE )
labels, centers = qdcl.kmedoids_quantum_states( dataset, 2, _func_distance=qdcl.MANHATTAN_DISTANCE )

b = qdcl.BlochVisualization()
b.set_view(15, 30)

b.set_title("Bloch Vector Points for first three features")

b.clear_points()
b.add_points( class1, "red", "+")
b.add_points( class3, "blue", ".")

b.set_vectors( centers )


b.enable_multi_batch_draw()

f=b.make_figure()
f.show()



