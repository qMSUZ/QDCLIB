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
from scipy import stats

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

n_components = 2

pca = decomposition.PCA( n_components )
org_iris_data_pca2 = pca.fit(org_iris_data[:, 0:4]).transform(org_iris_data[:, 0:4])

pca = decomposition.PCA( n_components )
dprime_pca2 = pca.fit(dprime).transform(dprime)
ddata = dprime_pca2



fig = plt.figure( figsize = (12,12) )
ax = fig.add_subplot( )
ax.scatter( ddata[0:49, 0],    ddata[0:49, 1],    color="red")
ax.scatter( ddata[50:99, 0],   ddata[50:99, 1],   color="green")
ax.scatter( ddata[100:149, 0], ddata[100:149, 1], color="blue")
ax.set_title("Oryginal Iris data after PCA 2 components")
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
plt.show()

minx=np.min(ddata[:, 0])
maxx=np.max(ddata[:, 0])

miny=np.min(ddata[:, 1])
maxy=np.max(ddata[:, 1])


# Gaussian KDE for Iris data expresed as qubit states after two components PCA
    
X, Y = np.mgrid[minx:maxx:150j, miny:maxy:150j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack( [ddata[:, 0], ddata[:, 1]] )
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)

fig = plt.figure( figsize = (12,12) )
ax = fig.add_subplot( )
ax.set_xlim([minx, maxx])
ax.set_ylim([miny, maxy])
ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[minx, maxx, miny, maxy])
ax.scatter( ddata[0:49, 0],    ddata[0:49, 1],    color="red")
ax.scatter( ddata[50:99, 0],   ddata[50:99, 1],   color="green")
ax.scatter( ddata[100:149, 0], ddata[100:149, 1], color="blue")
ax.contour(X, Y, Z)
ax.set_title("Gaussian KDE for Iris data expresed as qubit states after two components PCA")
# force for aspect ratio 1:1
(e0,e1,e2,e3) =  ax.get_images()[0].get_extent()
ax.set_aspect( np.abs( (e1-e0)/(e3-e2) ) )
plt.show()

# Schrodinger potential for Iris data expresed as qubit states after two components PCA

cpe = qdcl.ClusteringByPotentialEnergy()
cpe.set_data( ddata )
cpe.set_dimension( 2 )

sigma=0.25

Z = cpe.calc_v_function_on_2d_mesh_grid(0.025, 150, 150 )

fig = plt.figure( figsize = (12,12) )
ax = fig.add_subplot( )
ax.set_xlim([minx, maxx])
ax.set_ylim([miny, maxy])
ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[minx, maxx, miny, maxy])
ax.scatter( ddata[0:49, 0],    ddata[0:49, 1],    color="red")
ax.scatter( ddata[50:99, 0],   ddata[50:99, 1],   color="green")
ax.scatter( ddata[100:149, 0], ddata[100:149, 1], color="blue")
ax.contour(X, Y, Z)
(e0,e1,e2,e3) =  ax.get_images()[0].get_extent()
ax.set_aspect( np.abs( (e1-e0)/(e3-e2) ) )
plt.show()
