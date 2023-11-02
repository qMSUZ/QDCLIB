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

qdcl.datasets.iris.load_data()

org_iris_data = qdcl.datasets.iris.iris_dataset

# , d,  dprime, Y, d0, d1, d2 = read_iris_data( 'datasets/iris_data.txt')

n_components = 3
pca = decomposition.PCA( n_components )
#d_r = pca.fit(d).transform(d)
#d_r = pca.fit(dprime).transform(dprime)
d_r = pca.fit(org_iris_data[:, 0:4]).transform(org_iris_data[:, 0:4])


fig = plt.figure( figsize = (12,12) )
ax = fig.add_subplot( )
ax.scatter( org_iris_data[0:49, 0],    org_iris_data[0:49, 1],    color="red")
#ax.scatter( org_iris_data[50:99, 0],   org_iris_data[50:99, 1],   color="green")
ax.scatter( org_iris_data[100:149, 0], org_iris_data[100:149, 1], color="blue")
ax.set_title("Iris Data two first features")
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
plt.show()

   
fig = plt.figure( figsize = (12,12) )
ax = fig.add_subplot( projection = '3d' )
ax.scatter( d_r[0:49, 0],    d_r[0:49, 1],    d_r[0:49, 2],    color="red")
ax.scatter( d_r[50:99, 0],   d_r[50:99, 1],   d_r[50:99, 2],   color="green")
ax.scatter( d_r[100:149, 0], d_r[100:149, 1], d_r[100:149, 2], color="blue")
ax.set_title("PCA for n_components = {}".format(n_components))
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

class1=d_r[   0:49 , : ]
class2=d_r[  50:99 , : ]
class3=d_r[ 100:149, : ]

# ptns = np.empty((0,3))
# for i in range(0, 150):
# #for i in range(0, 50, 1):
# #for i in range(50, 100, 1):
# #for i in range(100, 150, 1):    
#     ptns = np.append(ptns, [ d_r[i] ], axis=0) 
#     # points will be normalised by set_points
#     # method

#labels, centers = qdcl.kmeans_quantum_states( d_r, 3, _func_distance=qdcl.COSINE_DISTANCE )
labels, centers = qdcl.kmedoids_quantum_states( d_r, 3, _func_distance=qdcl.MANHATTAN_DISTANCE )

b = qdcl.BlochVisualization()
b.set_title("Bloch Vector Points")
#b.set_view(15, 30)

b.clear_points()
b.add_points( class1, "red", "+")
b.add_points( class2, "green", "o")
b.add_points( class3, "blue", ".")

b.enable_multi_batch_draw()

b.set_vectors( centers )

f=b.make_figure()
f.show()
