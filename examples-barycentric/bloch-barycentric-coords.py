#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#/***************************************************************************
# *   Copyright (C) 2022 -- 2024 by Marek Sawerwain                         *
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
#import matplotlib.pyplot as plt
import numpy as np

#from sklearn import decomposition
from sklearn import datasets



#
# cleanup in progress ...
#

b = qdcl.BlochVisualization()
#b.set_view(-20, 15)
b.set_title("Bloch Vector Points")

#b.set_pure_states( ps_d, "green") 
#b.set_pure_states_as_vectors( centers, "red" )

ptns = np.empty((0,3))

ptns = np.append( ptns, 
    [ qdcl.convert_spherical_coordinates_to_bloch_vector(
       1.0, 
       np.radians(45),          # (0 <= _theta <= np.pi
       np.radians(90 + 55))  # (0 <= _phi <= 2.0 * np.pi)
    ], axis=0 )

ptns = np.append( ptns, 
    [ qdcl.convert_spherical_coordinates_to_bloch_vector(
       1.0, 
       np.radians(50),          # (0 <= _theta <= np.pi
       np.radians(90 + 17))  # (0 <= _phi <= 2.0 * np.pi)
    ], axis=0 )

ptns = np.append( ptns, 
    [ qdcl.convert_spherical_coordinates_to_bloch_vector(
       1.0, 
       np.radians(60),          # (0 <= _theta <= np.pi
       np.radians(90 + 40))  # (0 <= _phi <= 2.0 * np.pi)
    ], axis=0 )

ptns = np.append( ptns, 
    [ qdcl.convert_spherical_coordinates_to_bloch_vector(
       1.0, 
       np.radians(30),          # (0 <= _theta <= np.pi
       np.radians(90 + 30))  # (0 <= _phi <= 2.0 * np.pi)
    ], axis=0 )

# state inside polygon
ptns = np.append( ptns, 
    [ qdcl.convert_spherical_coordinates_to_bloch_vector(
       1.0, 
       np.radians(47),          # (0 <= _theta <= np.pi
       np.radians(90 + 35))  # (0 <= _phi <= 2.0 * np.pi)
    ], axis=0 )

labels=["q0", "q1", "q2", "q3"]
b.set_points( ptns )

f=b.make_figure()
f.show()


b = qdcl.BlochVisualization()
#b.set_view(-20, 15)

b.set_title("")

    
b.clear_points()
b.clear_vectors()
#b.enable_single_batch_draw()
b.enable_single_batch_draw()

q0=qdcl.convert_bloch_vector_to_pure_state( ptns[0][0], ptns[0][1], ptns[0][2] )
q1=qdcl.convert_bloch_vector_to_pure_state( ptns[1][0], ptns[1][1], ptns[1][2] )
q2=qdcl.convert_bloch_vector_to_pure_state( ptns[2][0], ptns[2][1], ptns[2][2] )
q3=qdcl.convert_bloch_vector_to_pure_state( ptns[3][0], ptns[3][1], ptns[3][2] )

qv=qdcl.convert_bloch_vector_to_pure_state( ptns[4][0], ptns[4][1], ptns[4][2] )
 
qvprime1 = (q0+q1+q2+q3)
qvprime1 = qvprime1 / np.linalg.norm(qvprime1)

qvprime2 = (q0+q3)
qvprime2 = qvprime2 / np.linalg.norm(qvprime2)


print(np.vdot(qv, q1))
print(np.vdot(qv, qvprime1))



ptns1 = np.empty((0,2))
ptns1 = np.append(ptns1, [ q0 ], axis=0)
ptns1 = np.append(ptns1, [ q1 ], axis=0)
ptns1 = np.append(ptns1, [ q2 ], axis=0)
ptns1 = np.append(ptns1, [ q3 ], axis=0)

ptns2 = np.empty((0,2))
ptns2 = np.append(ptns2, [ qvprime1 ], axis=0)
#ptns2 = np.append(ptns2, [ qvprime2 ], axis=0)


b.add_pure_states( ptns1, "red", "*" )
b.add_pure_states( ptns2, "green", "^" )

f=b.make_figure()
f.show()
f.savefig("simple-example.png")


#
# simple blobs
#

n_samples = 20
n_clusters = 2
   
centers = [ (1.0,  0.0), (0.0, 1.0) ]

#d = qdcl.create_blob_2d( n_samples, centers )
# d, org_labels = datasets.make_blobs( n_samples=n_samples, 
#                                       centers=centers, 
#                                       cluster_std=0.10, 
#                                       shuffle=False, 
#                                       random_state=1234 )

d = qdcl.create_focused_qubits_probes_with_uniform_placed_centers(n_samples, 2, 2, 0.15)

#dnrm = qdcl.encode_probes_by_normalization(d)
dnrm=d

labels, centroids = qdcl.quantum_kmeans( dnrm, n_samples, n_clusters, 128, qdcl.cosine_distance )

dnrm_set1 = dnrm[0:10]
dnrm_set2 = dnrm[10:20]

#f = qdcl.create_circle_plot_for_2d_data( dnrm )
#f = qdcl.create_circle_plot_for_2d_data( dnrm_set1 )
#f = qdcl.create_circle_plot_for_2d_data( dnrm_set2 )

b = qdcl.BlochVisualization()
#b.set_view(-20, 15)

b.set_title("Quantum States for two blobs")

b.clear_points()
b.clear_vectors()
b.enable_single_batch_draw()
#b.enable_single_batch_draw()

b.set_points( d )
#b.set_points( centroids )
b.enable_single_batch_draw()
    
#b.add_pure_states( dnrm_set1, "red" )
#b.add_pure_states( centroids, "green" )

f=b.make_figure()
f.show()
f.savefig("simple-blobs.png")


####
# f = plt.figure()
# f.set_rasterized(True)
# ax = f.add_subplot(111)
# ax.set_rasterized(True)
# f.savefig('figure_name.eps',rasterized=True,dpi=300)
