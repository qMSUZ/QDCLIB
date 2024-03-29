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

def basic_example():
    b = qdcl.BlochVisualization()
    
    b.set_title("Bloch Vector Points")
        
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
    
    b.set_points( ptns )
    
          
    b.clear_points()
    b.clear_vectors()
    
    b.enable_single_batch_draw()
    
    q0=qdcl.convert_bloch_vector_to_pure_state( ptns[0][0], ptns[0][1], ptns[0][2] )
    q1=qdcl.convert_bloch_vector_to_pure_state( ptns[1][0], ptns[1][1], ptns[1][2] )
    q2=qdcl.convert_bloch_vector_to_pure_state( ptns[2][0], ptns[2][1], ptns[2][2] )
    q3=qdcl.convert_bloch_vector_to_pure_state( ptns[3][0], ptns[3][1], ptns[3][2] )
     
    qvprime1 = (q0+q1+q2+q3)
    qvprime1 = qvprime1 / np.linalg.norm(qvprime1)
             
    
    ptns1 = np.empty((0,2))
    ptns1 = np.append(ptns1, [ q0 ], axis=0)
    ptns1 = np.append(ptns1, [ q1 ], axis=0)
    ptns1 = np.append(ptns1, [ q2 ], axis=0)
    ptns1 = np.append(ptns1, [ q3 ], axis=0)
    
    ptns2 = np.empty((0,2))
    ptns2 = np.append(ptns2, [ qvprime1 ], axis=0)   
    
    b.add_pure_states( ptns1, "red", "*" )
    b.add_pure_states( ptns2, "green", "^" )
    
    f=b.make_figure()
    f.show()
    f.savefig("simple-example.png")
    


def simple_blobs_example():

    n_samples = 20
    n_clusters = 2
   
    centers = [ (1.0,  0.0), (0.0, 1.0) ]
    
    d = qdcl.create_focused_qubits_probes_with_uniform_placed_centers(n_samples, 2, 2, 0.15)
    
    labels, centroids = qdcl.quantum_kmeans( d, n_samples, n_clusters, 128, qdcl.cosine_distance )
    
    dnrm_set1 = d[ labels == 0]
    dnrm_set2 = d[ labels == 1]   
    
    b = qdcl.BlochVisualization()
    
    b.set_title("Quantum States for two blobs")
    
    b.clear_points()
    b.clear_vectors()
    b.enable_single_batch_draw()
    
    b.set_points( dnrm_set1, "green", "*" )
    b.add_points( dnrm_set2, "green", "*" )
    b.add_points( centroids, "red", "^" )
    
    f=b.make_figure()
    f.show()
    f.savefig("simple-blobs.png")

def iris_set_example():
    pass



#basic_example()
simple_blobs_example()



# ####
# # f = plt.figure()
# # f.set_rasterized(True)
# # ax = f.add_subplot(111)
# # ax.set_rasterized(True)
# # f.savefig('figure_name.eps',rasterized=True,dpi=300)
