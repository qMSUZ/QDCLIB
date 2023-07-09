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


import numpy as np
import qdclib as qdcl

# direct points
# as Bloch vectors
def example1():  
    b = qdcl.BlochVisualization()
    b.set_title("Bloch Vector Points")
    
    ptns = np.empty((0,3))

    ptns = np.append(ptns, [[ 1, 0, 0]], axis=0) # positive x
    ptns = np.append(ptns, [[-1, 0, 0]], axis=0) # negative x
    ptns = np.append(ptns, [[ 0, 1, 0]], axis=0) # +y
    ptns = np.append(ptns, [[ 0,-1, 0]], axis=0) # -y
    ptns = np.append(ptns, [[ 0, 0, 1]], axis=0) # +z 
    ptns = np.append(ptns, [[ 0, 0,-1]], axis=0) # -z

    b.set_points( ptns )
    b.enable_draw_points()

    f=b.make_figure()
    f.show()

# direct points
# with spherical coords
def example2():  
    ptns = np.empty((0,3))

    for degree in range(0, 95, 5):
        ptns = np.append(ptns, [ qdcl.convert_spherical_point_to_bloch_vector(1.0, np.radians(135 + degree), np.radians(0)) ], axis=0)
    
    b = qdcl.BlochVisualization()
    b.set_title("Bloch Vector Points")

    b.set_points( ptns )
    b.enable_draw_points()

    f=b.make_figure()
    f.show()


# pure staes
# io Bloch sphere
def example3():  

    b = qdcl.BlochVisualization()
    b.set_title("Pure states")

    purestates = np.empty((0,2))

    purestates = np.append(purestates, [[ 1, 0 ]], axis=0)
    purestates = np.append(purestates, [[ -1.0/np.sqrt(2), 1.0/np.sqrt(2) ]], axis=0)
    purestates = np.append(purestates, [[ 0, 1 ]], axis=0)

    
    b.set_pure_states( purestates )
    b.enable_pure_states_draw()

    f=b.make_figure()
    f.show()

def example4a():
    n_clusters = 3
    probes = qdcl.create_focused_circle_probes_with_uniform_placed_centers(
                    30, 
                    n_clusters, 
                    _width_of_cluster=0.05 )

    purestates = np.empty((0,2))
    for d in probes:
        purestates =  np.append(purestates, [d], axis=0) 


    b = qdcl.BlochVisualization()
    b.set_title("Pure states")

    b.set_pure_states( purestates )
    b.enable_pure_states_draw()

    f=b.make_figure()
    f.show()

def example4b():
    probes = qdcl.create_focused_circle_probes(
                    30, 
                    2, 
                    _width_of_cluster=0.25 )

    purestates = np.empty((0,2))
    for d in probes:
        purestates =  np.append(purestates, [d], axis=0) 


    b = qdcl.BlochVisualization()
    b.set_title("Pure states")

    b.set_pure_states( purestates )
    b.enable_pure_states_draw()

    f=b.make_figure()
    f.show()


def example5():

    d = qdcl.create_focused_qubits_probes( 30, 2, 0.25)

    b = qdcl.BlochVisualization()
    b.set_title("Bloch Vector Points")
  
    b.set_points( d )
    b.enable_draw_points()
      
    f=b.make_figure()
    f.show()


def example6():
    
    _n_theta=2
    _n_psi=2
    
    centers_on_sphere = np.empty((0,3))
       
    for i in range( _n_theta ):
        for j in range( _n_psi+1 ):
            _theta = 2.0 * np.pi * (i/_n_theta)
            _psi = (np.pi/2.0) - np.pi*(j/_n_psi)
            sp = qdcl.convert_spherical_point_to_bloch_vector(1.0, _theta, _psi)
            centers_on_sphere =  np.append(centers_on_sphere, [ sp ] , axis=0) 

    b = qdcl.BlochVisualization()
    b.set_title("Bloch Vector Points")
  
    b.set_points( centers_on_sphere )
    b.enable_draw_points()
      
    f=b.make_figure()
    f.show()
 

#example1()
#example2()
example3()
#example4a()
#example4b()
#example5()
#example6()
