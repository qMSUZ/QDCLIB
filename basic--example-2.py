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

def bloch_sphere_direct_definition_pure_states():
    purestates = np.empty((0,2))
      
    purestates = np.append( purestates, [qxplus] , axis=0 )
    purestates = np.append( purestates, [qxminus], axis=0 )
    purestates = np.append( purestates, [qyplus] , axis=0 )
    purestates = np.append( purestates, [qyminus], axis=0 )
    purestates = np.append( purestates, [qzplus] , axis=0 )
    purestates = np.append( purestates, [qzminus], axis=0 )

    b = qdcl.BlochVisualization()
    b.set_title("Pure states")
    
    b.set_pure_states( purestates )
    b.enable_pure_states_draw()
    
    f=b.make_figure()
    f.show()

def bloch_sphere_definition_pure_states_by_spherical_corrds():
    purestates = np.empty((0,2))
      
    purestates = np.append( purestates, [qxplus_f_sp] , axis=0 )
    purestates = np.append( purestates, [qxminus_f_sp], axis=0 )
    purestates = np.append( purestates, [qyplus_f_sp] , axis=0 )
    purestates = np.append( purestates, [qyminus_f_sp], axis=0 )
    purestates = np.append( purestates, [qzplus_f_sp] , axis=0 )
    purestates = np.append( purestates, [qzminus_f_sp], axis=0 )

    b = qdcl.BlochVisualization()
    b.set_title("Pure states")
    
    b.set_pure_states( purestates )
    b.enable_pure_states_draw()
    
    f=b.make_figure()
    f.show()

def bloch_sphere_vectosr_from_pure_states():
    b = qdcl.BlochVisualization()
    b.set_title("Bloch Vector Points")
    
    ptns = np.empty((0,3))

    ptns = np.append(ptns, [ bvxplus ], axis=0) # positive x
    ptns = np.append(ptns, [ ], axis=0) # negative x
    ptns = np.append(ptns, [ ], axis=0) # +y
    ptns = np.append(ptns, [ ], axis=0) # -y
    ptns = np.append(ptns, [ ], axis=0) # +z 
    ptns = np.append(ptns, [ ], axis=0) # -z

    b.set_points( ptns )
    b.enable_draw_points()

    f=b.make_figure()
    f.show() 

qxplus  = np.array( [1.0/np.sqrt(2),  1.0/np.sqrt(2)  ] )
qxminus = np.array( [1.0/np.sqrt(2), -1.0/np.sqrt(2)  ] )

qyplus  = np.array( [1.0/np.sqrt(2),  1.0J/np.sqrt(2) ] )
qyminus = np.array( [1.0/np.sqrt(2), -1.0J/np.sqrt(2) ] )

qzplus  = np.array( [1.0, 0.0 ] )
qzminus = np.array( [0.0, 1.0 ] )
    
qxplus_f_sp = qdcl.convert_spherical_point_to_pure_state( np.pi/2.0, 0.0 )
qxminus_f_sp = qdcl.convert_spherical_point_to_pure_state( -np.pi/2.0, 0.0 )

qyplus_f_sp = qdcl.convert_spherical_point_to_pure_state( np.pi/2.0, np.pi/2.0 )
qyminus_f_sp = qdcl.convert_spherical_point_to_pure_state( np.pi/2.0, -np.pi/2.0)

qzplus_f_sp = qdcl.convert_spherical_point_to_pure_state( 0.0, 0.0 )
qzminus_f_sp = qdcl.convert_spherical_point_to_pure_state( np.pi, 0.0 )


bvxplus  = qdcl.convert_pure_state_to_bloch_vector( qxplus )
bvxminus = qdcl.convert_pure_state_to_bloch_vector( qxminus )

bvyplus  = qdcl.convert_pure_state_to_bloch_vector( qyplus )
bvyminus = qdcl.convert_pure_state_to_bloch_vector( qyminus )

bzplus   = qdcl.convert_pure_state_to_bloch_vector( qzplus )
bzminus  = qdcl.convert_pure_state_to_bloch_vector( qzminus )



