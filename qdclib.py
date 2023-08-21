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

from ExceptionsClasses import *

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d

from sklearn.datasets import make_blobs

import numpy as np
import pandas as pd
import math as math
import sympy as sympy

#
# Quantum Computing Simulator (QCS)
#

# import qcs


COSINE_DISTANCE    = 1000
DOT_DISTANCE       = 1001
FIDELITY_DISTANCE  = 1002
TRACE_DISTANCE     = 1003
MANHATTAN_DISTANCE = 1004
BURES_DISTANCE     = 1005
HS_DISTANCE        = 1006
P_CQA_DISTANCE     = 1007
P_CQB_DISTANCE     = 1008
SWAP_TEST_DISTANCE = 1009

EUCLIDEAN_DISTANCE_WITH_SQRT    = 1010
EUCLIDEAN_DISTANCE_WITHOUT_SQRT = 1011

POINTS_DRAW        = 2000
LINES_DRAW         = 2001

POINTS_MULTI_BATCH_DRAW   = 3000
LINES_MULTI_BATCH_DRAW    = 3001
VECTORS_SINGLE_BATCH_DRAW = 3002
VECTORS_MULTI_BATCH_DRAW  = 3003

OPT_COBYLA = 4000
OPT_SPSA   = 4001
OPT_SLSQP  = 4002
OPT_POWELL = 4003

def _internal_pauli_x():
    paulix=np.array( [0.0, 1.0, 1.0, 0.0] ).reshape(2,2)
    return paulix

def _internal_pauli_y():
    pauliy=np.array( [0.0, -1.0J, 1.0J, 0.0] ).reshape(2,2)
    return pauliy

def _internal_pauli_z():
    pauliz=np.array( [1.0, 0.0, 0.0, -1.0] ).reshape(2,2)
    return pauliz

def _internal_gell_mann_lambda1():
    gm_lNum=np.array( [0.0, 1.0, 0.0,
                       1.0, 0.0, 0.0,
                       0.0, 0.0, 0.0] ).reshape(3,3)
    return gm_lNum
    
def _internal_gell_mann_lambda4():
    gm_lNum=np.array( [0.0, 0.0, 1.0,
                       0.0, 0.0, 0.0,
                       1.0, 0.0, 0.0] ).reshape(3,3)
    return gm_lNum

def _internal_gell_mann_lambda6():
    gm_lNum=np.array( [0.0, 0.0, 0.0,
                       0.0, 0.0, 1.0,
                       0.0, 1.0, 0.0] ).reshape(3,3)
    return gm_lNum

def _internal_gell_mann_lambda2():
    gm_lNum=np.array( [ 0.0, -1.0J, 0.0,
                       1.0J,   0.0, 0.0,
                        0.0,   0.0, 0.0] ).reshape(3,3)
    return gm_lNum

def _internal_gell_mann_lambda5():
    gm_lNum=np.array( [ 0.0, 0.0, -1.0J,
                        0.0, 0.0,   0.0,
                       1.0J, 0.0,   0.0] ).reshape(3,3)
    return gm_lNum

def _internal_gell_mann_lambda7():
    gm_lNum=np.array( [0.0,  0.0,  0.0,
                       0.0,  0.0, -1.0J,
                       0.0, 1.0J,  0.0] ).reshape(3,3)
    return gm_lNum


def _internal_gell_mann_lambda3():
    gm_lNum=np.array( [1.0,  0.0, 0.0,
                       0.0, -1.0, 0.0,
                       0.0,  0.0, 0.0] ).reshape(3,3)
    return gm_lNum


def _internal_gell_mann_lambda8():
    gm_lNum=1.0 / np.sqrt(3) * np.array( [1.0, 0.0,  0.0,
                                          0.0, 1.0,  0.0,
                                          0.0, 0.0, -2.0] ).reshape(3,3)
    return gm_lNum



def _internal_qdcl_vector_state_to_density_matrix(q):
    return np.outer(q, np.transpose(q.conj()))

def _internal_qdcl_create_density_matrix_from_vector_state(q):
    return _internal_qdcl_vector_state_to_density_matrix(q)

def vector_state_to_density_matrix(q):
    """
    Calculates density matrix for a given vector state.

    Parameters
    ----------
    q : numpy array object
        A normalized vector state.

    Returns
    -------
    numpy ndarray
        A density matrix.
        
    Examples
    --------
    A density matrix for a correct state:
    >>> x=vector_state_to_density_matrix(np.array([1/np.sqrt(2),-1/np.sqrt(2)]))
    >>> print(x)
        [[ 0.5 -0.5]
         [-0.5  0.5]]
    If the state vector is not normalized:
    >>> print(vector_state_to_density_matrix(np.array([0+1j,1])))
        Traceback (most recent call last): ... 
        ValueError: The given vector is not a correct quantum state!

    """
    if (math.isclose(np.linalg.norm(q), 1, abs_tol=0.000001)):
        return np.outer(q, np.transpose(q.conj()))
    else:
        raise ValueError("The given vector is not a correct quantum state!")
        return None


def create_quantum_centroid(_qX, _n_elems_in_class=-1):
    rows, cols = _qX.shape
    
    centroid=np.zeros(shape=(cols,cols))
    
    for idx in range(0, rows):
        centroid = centroid + _internal_qdcl_create_density_matrix_from_vector_state( _qX[ idx, : ] )
    
    if _n_elems_in_class==-1:
        centroid = centroid * (1.0/ float(rows))
    else:
        centroid = centroid * (1.0/ float(_n_elems_in_class))
        
    return centroid

# code based on chop
# discussed at:
#   https://stackoverflow.com/questions/43751591/does-python-have-a-similar-function-of-chop-in-mathematica
def _internal_chop(expr, delta=10 ** -10):
    if isinstance(expr, (int, float, complex)):
        return 0 if -delta <= expr <= delta else expr
    else:
        return [_internal_chop(x) for x in expr]

chop = _internal_chop

def convert_qubit_pure_state_to_bloch_vector( qstate ):
    """
    
    Convert pure quantum state of qubit to
    the bloch vector representation 

    Parameters
    ----------
    qstate : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    # to check qstate is vector state
    # or matrix
    
    qstateden = _internal_qdcl_vector_state_to_density_matrix( qstate )
    
    xcoord = np.trace( _internal_pauli_x() @ qstateden )
    ycoord = np.trace( _internal_pauli_y() @ qstateden )
    zcoord = np.trace( _internal_pauli_z() @ qstateden )
    
    return np.array([xcoord, ycoord, zcoord])

def convert_spherical_point_to_bloch_vector( _r, _theta, _phi ):
    
    
    xcoord = _r * np.cos( _phi ) * np.cos( _theta )
    ycoord = _r * np.cos( _phi ) * np.sin( _theta )
    zcoord = _r * np.sin( _phi )

    return np.array([xcoord, ycoord, zcoord])

def convert_bloch_vector_to_spherical_point( _x, _y, _z ):
    
    r = np.sqrt( _x * _x + _y * _y + _z * _z )
    theta = np.arccos( _z / r )
    phi = np.sign(_y) *  np.arccos( _x / np.sqrt(_x*_x + _y*_y) ) 
    
    return np.array([r, theta, phi ])

def convert_spherical_point_to_pure_state( _theta, _phi):
    """
    Convert spherical point theta and phi angles to
    the pure quantum state

    Parameters
    ----------
    _theta : TYPE
        DESCRIPTION.
        
        0 <= _theta <= 2.0 * np.pi 
        0 <= _phi <= np.pi 
        
    _phi : TYPE
        DESCRIPTION.

    Returns
    -------
    pure_state_qubit : TYPE
        DESCRIPTION.

    """
    pure_state_qubit = create_zero_vector( 2 )
    
    pure_state_qubit[0] = np.cos( _theta / 2.0 )
    pure_state_qubit[1] = np.exp(1.0J * _phi) * np.sin( _theta / 2.0 )
    
    return pure_state_qubit

def convert_bloch_vector_to_pure_state( _x, _y, _z ):
    
    r,theta,phi = convert_bloch_vector_to_spherical_point( _x, _y, _z)
    pure_state_qubit = convert_spherical_point_to_pure_state( theta, phi  )
    
    return pure_state_qubit

def stereographic_projection_to_two_component_vector( _x, _y, _z ):
    
    two_component_vector = create_zero_vector( 2 )
    
    two_component_vector[0] = _x / (1.0 - _z)
    two_component_vector[1] = _y / (1.0 - _z)
    
    return two_component_vector
    
def vector_data_encode_with_inverse_stereographic_projection( _v ):
    d = _v.shape[0]
    
    rsltvec = np.zeros( shape=(d+1,) )
    normv = np.linalg.norm( _v )

    for idx in range(d): 
        rsltvec[ idx ] = _v[ idx ] / (normv * np.sqrt( (normv ** 2) + 1.0))

    rsltvec[d]=(normv / np.sqrt( (normv ** 2) + 1.0))

    return rsltvec

class BlochVisualization:

    def __init__( self ):
               
        self.additional_points  = [ ]
        self.additional_states  = [ ]
        self.additional_vectors = [ ]        
    
        self.radius = 2.0
        self.resolution_of_mesh = 31
        
        self.default_arrow_size = 0.2
        
        self.figure = None
        self.axes = None
        
        self.figuresize = [10, 10]
        self.viewangle = [-60, 30]
        
        self.xlabel = ["$x$", ""]
        self.xlabelpos = [2.5, -2.0]

        self.ylabel = ["$y$", ""]
        self.ylabelpos = [2.5, -2.0]

        self.zlabel = [r"$\left| 0 \right>$", r"$\left| 1 \right>$"]
        self.zlabelpos = [2.5, -2.75]   
        
        self.main_sphere_color = "#FEFEFE"
        self.main_sphere_alpha = 0.15
        
        self.frame_width = 1.0
        self.frame_color = "black"
        self.frame_axes_color = "red"
        self.frame_alpha = 0.15

        self.main_font_color = "blue"
        self.main_font_size = 25
        self.title = "Basic title for Bloch Sphere"
        
        self.point_color  = "green"
        self.vector_color = "green"

        self.point_draw_mode       = 0
        self.vector_draw_mode      = 0
        self.pure_states_draw_mode = 0
        
    def reset( self ):
        pass
    
    def make_figure( self ):
        
        self.prepare_mesh()
        f = self.render_bloch_sphere()
        
        return f
    
    def set_view(self, a, b):
        self.viewangle=[a,b]
        
    def set_title(self, _title):
        self.title = _title
    
    def prepare_mesh( self, _hemisphere = 0 ):
        if _hemisphere == 0: # north 
            self.u_angle = np.linspace(-np.pi, np.pi, self.resolution_of_mesh)
            self.v_angle = np.linspace(0.0, np.pi/2, self.resolution_of_mesh)
            
            self.x_dir = np.outer(np.cos(self.u_angle), np.sin(self.v_angle))
            self.y_dir = np.outer(np.sin(self.u_angle), np.sin(self.v_angle))
            self.z_dir = np.outer(np.ones(self.u_angle.shape[0]), np.cos(self.v_angle))
        
        if _hemisphere == 1: # south
            self.u_angle = np.linspace(-np.pi, np.pi, self.resolution_of_mesh)
            self.v_angle = np.linspace(np.pi/2, np.pi, self.resolution_of_mesh)
            
            self.x_dir = np.outer(np.cos(self.u_angle), np.sin(self.v_angle))
            self.y_dir = np.outer(np.sin(self.u_angle), np.sin(self.v_angle))
            self.z_dir = np.outer(np.ones(self.u_angle.shape[0]), np.cos(self.v_angle))
    
    def reset_draw_mode( self ):
        self.draw_mode = 0

    def enable_single_batch_draw( self ):
        self.draw_mode = POINTS_DRAW

    def enable_multi_batch_draw( self ):
        self.draw_mode = POINTS_MULTI_BATCH_DRAW

    def set_points(self, _points=None):
        self.enable_single_batch_draw()
        self.additional_points = _points.copy()
        
        for row in range(0, self.additional_points.shape[0]):
            # normalization points
            self.additional_points[row] /= np.linalg.norm(self.additional_points[row])
            self.additional_points[row] *= (self.radius + 0.01)
            
        # rescale to radius r
       
    def clear_points(self):
        self.additional_points = [ ]

    def add_points(self, _points=None, _color=None, _marker=None):
        self.draw_mode = POINTS_MULTI_BATCH_DRAW
        cp_points = _points.copy()
        
        for row in range(0, cp_points.shape[0]):
            # normalization points
            cp_points[row] /= np.linalg.norm( cp_points[row] )
            cp_points[row] *= (self.radius + 0.01)
        
        self.additional_points.append( [ cp_points, (_color, _marker) ] )
    
    def set_vectors(self, _points=None):
        self.additional_vectors = [ ]
        self.vector_draw_mode = VECTORS_SINGLE_BATCH_DRAW
        
        #
        # type check
        #
        
        self.additional_vectors = _points.copy()
        
        for row in range(0, self.additional_vectors.shape[0]):
            # normalization points
            self.additional_vectors[row] /= np.linalg.norm(self.additional_vectors[row])
            self.additional_vectors[row] *= (self.radius + 0.01)
            
        # rescale to radius r

    def clear_vectors(self):
        self.additional_vectors = [ ]

    def add_vectors(self, _points=None, _color=None, _marker=None):
        self.vector_draw_mode = VECTORS_MULTI_BATCH_DRAW
        cp_points = _points.copy()

        for row in range(0, cp_points.shape[0]):
            # normalization points
            cp_points[row] /= np.linalg.norm( cp_points[row] )
            cp_points[row] *= (self.radius + 0.01)
        
        self.additional_vectors.append( [ cp_points, (_color, _marker) ] )


    def set_pure_states(self, _states=None, _color=None):
        ptns = np.empty((0,3))
        for qstate in _states:
            qstateden = _internal_qdcl_vector_state_to_density_matrix( qstate )
            
            # change sign for x coords
            xcoord = - np.trace( _internal_pauli_x() @ qstateden )
            ycoord =   np.trace( _internal_pauli_y() @ qstateden )
            zcoord =   np.trace( _internal_pauli_z() @ qstateden )
        
            ptns = np.append( ptns, [[ xcoord, ycoord, zcoord]], axis=0)  # for state
    
        if _color is not None:
            self.point_color=_color
        self.set_points( ptns )

    def set_pure_states_as_vectors(self, _states=None, _color=None):
        ptns = np.empty((0,3))
        for qstate in _states:
            qstateden = _internal_qdcl_vector_state_to_density_matrix( qstate )
            
            # change sign for x coords
            xcoord = - np.trace( _internal_pauli_x() @ qstateden )
            ycoord =   np.trace( _internal_pauli_y() @ qstateden )
            zcoord =   np.trace( _internal_pauli_z() @ qstateden )
        
            ptns = np.append( ptns, [[ xcoord, ycoord, zcoord]], axis=0)  # for state
            
        if _color is not None:
            self.vector_color = _color
        self.set_vectors( ptns )
        
    def clear_pure_states(self):
        self.additional_states = [ ]

    def add_pure_states(self, _states=None, _color=None, _marker=None):
        pass
    
    def render_hemisphere(self):
        self.axes.plot_surface(
           self.radius * self.x_dir,
           self.radius * self.y_dir,
           self.radius * self.z_dir,
           rstride=2,
           cstride=2,
           color=self.main_sphere_color,
           linewidth=0.0,
           alpha=self.main_sphere_alpha)
        
        self.axes.plot_wireframe(
            self.radius * self.x_dir,
            self.radius * self.y_dir,
            self.radius * self.z_dir,
            rstride=5,
            cstride=5,
            color=self.frame_color,
            alpha=self.frame_alpha,
        )
       
    def render_equator_and_parallel( self ):
        self.axes.plot(
            self.radius * np.cos(self.u_angle),
            self.radius * np.sin(self.u_angle),
            zs=0,
            zdir="z",
            lw=self.frame_width,
            color=self.frame_color,
        )
        
        self.axes.plot(
            self.radius * np.cos(self.u_angle),
            self.radius * np.sin(self.u_angle),
            zs=0,
            zdir="x",
            lw=self.frame_width,
            color=self.frame_color,
        )        

        self.axes.plot(
            self.radius * np.cos(self.u_angle),
            self.radius * np.sin(self.u_angle),
            zs=0,
            zdir="y",
            lw=self.frame_width,
            color=self.frame_color,
        )    

    def render_sphere_axes( self ):
        span = np.linspace(-2.0, 2.0, 2)
        zero_span = 0.0 * span
        self.axes.plot( span, zero_span, 
                        zs=0, 
                        zdir="z", 
                        label="X", 
                        lw=self.frame_width, 
                        color=self.frame_axes_color )
        self.axes.plot( zero_span, span, 
                        zs=0, 
                        zdir="z", 
                        label="Y", 
                        lw=self.frame_width, 
                        color=self.frame_axes_color )
        self.axes.plot( zero_span, span, 
                        zs=0, 
                        zdir="y", 
                        label="Z", 
                        lw=self.frame_width, 
                        color=self.frame_axes_color )
    
    def render_labels_for_axes( self ):
        
        common_opts = { "fontsize" : self.main_font_size,
                        "color" : self.main_font_color,
                        "horizontalalignment" : "center",
                        "verticalalignment" : "center" }

        self.axes.text(0, -self.xlabelpos[0], 0, self.xlabel[0], **common_opts)
        self.axes.text(0, -self.xlabelpos[1], 0, self.xlabel[1], **common_opts)

        self.axes.text(self.ylabelpos[0], 0, 0, self.ylabel[0], **common_opts)
        self.axes.text(self.ylabelpos[1], 0, 0, self.ylabel[1], **common_opts)

        self.axes.text(0, 0, self.zlabelpos[0], self.zlabel[0], **common_opts)
        self.axes.text(0, 0, self.zlabelpos[1], self.zlabel[1], **common_opts)
    
    def render_points( self ):
        # warning: ?axis needs reorganisation?
        if  self.additional_points == []:
            return
        
        if self.draw_mode == POINTS_DRAW:
            self.axes.scatter(
                np.real(self.additional_points[:,1]),
                np.real(self.additional_points[:,0]),
                np.real(self.additional_points[:,2]),
                s=200,
                alpha=1,
                edgecolor=None,
                zdir="z",
                color=self.point_color,
                marker=".",
            )
            
        if self.draw_mode == POINTS_MULTI_BATCH_DRAW:
            for t,(c,m) in self.additional_points:
                self.axes.scatter(
                    np.real(t[:,1]),
                    np.real(t[:,0]),
                    np.real(t[:,2]),
                    s=200,
                    alpha=1,
                    edgecolor=None,
                    zdir="z",
                    color=c,
                    marker=m,
                )
                
    def render_vectors( self ):
        if self.additional_vectors == []:
            return        
        if self.vector_draw_mode == VECTORS_SINGLE_BATCH_DRAW:
            for idx in range(self.additional_vectors.shape[0]):
                self.axes.quiver(
                    0.0,0.0,0.0,
                    np.real(self.additional_vectors[idx,1]),
                    np.real(self.additional_vectors[idx,0]),
                    np.real(self.additional_vectors[idx,2]),
                    color=self.vector_color if not None else "green",
                    arrow_length_ratio=self.default_arrow_size,
                    #marker="x",
                )            
        
        if self.vector_draw_mode == VECTORS_MULTI_BATCH_DRAW:
            for t,(c,m) in self.additional_vectors:
                for idx in range(t.shape[0]):
                    self.axes.quiver(
                        0.0,0.0,0.0,
                        np.real(t[idx,1]),
                        np.real(t[idx,0]),
                        np.real(t[idx,2]),
                        color=c if not None else "green",
                        arrow_length_ratio=self.default_arrow_size,
                        #marker="x",
                    )            
    
    def render_pure_states( self ):
        pass

    def render_pure_states_as_vectors( self ):
        pass
    
    def render_bloch_sphere( self ):        
        self.figure = plt.figure( figsize=self.figuresize )
        self.axes = Axes3D( self.figure,
                            azim=self.viewangle[0],
                            elev=self.viewangle[1] )

        self.axes.clear()
        self.axes.set_axis_off()
        
        self.figure.add_axes( self. axes )

        self.axes.set_xlim3d( -2.0, 2.0 )
        self.axes.set_ylim3d( -2.0, 2.0 )
        self.axes.set_zlim3d( -2.0, 2.0 )
        self.axes.set_aspect( 'equal' )
        
        self.axes.grid(False)
        
        self.axes.set_title(self.title, fontsize=self.main_font_size, y=0.95)

        
        # top/north hemisphare 
        # for state |0>
        self.prepare_mesh(0)
        self.render_hemisphere()
        
        # bottom/south hemisphare 
        # for state |1>        
        self.prepare_mesh(1)
        self.render_hemisphere()

        self.render_points()
        self.render_vectors()

        self.render_equator_and_parallel()

        self.render_sphere_axes()

        self.render_labels_for_axes()


        return self.figure
    
    def save_to_file(self, filename = None):
        pass

# in preparation
class QuantumSVM:
    
    def __int__( self ):
        pass
    
    def reset( self ):
        pass
        
# in preparation    
class VQEClassification:
    
    def __int__( self ):
        self.params_filename = None, 
        self.save_params=0
        
        self.optymizer = None
        self.optimizer_type = OPT_COBYLA

    def reset( self ):
        pass
    
    def objective_function( self ):
        pass
    
    def create_variational_circuit( self, _n_qubits, _tab_parameters, _circuit_type_form, _n_layers):
        pass    
    
    def train_vqe( self, _initial_params, _state_for_train):    
        pass
    
    def set_angles_file_name( self, _fname ):
        self.params_filename = _fname
    
    def save_angles_to_file( self, _fname = None ):
        pass

    def load_angles_from_file( self, _fname = None ):
        pass

# in preparation
class DistanceQuantumClassification:
    def __int__( self ):
        pass
    
    def reset( self ):
        pass

# in preparation
class QuantumSpectralClustering:
    def __int__( self ):
        pass
    
    def reset( self ):
        pass

# in preparation
class ClusteringByPotentialEnergy:
    def __int__( self ):
        self.dimension = 2
        self.bigE = 0
        self.data_for_cluster = [] 
        self._func_distance = None

    def reset( self ):
        pass
    
    def set_distnace(self, _f_dist):
        self._func_distance = _f_dist
            
    def set_dimension( self, _d ):
        self.dimension = _d
    
    def set_data(self, _qdX):
        self.data_for_cluster = _qdX
        
    def calc_V( self, _x, _E, _sigma ):
        two_sigma_sqr = ( 2.0 * (_sigma ** 2.0) )
        
        _psi    = 0.0
        sumval  = 0.0
        sumval1 = 0.0
        sumval2 = 0.0

        for dval in self.data_for_cluster:
            # individual index in probe
            for idx in range(self.dimension):
                #dij2 = self._func_distance( _x, dval ) ** 2.0
                dij2 = (_x[idx] - dval[idx]) ** 2
                evalue = np.exp( -1.0 * ( (dij2)/(two_sigma_sqr) ) )
                _psi = _psi + evalue
                sumval1 = sumval1 + dij2 * evalue
                sumval2 = sumval2 + evalue
        
        sumval = sumval + (sumval1/sumval2)
        
        coeff = 1.0 / ( 2.0 * (two_sigma_sqr) * _psi )
        
        rslt = _E - (self.dimension/2.0) + coeff * sumval
        
        return rslt

    def calc_V_with_distance( self, _x, _E, _sigma ):
        pass

    
    def calc_v_function_on_2d_mesh_grid(self, _sigma, _mesh_grid_x = 50, _mesh_grid_y = 50 ):

        minx=np.min(self.data_for_cluster[:, 0])
        maxx=np.max(self.data_for_cluster[:, 0])
        
        miny=np.min(self.data_for_cluster[:, 1])
        maxy=np.max(self.data_for_cluster[:, 1])    

        X, Y = np.mgrid[minx:maxx:_mesh_grid_x*1J, miny:maxy:_mesh_grid_y*1J]
        Z = np.zeros( shape=X.shape)
        for idx in range(_mesh_grid_x):
            for idy in range(_mesh_grid_y):
                v=(self.calc_V( [X[idx,idy], Y[idx,idy]], 0.0, _sigma))
                Z[idx,idy] = (-v)/(1.0+v)
        
        return Z

    def calc_v_function_with_distance_on_2d_mesh_grid(self, _sigma):
        pass
    

def create_circle_plot_for_2d_data(_qX, _first_col=0, _second_col=1, _limits=None):
    """
        Drawing a circle plot for two-dimensional data. 

        Parameters
        ----------
        _qX : numpy ndarray
            File of input data.
        _first_col : interger
            The variable defining the first dimension (default column indexed 
            as 0). 
        _second_col : interger
            The variable defining the second dimension (default column indexed 
            as 1). 
        _limits : DESC TO FIX a single row numpy ndarray
            DESC TO FIX
        Returns
        -------
        fig : plot

        Example 1
        -------
        From file 'SYNTH_Training.xlsx', we fetch first two columns and draw
        two-dimensional plot:
        >>> df = pd.read_excel(r'SYNTH_Training.xlsx')
        >>> tab = pd.DataFrame(df).to_numpy()
        >>> create_circle_plot_for_2d_data(tab)
        Example 2
        -------
        DESC TO FIX
        >>> f=qdcl.create_circle_plot_for_2d_data( 
                qdcl.get_data_for_class(d,labels,3),
                0, 1, [-1,1,-1,1] )
        
    """

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    
    if _limits == None:
        ax.set_xlim( [-1.0, 1.0] )
        ax.set_ylim( [-1.0, 1.0] )
        circle = plt.Circle( (0,0), 1,  color='r', fill=False)
    else:
        ax.set_xlim( _limits[0], _limits[1] )
        ax.set_ylim( _limits[2], _limits[3] )
        circle = plt.Circle( (0,0), _limits[4],  color='r', fill=False)

    ax.scatter( _qX[ :, _first_col ], _qX[ :, _second_col ])
    ax.add_patch(circle)
    
    ax.set_xlabel('Feature 1 (X axis)')
    ax.set_ylabel('Feature 2 (Y axis)')

    return fig

def create_scatter_plot_for_2d_data(_qX, _first_col=0, _second_col=1, _limits=None):
    """
        Drawing a scatter plot for two-dimensional data. 

        Parameters
        ----------
        _qX : numpy ndarray
            File of input data.
        _first_col : interger
            The variable defining the first dimension (default column indexed 
            as 0). 
        _second_col : interger
            The variable defining the second dimension (default column indexed 
            as 1). 
        _limits : DESC TO FIX a single row numpy ndarray
            DESC TO FIX
        Returns
        -------
        fig : plot

        Example 1
        -------
        From file 'SYNTH_Training.xlsx', we fetch first two columns and draw
        two-dimensional plot:
        >>> df = pd.read_excel(r'SYNTH_Training.xlsx')
        >>> tab = pd.DataFrame(df).to_numpy()
        >>> create_scatter_plot_for_2d_data(tab)
        
        Example 2
        -------
        DESC TO FIX
        >>> f=qdcl.create_scatter_plot_for_2d_data( 
                qdcl.get_data_for_class(d,labels,3),
                0, 1, [-1,1,-1,1] )
        
    """

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    if _limits == None:
        ax.set_xlim( [-1.0, 1.0] )
        ax.set_ylim( [-1.0, 1.0] )
    else:
        ax.set_xlim( _limits[0], _limits[1] )
        ax.set_ylim( _limits[2], _limits[3] )
    
    ax.scatter( _qX[ :, _first_col ], _qX[ :, _second_col ])
    
    ax.set_xlabel('Feature 1 (X axis)')
    ax.set_ylabel('Feature 2 (Y axis)')
    
    return fig

def create_circle_plot_with_centers_for_2d_data(_qX, _n_clusters, _centers, _labels):
    """
        Drawing a circle plot for two-dimensional data with pointed out clusters
        centers. 

        Parameters
        ----------
        _qX : numpy ndarray
            Input data from which first two columns will be fetched 
            (coordinates of the points). 
        _n_clusters : integer
            The number of clusters.
        _centers : numpy ndarray
            The coordinates of clusters centers.
        _labels : numpy ndarray
            The labels for each pair of coordinates from _qX. 

        Returns
        -------
        fig : plot

        Example
        -------
        We can use other functions from this library to generate probes 
        (create_focused_circle_probes_with_uniform_placed_centers) and use 
        k-means method to cluster them (kmeans_quantum_states). Finally, the
        plot will be drawn:
        >>> n_clusters = 3
        >>> d = create_focused_circle_probes_with_uniform_placed_centers(20, n_clusters, _width_of_cluster=0.15)
        >>> labels, centers = kmeans_quantum_states( d, n_clusters, _func_distance=COSINE_DISTANCE )
        >>> print(d)
            [[-0.00353932  0.99999374]
             [ 0.00823482  0.99996609]
             [-0.79415079 -0.60772076]
             [ 0.19272468  0.98125287]
             [ 0.12038379  0.99272743]
             [-0.22356451  0.97468913]...
        >>> print(labels)
            [0 0 2 0 0 0 1 2 1 2 1 2 1 2 1 1 2 2 0 1]
        >>> print(centers)
            [[ 0.04627963  0.99892852]
             [ 0.80666692 -0.59100633]
             [-0.82687593 -0.56238438]]
        >>> f = create_circle_plot_with_centers_for_2d_data( d, n_clusters, centers, labels )
        >>> f.show()
        
    """
    
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    circle = plt.Circle( (0,0), 1,  color='r', fill=False)
    ax.scatter( _qX[:,0], _qX[:,1], c=_labels)
    ax.scatter(_centers[:, 0], _centers[:, 1], marker='x', color='g')
    for idx in range(_n_clusters):
        ax.annotate("", xy=(_centers[idx, 0], _centers[idx, 1]), xytext=(0, 0), arrowprops=dict(arrowstyle="->"))
    ax.add_patch(circle)

    return fig
    
def convert_data_to_vector_states_double_norm(inputDF, cols=0):
    """
        Create quantum states - input data comes from a Pandas Data Frame, 
        each variable is normalized to avoid domination of some variables 
        (approach known from the classical machine learning), finally, 
        each observation is normalized (to generate correct quantum state). 

        Parameters
        ----------
        inputDF : pandas.DataFrame
            File of input data.
        cols : interger
            If we would like to fetch fewer columns than the file contains, 
            this number should be assigned to the variable (if cols==0, 
            then all columns will be used). 

        Returns
        -------
        Qtab : numpy array
            Numpy array of normalized quantum states.

        Examples
        --------
        From file 'name.xlsx', seven columns were fetched to produce 3-qubit states.
        >>> df = pd.read_excel(r'name.xlsx')
        >>> print(convert_data_to_vector_states(df,7))
        [0.21382269 0.23794088 0.         0.08154363 0.54498154 0.54498154
         0.54498154 0.        ]...
    """
    a,b=inputDF.shape
    if cols==0:
        print("The number of variables is:", b)
    elif b>cols and cols>0:
        b=cols
        print("The number of variables is:", b)
    else:
        raise ValueError("The number of variables is incorrect!")
        return None
    #classical normalization
    Ktab=pd.DataFrame(inputDF).to_numpy()
    #intervals
    maxs=np.amax(Ktab, axis=0)
    mins=np.amin(Ktab, axis=0)
    intervals=np.zeros(shape=(b))
    for i in range(b):
        intervals[i]=maxs[i]-mins[i]
    KNtab=np.ndarray(shape=(a,b))
    for j in range(b):
        for i in range(a):
            KNtab[i,j]=(Ktab[i,j]-mins[j])/intervals[j]
    #quantum normalization
    if math.ceil(math.log2(b)) != math.floor(math.log2(b)):
        c=math.ceil(math.log2(b))
        c=2**c
        Qtab=np.zeros(shape=(a,c))
    else:
        Qtab=np.ndarray(shape=(a,b))
    for i in range(a):
        sum_all=0
        for j in range(b):
            sum_all+=KNtab[i,j]
        for j in range(b):
            Qtab[i,j]=sympy.sqrt(KNtab[i,j]/sum_all)
    return Qtab

def convert_data_to_vector_states(inputDF, cols=0):
    """
        Create quantum states - input data from Pandas Data Frame

        Parameters
        ----------
        inputDF : pandas.DataFrame
            File of input data.
        cols : interger
            If we would like to fetch fewer columns than the file contains, 
            this number should be assigned to the variable (if cols==0, 
            then all columns will be used).

        Returns
        -------
        Qtab : numpy array
            Numpy array of normalized quantum states.

        Examples
        --------
        From file 'name.xlsx', four columns were fetched to produce 2-qubit states.
        >>> df = pd.read_excel(r'name.xlsx')
        >>> print(convert_data_to_vector_states(df,4))
        [[0.32438643 0.94034392 0.00251242 0.10256923]
         [0.38518862 0.91692333 0.00243558 0.10428524]
         [0.39649659 0.91048255 0.00235002 0.11750089]
         [0.40813284 0.91291981 0.00223538 0.        ]...
    """
    a,b=inputDF.shape
    if cols==0:
        print("The number of variables is:", b)
    elif b>cols and cols>0:
        b=cols
        print("The number of variables is:", b)
    else:
        raise ValueError("The number of variables is incorrect!")
        return None
    Ktab=pd.DataFrame(inputDF).to_numpy()
    #quantum normalization
    if math.ceil(math.log2(b)) != math.floor(math.log2(b)):
        c=math.ceil(math.log2(b))
        c=2**c
        Qtab=np.zeros(shape=(a,c))
    else:
        Qtab=np.ndarray(shape=(a,b))
    for i in range(a):
        sum_all=0
        for j in range(b):
            sum_all+=Ktab[i,j]
        for j in range(b):
            Qtab[i,j]=sympy.sqrt(Ktab[i,j]/sum_all)
    return Qtab


def convert_data_to_vector_state(dataTuple):
    """
        Create a quantum state - parameters as input data

        Parameters
        ----------
        dataTuple : tuple of integers or real numbers 
            Input data as tuple of integers or real numbers.

        Returns
        -------
        Qvec : numpy vector
            Numpy vector containing a normalized quantum state.

        Examples
        --------
        Create a 2-qubit state.
        >>> print(convert_data_to_vector_state((5,4,3,1)))
        [0.62017367 0.5547002  0.48038446 0.2773501 ]
        Create a 3-qubit state.
        >>> print(convert_data_to_vector_state((5,4,3,1,7)))
        [0.5        0.4472136  0.38729833 0.2236068  0.59160798 0.
         0.         0.        ]
    """
    a=len(dataTuple)
    #quantum normalization
    if math.ceil(math.log2(a)) != math.floor(math.log2(a)):
        b=math.ceil(math.log2(a))
        b=2**b
        Qvec=np.zeros(shape=(b))
    else:
        Qvec=np.ndarray(shape=(a))
    for i in range(a):
        Qvec[i]=dataTuple[i]
    sum_all=0
    for i in range(a):
        sum_all+=Qvec[i]
    for i in range(a):
        Qvec[i]=sympy.sqrt(Qvec[i]/sum_all)
    return Qvec

def create_learning_and_test_set(inputDF, var_no, learn_set_size=0.8, norm_type=0):
    """
        Create learning and test set of quantum states - input data from 
        Pandas Data Frame

        Parameters
        ----------
        inputDF : pandas.DataFrame
            File of input data.
        var_no : interger
            The number of variables to fetch from file. The last column 
            (variable) is always treated as a target variable.
        learn_set_size : float in range (0,1)
            The percent of observations to include in the learning set (the rest
            forms the test set). Default partition is 80% observations in 
            the learning set and 20% in the test set.
        norm_type : Boolean
            If single normalization is to be utilized then norm_type==0 (realized 
            by the function convert_data_to_vector_states). If double normalization 
            has to be performed norm_type==1 (realized by the function 
            convert_data_to_vector_states_double_norm).

        Returns
        -------
        Tuple
            Tuple of two numpy arrays containing learning and test set.

        Examples
        --------
        From file 'name.xlsx', three columns were fetched to produce 2-qubit states.
        >>> df = pd.read_excel(r'name.xlsx')
        >>> tp=create_learning_and_test_set(df,3)
        The number of variables is: 3
        The number of classes is: 2
        >>> print(tp[0])
        [[0.51199211 0.46564845 0.72182796 0.         0.        ]
         [0.50431663 0.47174501 0.72327131 0.         0.        ]
         [0.50552503 0.46547467 0.72648316 0.         0.        ]
         [0.50529115 0.45837363 0.731146   0.         0.        ]
         [0.50716649 0.45822893 0.72993726 0.         0.        ]...
        >>> print(tp[1])
        [[0.50597801 0.47623957 0.71915376 0.         0.        ]
         [0.4979296  0.47759729 0.72385561 0.         0.        ]
         [0.49186938 0.47857315 0.72734604 0.         0.        ]
         [0.50082988 0.47003216 0.72680066 0.         0.        ]
         [0.49958661 0.47237749 0.72613547 0.         0.        ]...
    """
    if learn_set_size <=0 or learn_set_size>=1:
        raise ValueError("Incorrect proportion between learning and testing set!")
        return None
    else:
        if norm_type==0:
            set_big=convert_data_to_vector_states(inputDF, var_no)
        elif norm_type==1:
            set_big=convert_data_to_vector_states_double_norm(inputDF, var_no)
        else:
            raise ValueError("Incorrect value of parameter norm_type!")
            return None
        target=pd.DataFrame(inputDF.iloc[: , -1]).to_numpy()
        Tab_class, cl_counts = np.unique(target, return_counts=True)
        class_no=len(Tab_class)
        print('The number of classes is:',class_no)
        set_big = np.append(set_big, target, axis=1)
        Tab_data = np.ndarray(shape=(2,class_no))
        for i in range(class_no):
            Tab_data[0,i]=round(cl_counts[i]*learn_set_size)
            Tab_data[1,i]=cl_counts[i]-Tab_data[0,i]
        a,b=set_big.shape
        lset = np.ndarray(shape=(1,b))
        tset = np.ndarray(shape=(1,b))  #int(y)
        counter=np.zeros(shape=class_no)
        for j in range(a): 
            cl=set_big[j,b-1]
            z=0
            i=0
            while z==0:
                if Tab_class[i]==cl:
                    z=1
                else:
                    i+=1
            if counter[i]<Tab_data[0,i]:
                lset=np.append(lset, [set_big[j,:]], axis=0)
                counter[i]+=1
            else:
                tset=np.append(tset, [set_big[j,:]], axis=0)
        lset = np.delete(lset, 0, 0)
        tset = np.delete(tset, 0, 0)
        return lset,tset

def is_vector_normalized(vec):
    """
    Checks if the entered vector is normalized (is a correct quantum state).

    Parameters
    ----------
    vec : numpy array object
        A vector state.

    Returns
    -------
    Boolean value, None
        True if the vector is normalized, False if not, and None if entered 
        parameter is not a numpy array.
        
    Examples
    --------
    If the entered vector is a correct quantum state:
    >>> x=is_vector_normalized(np.array([1/math.sqrt(2),-1/math.sqrt(2)]))
    >>> print(x)
        True
    If the state vector is not normalized:
    >>> print(is_vector_normalized(np.array([0+1j,1])))
        False
    If the entered parameter is not a vector:
    >>> print(is_vector_normalized(7))
        None

    """
    if not(isinstance(vec, np.ndarray)):
        return None
    else:
        if (math.isclose(np.linalg.norm(vec), 1, abs_tol=0.000001)):
            return True
        else:
            return False

def vector_check(vec):
    """
    Calls the function is_vector_normalized which checks if the entered vector 
    is normalized (is a correct quantum state) and returns 1 if its true 
    (otherwise raises the exceptions).

    Parameters
    ----------
    vec : numpy array object
        A vector state.

    Returns
    -------
    1, None
        One if the entered parameter is a normalized vector, None otherwise.
        
    Examples
    --------
    If the entered vector is a correct quantum state:
    >>> x=vector_check(np.array([1/math.sqrt(2),-1/math.sqrt(2)]))
    >>> print(x)
        1
    If the state vector is not normalized:
    >>> print(is_vector_normalized(np.array([0+1j,1])))
        ...
        ValueError: The vector is not normalized!
    If the entered parameter is not a vector:
    >>> print(is_vector_normalized(5))
        ...
        TypeError: The parameter is not a vector!

    """
    x=is_vector_normalized(vec)
    if x is True:
        return 1
    else:
        if x is None:
            raise TypeError("The parameter is not a vector!")
        if x is False:
            raise ValueError("The vector is not normalized!")
        return None


def manhattan_distance(uvector, vvector, r=0, check=0): 
    """
    Caclutales the Manhattan distance between two pure (vector) states.

    Parameters
    ----------
    uvector, vvector : numpy array objects
        Vectors of complex numbers describing quantum states.
    r : integer
        The number of decimals to use while rounding the number (default is 0,
        i.e. the number is not rounded).
    check : Boolean
        If check==1 then paramatres uvector, vvector are checked for being 
        normalized vectors (as default it is not checked).

    Returns
    -------
    d : float, None
        The distance between given quantum states according to the Manhattan 
        (Taxicab) distance. None if the entered vectors are checked for being
        correct quantum states and they are not.
        
    Examples
    --------
    A distance between the same states:
    >>> v=np.array([1,0])
    >>> u=np.array([1,0])
    >>> print(manhattan_distance(u, v))
        0.0
    A distance between the orthogonal states:
    >>> v=np.array([1/math.sqrt(2),0 + 1j/math.sqrt(2)])
    >>> u=np.array([1/math.sqrt(2),0 - 1j/math.sqrt(2)])
    >>> print(manhattan_distance(u, v, 5))
        1.41421
    A distance between two examplary states:
    >>> v=np.array([1/math.sqrt(2),1/math.sqrt(2)])
    >>> u=np.array([1/math.sqrt(2),0 + 1j/math.sqrt(2)])
    >>> print(manhattan_distance(u, v, 3))
        1.0
    If entered vector v is not a correct quantum state:
    >>> v=np.array([1,1])
    >>> u=np.array([1,0])
    >>> print(manhattan_distance(u, v, 0, 1))
        ...
        ValueError: The vector is not normalized!

    """
    x=1
    y=1
    if check==1:
        x=vector_check(uvector)
        y=vector_check(vvector)
    if (x==1 and y==1):
        d=0.0
        dim=uvector.shape[0]
        for idx in range(dim):
            d = d + np.abs( (uvector[idx] - vvector[idx]) )
        if r!=0:
            d=round(d,r)
        return d
    else:
        return None

def cosine_distance( uvector, vvector, r = 0, check=0 ):
    """
    Calculates a cosine distance between two vectors.

    Parameters
    ----------
    uvector, vvector : numpy array objects
        Vectors of complex numbers describing quantum states.
    r : integer
        The number of decimals to use while rounding the number (default is 0,
        i.e. the number is not rounded).
    check : Boolean
        If check==1 then paramatres uvector, vvector are checked for being 
        normalized vectors (as default it is not checked).

    Returns
    -------
    distance_value : complex
        The distance between given quantum states according to the cosine 
        distance. In case of cosine similarity, its value is 1 for the same 
        vectors, 0 for ortogonal vectors, and (-1) for opposite vectors.
        The cosine distance is 0 for the same vectors, 1 for ortogonal vectors, 
        and 2 for opposite vectors.
        
    Examples
    --------
    A distance between the same states:
    >>> v=np.array([1,0])
    >>> u=np.array([1,0])
    >>> print(cosine_distance(u, v))
        0.0
    A distance between the opposite states:
    >>> v=np.array([1,0])
    >>> u=np.array([-1,0])
    >>> print(cosine_distance(u, v))
        2.0
    A distance between the orthogonal states:
    >>> v=np.array([1/math.sqrt(2),0 + 1j/math.sqrt(2)])
    >>> u=np.array([1/math.sqrt(2),0 - 1j/math.sqrt(2)])
    >>> print(cosine_distance(u, v))
        (1+0j)
    >>> v=np.array([1/math.sqrt(2),1/math.sqrt(2)])
    >>> u=np.array([1/math.sqrt(2),-1/math.sqrt(2)])
    >>> print(cosine_distance(u, v))
        1.0
    A distance between two examplary states:
    >>> v=np.array([1/math.sqrt(2),1/math.sqrt(2)])
    >>> u=np.array([1/math.sqrt(2),0 + 1j/math.sqrt(2)])
    >>> print(cosine_distance(u, v, 3))
        (0.5+0.5j)
    If entered vector v is not a correct quantum state:
    >>> v=np.array([1,1])
    >>> u=np.array([1,0])
    >>> print(cosine_distance(u, v, 0, 1))
        ...
        ValueError: The vector is not normalized!

    """
    x=1
    y=1
    if check==1:
        x=vector_check(uvector)
        y=vector_check(vvector)
    if (x==1 and y==1):
        similarity = np.vdot(uvector, vvector) 
        if r == 0:
            distance_value = 1.0 - similarity
        else:
            distance_value = np.round(1.0 - similarity, r)
            
        return np.linalg.norm( distance_value )
    else:
        return None


def dot_product_as_distance( uvector, vvector, r=0, check=0 ):
    """
    Calculates a dot product as a distance between two vectors.

    Parameters
    ----------
    uvector, vvector : numpy array objects
        Vectors of complex numbers describing quantum states.
    r : integer
        The number of decimals to use while rounding the number (default is 0,
        i.e. the number is not rounded).
    check : Boolean
        If check==1 then paramatres uvector, vvector are checked for being 
        normalized vectors (as default it is not checked).

    Returns
    -------
    rslt : float
        The distance between given quantum states according to the dot product.
        If the states are the same, then the distance is zero; for orthogonal 
        states the distance is one.
        
    Examples
    --------
    A distance between the same states:
    >>> v=np.array([1,0])
    >>> u=np.array([1,0])
    >>> print(dot_product_as_distance(u, v))
        0.0
    A distance between the orthogonal states:
    >>> v=np.array([1/math.sqrt(2),0 + 1j/math.sqrt(2)])
    >>> u=np.array([1/math.sqrt(2),0 - 1j/math.sqrt(2)])
    >>> print(dot_product_as_distance(u, v))
        1.0
    A distance between two examplary states:
    >>> v=np.array([1/math.sqrt(2),1/math.sqrt(2)])
    >>> u=np.array([1/math.sqrt(2),0 + 1j/math.sqrt(2)])
    >>> print(dot_product_as_distance(u, v, 5))
        0.29289
    If entered vector v is not a correct quantum state:
    >>> v=np.array([1,1])
    >>> u=np.array([1,0])
    >>> print(dot_product_as_distance(u, v, 0, 1))
        ...
        ValueError: The vector is not normalized!

    """
    x=1
    y=1
    if check==1:
        x=vector_check(uvector)
        y=vector_check(vvector)
    if (x==1 and y==1):
        if r==0:
            rslt = 1.0 - np.linalg.norm( np.vdot( vvector, uvector ) )
        else:
            rslt = round( 1.0 - np.linalg.norm( np.vdot( vvector, uvector ) ), r )
        
        return rslt
    else:
        return None

def fidelity_measure( uvector, vvector, r=0, check=0):
    """
    Calculates the Fidelity measure for two pure (vector) states.
    If the states are similar, the measure's value tends to one. 
    For orthogonal states this value is zero.

    Parameters
    ----------
    uvector, vvector : numpy array objects
        Vectors of complex numbers describing quantum states.
    r : integer
        The number of decimals to use while rounding the number (default is 0,
        i.e. the number is not rounded).
    check : Boolean
        If check==1 then paramatres uvector, vvector are checked for being 
        normalized vectors (as default it is not checked).

    Returns
    -------
    rslt: float
        A value of the Fidelity measure.
    Examples
    --------
    A value of the Fidelity measure for the same states:
    >>> v=np.array([1,0])
    >>> u=np.array([1,0])
    >>> print(fidelity_measure(u, v))
        1.0
    A value of the Fidelity measure for the orthogonal states:
    >>> v=np.array([1/math.sqrt(2),1/math.sqrt(2)])
    >>> u=np.array([1/math.sqrt(2),-1/math.sqrt(2)])
    >>> print(fidelity_measure(u, v))
        0.0
    A value of the Fidelity measure for two examplary states:
    >>> v=np.array([1/math.sqrt(2),1/math.sqrt(2)])
    >>> u=np.array([1/math.sqrt(2),0 + 1j/math.sqrt(2)])
    >>> print(fidelity_measure(u, v, 5))
        0.5
    If entered vector v is not a correct quantum state:
    >>> v=np.array([1,1])
    >>> u=np.array([1,0])
    >>> print(*_distance(u, v, 0, 1))
        ...
        ValueError: The vector is not normalized!

    """  
    x=1
    y=1
    if check==1:
        x=vector_check(uvector)
        y=vector_check(vvector)
    if (x==1 and y==1):
        if r==0:
            rslt = ( np.linalg.norm( np.vdot( vvector, uvector ) ) ** 2.0 )
        else:
            rslt = round(( np.linalg.norm( np.vdot( vvector, uvector ) ) ** 2.0 ), r)
        
        return rslt
    else:
        return None

def fidelity_as_distance( uvector, vvector, r=0, check=0 ):
    """
    Calculates a distance between two pure states using the Fidelity measure.
    If the states are similar, the distance tends to zero. For orthogonal states
    the distance is one. 

    Parameters
    ----------
    uvector, vvector : numpy array objects
        Vectors of complex numbers describing quantum states.
    r : integer
        The number of decimals to use while rounding the number (default is 0,
        i.e. the number is not rounded).
    check : Boolean
        If check==1 then paramatres uvector, vvector are checked for being 
        normalized vectors (as default it is not checked).
        
    Returns
    -------
    Float
        The distance between given quantum states.
    Examples
    --------
    A distance between the same states (according to the Fidelity measure):
    >>> v=np.array([0+1j,0])
    >>> u=np.array([1,0])
    >>> print(fidelity_as_distance(u, v))
        0.0
    A distance between the orthogonal states:
    >>> v=np.array([1/math.sqrt(2),0 + 1j/math.sqrt(2)])
    >>> u=np.array([1/math.sqrt(2),0 - 1j/math.sqrt(2)])
    >>> print(fidelity_as_distance(u, v))
        1.0
    If entered vector v is not a correct quantum state:
    >>> v=np.array([1,1])
    >>> u=np.array([1,0])
    >>> print(fidelity_as_distance(u, v, 0, 1))
        ...
        ValueError: The vector is not normalized!

    """
    return 1.0 - fidelity_measure( vvector, uvector, r, check )
 

def bures_distance( uvector, vvector, r=0, check=0 ):
    """
    Caclutales the Bures distance between two pure states.

    Parameters
    ----------
    uvector, vvector : numpy array objects
        Vectors of complex numbers describing quantum states.
    r : integer
        The number of decimals to use while rounding the number (default is 0,
        i.e. the number is not rounded).
    check : Boolean
        If check==1 then paramatres uvector, vvector are checked for being 
        normalized vectors (as default it is not checked).

    Returns
    -------
    Float
        The distance between given quantum states according to the Bures measure.
    Examples
    --------
    A distance between the same states:
    >>> v=np.array([1,0])
    >>> u=np.array([1,0])
    >>> print(bures_distance(u, v))
        0.0
    A distance between the orthogonal states:
    >>> v=np.array([1/math.sqrt(2),0 + 1j/math.sqrt(2)])
    >>> u=np.array([1/math.sqrt(2),0 - 1j/math.sqrt(2)])
    >>> print(bures_distance(u, v))
        2.0
    A distance between two examplary states:
    >>> v=np.array([1/math.sqrt(2),1/math.sqrt(2)])
    >>> u=np.array([1/math.sqrt(2),0 + 1j/math.sqrt(2)])
    >>> print(bures_distance(u, v, 3))
        0.586
    If entered vector v is not a correct quantum state:
    >>> v=np.array([1,1])
    >>> u=np.array([1,0])
    >>> print(bures_distance(u, v, 0, 1))
        ...
        ValueError: The vector is not normalized!

    """
    
    if r==0:
        rslt = 2 - 2*math.sqrt(fidelity_measure(uvector, vvector, r, check))
    else:
        rslt = round(( 2 - 2*math.sqrt(fidelity_measure(uvector, vvector, r, check)) ), r)
    
    return rslt

def hs_distance( uvector, vvector, r=0, check=0 ):
    """
    Caclutales the Hilbert-Schmidt distance between two pure states.

    Parameters
    ----------
    uvector, vvector : numpy array objects
        Vectors of complex numbers describing quantum states.
    r : integer
        The number of decimals to use while rounding the number (default is 0,
        i.e. the number is not rounded).
    check : Boolean
        If check==1 then paramatres uvector, vvector are checked for being 
        normalized vectors (as default it is not checked).

    Returns
    -------
    Float
        The distance between given quantum states according to 
        the Hilbert-Schmidt measure.
    Examples
    --------
    A distance between the same states:
    >>> v=np.array([1,0])
    >>> u=np.array([1,0])
    >>> print(hs_distance(u, v))
        0.0
    A distance between the orthogonal states:
    >>> v=np.array([1/math.sqrt(2),0 + 1j/math.sqrt(2)])
    >>> u=np.array([1/math.sqrt(2),0 - 1j/math.sqrt(2)])
    >>> print(hs_distance(u, v, 5))
        1.41421
    A distance between two examplary states:
    >>> v=np.array([1/math.sqrt(2),1/math.sqrt(2)])
    >>> u=np.array([1/math.sqrt(2),0 + 1j/math.sqrt(2)])
    >>> print(hs_distance(u, v, 3))
        1.0
    If entered vector v is not a correct quantum state:
    >>> v=np.array([1,1])
    >>> u=np.array([1,0])
    >>> print(hs_distance(u, v, 0, 1))
        ...
        ValueError: The given vector is not a correct quantum state!

    """
    if check==0:
        qu=_internal_qdcl_vector_state_to_density_matrix(uvector)
        qv=_internal_qdcl_vector_state_to_density_matrix(vvector)
    elif check==1:
        qu=vector_state_to_density_matrix(uvector)
        qv=vector_state_to_density_matrix(vvector)
    else:
        raise ValueError("Incorrect value of the parameter 'check'!")
        return None
    rp=sympy.re(np.trace(np.subtract(qu,qv) @ np.subtract(qu,qv)))
    if r==0:
        rslt=math.sqrt(rp)
    else:
        rslt=round(math.sqrt(rp),r)
    return rslt


def trace_distance_vector( _uvector, _vvector, _r=0, _check=0 ):
    """
    Calculates the distance based on density matrix trace of two pure states.

    Parameters
    ----------
    uvector, vvector : numpy array objects
        Vectors of complex numbers describing quantum states.
    r : integer
        The number of decimals to use while rounding the number (default is 0,
        i.e. the number is not rounded).
    check : Boolean
        If check==1 then paramatres uvector, vvector are checked for being 
        normalized vectors (as default it is not checked).

    Returns
    -------
    rslt : float
        The distance between given quantum states according to 
        the trace distance.
    Examples
    --------
    A distance between the same states:
    >>> v=np.array([1,0])
    >>> u=np.array([1,0])
    >>> print(trace_distance(u, v))
        0.0
    A distance between the orthogonal states:
    >>> v=np.array([1/math.sqrt(2),0 + 1j/math.sqrt(2)])
    >>> u=np.array([1/math.sqrt(2),0 - 1j/math.sqrt(2)])
    >>> print(trace_distance(u, v))
        1.0
    A distance between two examplary states:
    >>> v=np.array([1/math.sqrt(2),1/math.sqrt(2)])
    >>> u=np.array([1/math.sqrt(2),0 + 1j/math.sqrt(2)])
    >>> print(trace_distance(u, v, 5))
        0.70711
    If entered vector v is not a correct quantum state:
    >>> v=np.array([1,1])
    >>> u=np.array([1,0])
    >>> print(trace_distance(u, v, 0, 1))
        ...
        ValueError: The vector is not normalized!
        
    """
    _x=1
    _y=1
    if _check==1:
        _x=vector_check(_uvector)
        _y=vector_check(_vvector)
    if (_x==1 and _y==1):
        if _r==0:
            _rslt = np.sqrt( 1.0 - ( np.linalg.norm( np.vdot( _vvector, _uvector ) ) ** 2.0 ) )
        else:
            _rslt = round( np.sqrt( 1.0 - ( np.linalg.norm( np.vdot( _vvector, _uvector ) ) ** 2.0 ) ), _r )
        return _rslt
    else:
        return None

def trace_distance_density_matrix( _rho, _sigma):
   
    _val = 0.0
    
    _diff_dm =  _rho - _sigma
    
    _evals = np.linalg.eig( _diff_dm )
    
    for idx in range(0, len(_evals[0]) ):
        _val = _val + np.abs(_evals[0][idx])
    
    _val = 0.5 * _val;
    
    return _val

def probability_as_distance(uvector, vvector, amp_no, r=0, check=0 ):
    """
    Calculates the distance based on pointed aplitude probability for 
    two pure states.

    Parameters
    ----------
    uvector, vvector : numpy array objects
        Vectors of complex numbers describing quantum states.
    amp_no : integer
        The number of amplitude (starting from zero) for which the distance 
        should be calculated.
    r : integer
        The number of decimals to use while rounding the number (default is 0,
        i.e. the number is not rounded).
    check : Boolean
        If check==1 then paramatres uvector, vvector are checked for being 
        normalized vectors (as default it is not checked).

    Returns
    -------
    rslt : float
        The distance between given quantum states as distance between pointed 
        aplitudes.
    Examples
    --------
    A distance between exemplary states:
    >>> v=np.array([1,0,0,0])
    >>> u=np.array([0,0,1,0])
    >>> print(probability_as_distance(u, v, 0))
        1.0
    A distance between exemplary states:
    >>> v=np.array([1,0,0,0])
    >>> u=np.array([0,0,1,0])
    >>> print(probability_as_distance(u, v, 1))
        0.0
    A distance between states of different sizes:
    >>> v=np.array([1,0,0,0])
    >>> u=np.array([0,0,1])
    >>> print(probability_as_distance(u, v, 1))
        ...
        ValueError: The given vectors heve different dimensions!
    If entered vector v is not a correct quantum state:
    >>> v=np.array([1,1])
    >>> u=np.array([1,0])
    >>> print(probability_as_distance(u, v, 1, 0, 1))
        ...
        ValueError: The vector is not normalized!
        
    """
    if len(uvector)!=len(vvector):
        raise ValueError("The given vectors heve different dimensions!")
        return None
    else:
        x=1
        y=1
        if check==1:
            x=vector_check(uvector)
            y=vector_check(vvector)
        if (x==1 and y==1):
            rslt = np.abs( (np.linalg.norm(uvector[amp_no])**2) - (np.linalg.norm(vvector[amp_no])**2) )
            if r==0:
                return float(rslt)
            else:
                return round(float(rslt),r)
        else:
            return None

def probability_as_distance_case_qubit_alpha(uvector, vvector, r=0, check=0):
    return probability_as_distance(uvector, vvector, 0, r, check)

def probability_as_distance_case_qubit_beta(uvector, vvector, r=0, check=0):
    return probability_as_distance(uvector, vvector, 1, r, check)


def probability_as_distance_all(uvector, vvector, r=0, check=0 ):
    """
    Calculates the distance based on probability amplitudes for 
    two pure states.

    Parameters
    ----------
    uvector, vvector : numpy array objects
        Vectors of complex numbers describing quantum states.
    r : integer
        The number of decimals to use while rounding the number (default is 0,
        i.e. the number is not rounded).
    check : Boolean
        If check==1 then paramatres uvector, vvector are checked for being 
        normalized vectors (as default it is not checked).

    Returns
    -------
    rslt : float
        The distance between given quantum states as distance between values 
        of all probability aplitudes.
    Examples
    --------
    A distance between exemplary states:
    >>> v=np.array([1,0,0,0])
    >>> u=np.array([0,0,1,0])
    >>> print(probability_as_distance_all(u, v))
        0.5
    A distance between exemplary states:
    >>> v=np.array([[1/math.sqrt(5),1/math.sqrt(5),1/math.sqrt(5),math.sqrt(3)/math.sqrt(5)])
    >>> u=np.array([0,0,0,1])
    >>> print(probability_as_distance_all(u, v))
        0.25
    A distance between states of different sizes:
    >>> v=np.array([1,0,0,0])
    >>> u=np.array([0,0,1])
    >>> print(probability_as_distance_all(u, v))
        ...
        ValueError: The given vectors heve different dimensions!
    If entered vector v is not a correct quantum state:
    >>> v=np.array([1,1])
    >>> u=np.array([1,0])
    >>> print(probability_as_distance_all(u, v, 0, 1))
        ...
        ValueError: The vector is not normalized!
        
    """
    if len(uvector)!=len(vvector):
        raise ValueError("The given vectors heve different dimensions!")
        return None
    else:
        x=1
        y=1
        if check==1:
            x=vector_check(uvector)
            y=vector_check(vvector)
        if (x==1 and y==1):
            rslt=0.0
            for i in range(len(uvector)):
                rslt += np.abs( (np.linalg.norm(uvector[i])**2) - (np.linalg.norm(vvector[i])**2) )
            rslt /= len(uvector)
            if r==0:
                return float(rslt)
            else:
                return round(float(rslt),r)
        else:
            return None

def swap_test_as_distance_p0(uvector, vvector, r=0, check=0):
    rslt = (0.5 + 0.5 * np.linalg.norm( (uvector @ vvector.T) ) ** 2)
    return float(1.0 - rslt)

def swap_test_as_distance_p1(uvector, vvector, r=0, check=0):
    rslt = (1.0 - swap_test_as_distance_p0(uvector, vvector, r, check))
    return float(1.0 - rslt)


def euclidean_distance_without_sqrt(uvector, vvector, r=0, check=0):
    rslt = np.sum( ( np.abs( (uvector - vvector) ) ) ** 2.0 )
    
    return rslt

def euclidean_distance_with_sqrt(uvector, vvector, r=0, check=0):
    rslt = np.sum( ( np.abs( (uvector - vvector) ) ) ** 2.0 )
    
    return np.sqrt(rslt)
    

def create_zero_vector( _n_dim=3 ):
    """
    
    Parameters
    ----------
    _n_dim : TYPE, optional
        DESCRIPTION. The default is 3.

    Returns
    -------
    _vector_zero : TYPE
        DESCRIPTION.

    Examples
    --------
    """
    
    _vector_zero = np.zeros( (_n_dim), dtype=np.complex )
    
    return _vector_zero

def create_one_vector( _axis=0, _n_dim=3 ):
    """

    Parameters
    ----------
    _axis : TYPE, optional
        DESCRIPTION. The default is 0.
    _n_dim : TYPE, optional
        DESCRIPTION. The default is 3.

    Returns
    -------
    _vector_one : TYPE
        DESCRIPTION.

    """
    
    _vector_one = np.zeros( (_n_dim), dtype=np.complex )
    _vector_one[ _axis ] = 1.0  
    
    return _vector_one


def create_spherical_probes( _n_points, _n_dim=2):
    """
    
    Parameters
    ----------
    _n_points : TYPE
        DESCRIPTION.
    _n_dim : TYPE, optional
        DESCRIPTION. The default is 3.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    if _n_points > 0:
        _unit_vectors = np.random.randn( _n_dim, _n_points )
        _unit_vectors /= np.linalg.norm( _unit_vectors, axis=0 )
    else:
        raise ValueError("The number of points must be positive integer number!")
        return None
    
    return _unit_vectors.T

def create_focused_circle_probes( _n_points, _n_focus_points, _width_of_cluster=0.25 ):
    
    d, _ = make_blobs( n_samples=_n_points,
                       n_features=2,
                       centers = _n_focus_points,
                       cluster_std=_width_of_cluster )

    for i in range(_n_points):
        d[i] = d[i] / np.linalg.norm(d[i])
    
    return d

def create_focused_circle_probes_with_uniform_placed_centers( _n_points, _n_focus_points, _width_of_cluster=0.1 ):
    
    theta=0.0
    theta_delta = (2.0 * np.pi) / _n_focus_points
    centers_on_circle = [ ]
    
    for _ in range(_n_focus_points):
        theta = theta + theta_delta
        x = np.sin(theta)
        y = np.cos(theta)  
        centers_on_circle.append((x,y))
    
    d, _ = make_blobs( n_samples=_n_points, 
                            n_features=2, 
                            centers=centers_on_circle, 
                            cluster_std=_width_of_cluster )
    
    for i in range(_n_points):
        d[i] = d[i] / np.linalg.norm(d[i])
    
    return d

def create_focused_qubits_probes( _n_points, _n_focus_points, _width_of_cluster=0.25 ):
    
    d, _ = make_blobs( n_samples=_n_points,
                       n_features=3,
                       centers = _n_focus_points,
                       cluster_std=_width_of_cluster )

    for i in range(_n_points):
        d[i] = d[i] / np.linalg.norm(d[i])
    
    return d

def create_focused_qubits_probes_with_uniform_placed_centers( _n_points, _n_theta, _n_psi, _width_of_cluster=0.1, _return_labels = False, _return_centers = False ):
    
    centers_on_sphere = [ ]
    
    _theta = 0.0
    _psi = 0.0
    _theta_delta= (2.0*np.pi) / _n_theta
    _psi_delta= (np.pi) / _n_psi
    for i in range( _n_theta ):
        for j in range( _n_psi ):
            sp = convert_spherical_point_to_bloch_vector(1.0, _theta, _psi)
            centers_on_sphere.append( ( sp[0], sp[1], sp[2]) ) 
            _theta = _theta +_theta_delta
            _psi = _psi + _psi_delta

    convert_spherical_point_to_bloch_vector

    d, labels = make_blobs( n_samples=_n_points,
                       n_features=3,
                       centers = centers_on_sphere,
                       cluster_std=_width_of_cluster )

    for i in range(_n_points):
        d[i] = d[i] / np.linalg.norm(d[i])
    
    
    if _return_labels==True and _return_centers==False:
        return d, labels

    if _return_labels==True and _return_centers==True:
        return d, labels, centers_on_sphere
    
    return d

def slerp(p0, p1, t):
    """

    Parameters
    ----------
    p0 : TYPE
        DESCRIPTION.
    p1 : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # p0,p1 to wektory to można sprawdzić
    # czy są tych samych wymiarów,
    # a jak nie to wyjątkiem ;-) DimensionError, podobnie jak w EntDetectorze
    omega = np.arccos(np.dot(p0/np.linalg.norm(p0), p1/np.linalg.norm(p1)))
    so = np.sin(omega)
    
    return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1

def kmeans_spherical(_X, _n_clusters, _max_iteration=128, _func_distance=None):
    """
    
    Parameters
    ----------
    _X : TYPE
        DESCRIPTION.
    _n_clusters : TYPE
        DESCRIPTION.
    _max_iteration : TYPE, optional
        DESCRIPTION. The default is 128.

    Returns
    -------
    closest : TYPE
        DESCRIPTION.
    centers : TYPE
        DESCRIPTION.

    """

    if _func_distance == None:
        raise ArgumentValueError("The _func_distance argment is None, a distance function shoud be provided.")
        return None

    _n_probes = _X.shape[0]
    _distances = np.zeros( (_n_probes, _n_clusters) )
    centers = _X[np.random.choice(_n_probes, _n_clusters, replace=False)]
    closest = np.argmin(_distances, axis=1)

    _iteration=0
    
    while _iteration < _max_iteration:
        old_closest = closest
        
        for idx in range(_n_probes):
            for ncnt in range(_n_clusters):
                _distances[ idx, ncnt ] = _func_distance( _X[idx], centers[ncnt] )

        closest = np.argmin(_distances, axis=1)
        
        for i in range(_n_clusters):
            # fix required for other function distances 
            centers[i,:] = _X[closest == i].mean(axis=0)
            centers[i,:] = centers[i,:] / np.linalg.norm(centers[i,:])
        
        if all(closest == old_closest):
            break
        
        _iteration = _iteration + 1
    return closest, centers 

def kmeans_quantum_states(_qX, _n_clusters, _func_distance=COSINE_DISTANCE, _max_iterations=128, _verification=0):
    """
    
    Parameters
    ----------
    qX : TYPE
        DESCRIPTION.
    _n_clusters : TYPE
        DESCRIPTION.

    Returns
    -------
    closest : TYPE
        DESCRIPTION.
    centers : TYPE
        DESCRIPTION.

    """

    # vectors qX should be treated as quantum pure states
    # but verification in performed when 
    # verification == 1

    if _func_distance==COSINE_DISTANCE:
        _funcdist = cosine_distance

    if _func_distance==DOT_DISTANCE:
        _funcdist = dot_product_as_distance

    if _func_distance==FIDELITY_DISTANCE:
        _funcdist = fidelity_as_distance

    if _func_distance==TRACE_DISTANCE:
        _funcdist = trace_distance_vector

    if _func_distance==MANHATTAN_DISTANCE:
        _funcdist = manhattan_distance
    
    if _func_distance==BURES_DISTANCE:
        _funcdist =bures_distance
    
    if _func_distance==HS_DISTANCE:
        _funcdist = hs_distance 

    if _func_distance==P_CQA_DISTANCE:
        _funcdist = probability_as_distance_case_qubit_alpha

    if _func_distance==P_CQB_DISTANCE:
        _funcdist = probability_as_distance_case_qubit_beta

    if _func_distance == SWAP_TEST_DISTANCE:
        _funcdist = swap_test_as_distance_p0
    
    closest, centers = kmeans_spherical( _qX, _n_clusters, _max_iterations, _funcdist )
        
    return closest, centers 

def kmedoids_calculate_costs(_qX, _medoids, _func_distance = None):
    """
    

    Parameters
    ----------
    _qX : TYPE
        DESCRIPTION.
    _medoids : TYPE
        DESCRIPTION.
    _func_distance : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    clusters : TYPE
        DESCRIPTION.
    total_cost_for_clusters : TYPE
        DESCRIPTION.

    """
    
    clusters = {i:[] for i in range(len(_medoids))}
    total_cost_for_clusters = 0

    for probe in _qX:
        distances = np.array( [_func_distance(probe, single_center) for single_center in _medoids] )
        closest_medoid = distances.argmin()
        clusters[closest_medoid].append(probe)
        total_cost_for_clusters = total_cost_for_clusters + distances.min()
 
    clusters = {k:np.array(v) for k,v in clusters.items()}
    return clusters, total_cost_for_clusters


def kmedoids(_qX, _n_clusters, _max_iterations=128, _func_distance = None):
    """
    

    Parameters
    ----------
    _qX : TYPE
        DESCRIPTION.
    _n_clusters : TYPE
        DESCRIPTION.
    _max_iterations : TYPE, optional
        DESCRIPTION. The default is 128.
    _func_distance : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    labels : TYPE
        DESCRIPTION.
    medoids : TYPE
        DESCRIPTION.

    """
    _iteration=0
    isSwapPerformed = False

    medoids = np.array([_qX[i] for i in range(_n_clusters)])
    samples = _qX.shape[0]
    
    clusters, current_cost = kmedoids_calculate_costs(_qX, medoids, _func_distance)
    
    while (isSwapPerformed == False) and (_iteration < _max_iterations):
        isSwapPerformed = False
        for i in range(samples):
            if not i in medoids:
                for j in range(_n_clusters):
                    medoids_tmp = medoids.copy()
                    medoids_tmp[j] = _qX[i]
                    current_clusters, cost_tmp = kmedoids_calculate_costs(_qX, medoids_tmp, _func_distance)
    
                    if cost_tmp < current_cost:
                        medoids = medoids_tmp
                        current_cost = cost_tmp
                        isSwapPerformed = True
                        
        _iteration = _iteration + 1
    
    # create labels
    labels=np.zeros(samples, dtype=np.int64)
    tmpdist=np.zeros(_n_clusters)
    for idx in range(0, samples):
        labels[idx] = -1
        ridx=0
        for v in medoids:
            tmpdist[ridx]=_func_distance(_qX[idx], v)
            ridx=ridx+1
        labels[idx] = np.int64(tmpdist.argmin())
            

    return labels, medoids

def kmedoids_quantum_states(_qX, _n_clusters, _func_distance=COSINE_DISTANCE, _max_iterations=128, _verification=0):
    # vectors qX should be treated as quantum pure states
    # but verification in performed when 
    # verification == 1

    if _func_distance==COSINE_DISTANCE:
        _funcdist = cosine_distance

    if _func_distance==DOT_DISTANCE:
        _funcdist = dot_product_as_distance

    if _func_distance==FIDELITY_DISTANCE:
        _funcdist = fidelity_as_distance

    if _func_distance==TRACE_DISTANCE:
        _funcdist = trace_distance_vector

    if _func_distance==MANHATTAN_DISTANCE:
        _funcdist = manhattan_distance
    
    if _func_distance==BURES_DISTANCE:
        _funcdist =bures_distance
    
    if _func_distance==HS_DISTANCE:
        _funcdist = hs_distance 
    
    if _func_distance==P_CQA_DISTANCE:
        _funcdist = probability_as_distance_case_qubit_alpha

    if _func_distance==P_CQB_DISTANCE:
        _funcdist = probability_as_distance_case_qubit_beta

    if _func_distance == SWAP_TEST_DISTANCE:
        _funcdist = swap_test_as_distance_p0

    closest, centers = kmedoids( _qX, _n_clusters, _max_iterations, _funcdist )
        
    return closest, centers 

def calculate_distance(_data, _vector, _func_distance):
    """
    

    Parameters
    ----------
    _data : TYPE
        DESCRIPTION.
    _vector : TYPE
        DESCRIPTION.
    _func_distance : TYPE
        DESCRIPTION.

    Returns
    -------
    distance_table : TYPE
        DESCRIPTION.

    """
    distance_table=np.zeros( shape=(_data.shape[0] ) )
    idx=0
    for e in _data:
        distance_table[idx] = _func_distance(e, _vector)
        #distance_table[idx, 1] = l
        idx=idx+1
    
    return distance_table
    

def create_distance_table( _data, _centers, _labels, _n_clusters, _func_distance=None ):
    """
    

    Parameters
    ----------
    _data : TYPE
        DESCRIPTION.
    _centers : TYPE
        DESCRIPTION.
    _labels : TYPE
        DESCRIPTION.
    _n_clusters : TYPE
        DESCRIPTION.
    _func_distance : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    distance_table : TYPE
        DESCRIPTION.

    """
    idx=0
    distance_table=np.zeros( shape=(_data.shape[0], 2) )
    for l in range(0, _n_clusters):
        cntr=_centers[l]
        for e in _data[_labels == l]:
            distance_table[idx, 0] = _func_distance(e, cntr)
            distance_table[idx, 1] = l
            idx=idx+1
    
    return distance_table

def get_distances_for_cluster( _data, _n_cluster ):
    """
    

    Parameters
    ----------
    _data : TYPE
        DESCRIPTION.
    _n_cluster : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return _data[ _data[:, 1] == _n_cluster ]

def get_data_for_class(_data, _labels, _class):
    return _data[ _labels==_class ]

def get_min_label_class(_labels):
    return np.min( _labels )

def get_max_label_class(_labels):
    return np.max( _labels )

def true_positive_rate(TP, FN):
    """
    Calculates True Positive Rate for a classification task (supervised 
    learning).

    Parameters
    ----------
    TP : integer
        The number of observations corretly clasiffied to the l-th class.
    FN : integer
        The number of observations belonging to the l-th class but 
        incorrectly classiﬁed as not in l.

    Returns
    -------
    float
        The True Positive Rate.

    """
    return TP/(TP+FN)



def true_negative_rate(TN, FP):
    """
    Calculates True Negative Rate for a classification task (supervised 
    learning).

    Parameters
    ----------
    TN : integer
        The number of observations from a class different than l and correctly 
        classiﬁed as not in l.
    FP : integer
        The number of observations belonging to a class different than l and 
        incorrectly classiﬁed as l.

    Returns
    -------
    float
        The True Negative Rate.

    """
    return TN/(TN+FP)



def false_positive_rate(FP, TN):
    """
    Calculates False Positive Rate for a classification task (supervised 
    learning).

    Parameters
    ----------
    FP : integer
        The number of observations belonging to a class different than l and 
        incorrectly classiﬁed as l.
    TN : integer
        The number of observations from a class different than l and correctly 
        classiﬁed as not in l.

    Returns
    -------
    float
        The False Positive Rate.

    """
    return FP/(FP+TN)


def false_negative_rate(FN, TP):
    """
    Calculates False Negative Rate for a classification task (supervised 
    learning).

    Parameters
    ----------
    FN : integer
        The number of observations belonging to the l-th class but 
        incorrectly classiﬁed as not in l.
    TP : integer
        The number of observations corretly clasiffied to the l-th class.

    Returns
    -------
    float
        The False Negative Rate.

    """
    return FN/(FN+TP)


def classification_error(TP, STS):
    """
    Calculates the Classication Error (supervised learning).

    Parameters
    ----------
    TP : integer
        The number of observations corretly clasiffied to the l-th class.
    STS : integer
        The number of observations in the test set.

    Returns
    -------
    float
        The percentage of misclassiﬁed observations for the l-th class.

    """
    return 1.0-(TP/STS)

def precision(TP, FP):
    """
    Calculates a Precision of the classification (supervised learning).

    Parameters
    ----------
    TP : integer
        The number of observations corretly clasiffied to the l-th class.
    FP : integer
        The number of observations belonging to a class different than l and 
        incorrectly classiﬁed as l.

    Returns
    -------
    float
        The statistical variability in the l-th class.

    """
    return TP/(TP+FP)

def cohens_kappa(TP, TN, FP, FN, STS):
    """
    Calculates the Cohen's Kappa (supervised learning) which shows the degree 
    of reliability and accuracy of a statistical classiﬁcation.

    Parameters
    ----------
    TP : integer
        The number of observations corretly clasiffied to the l-th class.
    TN : integer
        The number of observations from a class different than l and correctly 
        classiﬁed as not in l.
    FP : integer
        The number of observations belonging to a class different than l and 
        incorrectly classiﬁed as l.
    FN : integer
        The number of observations belonging to the l-th class but 
        incorrectly classiﬁed as not in l.
    STS : integer
        The number of observations in the test set.

    Returns
    -------
    float
        A value of the Cohen's Kappa [-1,1].

    """
    if (TP+TN+FP+FN)!=STS:
        raise ValueError("The sum of TP, TN, FP, FN and the number of elements in the test set should be equal!")
        return None
    else:
        pra=(TP+TN)/STS
        pre=((TP+FP)*(TP+FN)+(FP+TN)*(TN+FN))/(STS*STS)
        return (pra-pre)/(1-pre)


def difference_matrix(rho1, rho2):
    
    # dimension check for rho1, rho2
    # rows,cols = rho1.shape
    
    result_rho = np.zeros(rho1.shape)
    
    result_rho = rho1 - rho2
    
    return result_rho

def create_covariance_matrix( _qX ):  
    
    _n_samples = np.shape(_qX)[0]
    
    scale = (1.0 / (_n_samples - 1.0))
    
    _qXDiffWithMean = _qX - _qX.mean(axis=0)
    
    covariance_matrix = scale * (( _qXDiffWithMean ).T.dot( _qXDiffWithMean ))

    return np.array(covariance_matrix, dtype=complex)
        
def version():
    pass

def about():
    pass

def how_to_cite():
    pass
