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
from sklearn.cluster import KMeans


import numpy as np
import pandas as pd
import math as math
import sympy as sympy

import cvxopt
import scipy

import datasets



#
# Quantum Computing Simulator (QCS)
#

# import qcs



CUSTOM_DISTANCE    =  999
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

LINEAR_KERNEL     = 4000
POLYNOMIAL_KERNEL = 4001
GAUSSIAN_KERNEL   = 4002

OPT_COBYLA = 5000
OPT_SPSA   = 5001
OPT_SLSQP  = 5002
OPT_POWELL = 5003

QDCL_SEED = 1234

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


# code based on chop
# discussed at:
#   https://stackoverflow.com/questions/43751591/does-python-have-a-similar-function-of-chop-in-mathematica
def _internal_chop(expr, delta=10 ** -10):
    if isinstance(expr, (int, float)):
        return 0 if -delta <= expr <= delta else expr
    
    if isinstance(expr, complex):
        realpart  = expr.real
        impart = expr.imag
        realpart  = 0 if -delta <= realpart <= delta else realpart
        impart =  0 if -delta <= impart <= delta else impart
               
        return complex(realpart, impart)
        
    return [_internal_chop(x) for x in expr]

chop = _internal_chop


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
    """
    Calculates a centroid for the quantum data. The input data table contains
    normalized quantum states. For each state, a density matrix is calculated and
    the centroid is calculated as a sum of these matrices. Finally, the centroid
    values are averaged by the value of the second function's argument.

    Parameters
    ----------
    _qX : numpy ndarray
        Data table containing a normalized quantum state in each row.
    _n_elems_in_class : integer
        Number of elements to average the result centroid array. The default 
        value (-1) means that the averaging is performed for all observations
        in _qX array.

    Returns
    -------
    centroid : numpy ndarray
        A centroid matrix.
        
    Examples
    --------
    A centriod matrix for an exemplary data set (four variables utilized to 
    calculate 2-qubit states) with everaging by the number of all observations:
    >>> df = pd.read_excel(r'CRABS.xlsx')
    >>> data_tab=qdcl.convert_data_to_vector_states(df,4)
    >>> print(qdcl.create_quantum_centroid(data_tab))
        [[0.16072472 0.14569546 0.23059843 0.24570628]
         [0.14569546 0.1323407  0.2090742  0.22284072]
         [0.23059843 0.2090742  0.33100732 0.35273057]
         [0.24570628 0.22284072 0.35273057 0.37592726]]
    A centriod matrix for an exemplary data set (four variables utilized to 
    calculate 2-qubit states) with everaging by 3:
    >>> df = pd.read_excel(r'CRABS.xlsx')
    >>> data_tab=qdcl.convert_data_to_vector_states(df,4)
    >>> print(qdcl.create_quantum_centroid(data_tab,3))
        [[10.7149813   9.71303089 15.37322885 16.38041858]
         [ 9.71303089  8.82271323 13.93827986 14.85604802]
         [15.37322885 13.93827986 22.0671549  23.51537111]
         [16.38041858 14.85604802 23.51537111 25.06181724]]

    """
    rows, cols = _qX.shape
    
    centroid=np.zeros(shape=(cols,cols))
    
    for idx in range(0, rows):
        centroid = centroid + _internal_qdcl_create_density_matrix_from_vector_state( _qX[ idx, : ] )
    
    if _n_elems_in_class==-1:
        centroid = centroid * (1.0/ float(rows))
    else:
        centroid = centroid * (1.0/ float(_n_elems_in_class))
        
    return centroid

def chop_and_round_for_array(_expr, _delta=10 ** -10):
    """
    Rounds the numbers in the array and also values convergent to zero converts 
    to zero. 

    Parameters
    ----------
    _expr : numpy array
        A two-dimensional array.
    _delta : float
        The scale of rounding and chopping. The default value is 10 ** -10.

    Returns
    -------
    expr : numpy array
        A two-dimensional array.
        
    Examples
    --------
    >>> x=np.array([[1.9876,0.0000001],[100.725,0]] )
    >>> print(qdcl.chop_and_round_for_array(x))
        [[1.98760e+00 1.00000e-07]
         [1.00725e+02 0.00000e+00]]
    >>> print(qdcl.chop_and_round_for_array(x,0.1))
        [[  2.   0.]
         [101.   0.]]
    >>> print(qdcl.chop_and_round_for_array(x,0.01))
        [[  2.    0. ]
         [100.7   0. ]]
        
    """
    expr=_expr.copy()
    
    for i in range( expr.shape[0] ):
        for j in range( expr.shape[1] ):
            expr[i,j] = round( chop(expr[i,j]), int(-np.log(_delta)/np.log(10)) )
            
    return expr

def convert_qubit_pure_state_to_bloch_vector( qstate ):
    """
    Converts a pure 1-qubit state, given as a vector or a density matrix, 
    to the Bloch vector representation. 

    Parameters
    ----------
    qstate : numpy array
        A two-element vector or a matrix sized 2x2 representing one qubit state.

    Returns
    -------
    numpy array
        The Bloch vector representation of a given 1-qubit state.
    Examples
    --------
    >>> print(qdcl.convert_qubit_pure_state_to_bloch_vector( np.array( [1,0] )))
        [0. 0. 1.]
    >>> print(qdcl.convert_qubit_pure_state_to_bloch_vector( np.array( [1/np.sqrt(2),-1/np.sqrt(2)] )))
        [-1.  0.  0.]
    >>> print(qdcl.convert_qubit_pure_state_to_bloch_vector( np.array [1/np.sqrt(2),0+1j/np.sqrt(2)] )))
        [0. 1.  0.]
    >>> print(qdcl.convert_qubit_pure_state_to_bloch_vector( np.array( [[1,0],[0,0]] )))
        [0. 0. 1.]
    >>> print(qdcl.convert_qubit_pure_state_to_bloch_vector( np.array( [1,0,0] )))
        ...
        ValueError: Incorrect size of 1-qubit state!
    """
    
    # check if qstate is vector state or a density matrix
    _x=qstate.ndim
    if _x==1:
        b=qstate.shape[0]
        a=1
    elif _x==2:
        a,b=qstate.shape
    else:
        raise ValueError("Incorrect size of 1-qubit state!")
        return None
        
    if a==1 and b==2:
        qstateden = _internal_qdcl_vector_state_to_density_matrix( qstate )
    elif a==2 and b==2:
        qstateden = qstate 
    else:
        raise ValueError("Incorrect size of 1-qubit state!")
        return None
    
    xcoord = np.trace( _internal_pauli_x() @ qstateden )
    ycoord = np.trace( _internal_pauli_y() @ qstateden )
    zcoord = np.trace( _internal_pauli_z() @ qstateden )
    
    return np.array([ np.real(xcoord), np.real(ycoord), np.real(zcoord) ])

def convert_spherical_coordinates_to_bloch_vector( _r, _theta, _phi, _round=0 ):
    """
    Converts spherical coordinates of r radius, theta, and phi angles to
    a Bloch vector (only if r==1, we refer to normalized quantum state).

    Parameters
    ----------
    _r : float
        The distance between the point and the sphere center (radius).
    _theta : float
        An angle in radians describing the tilt with respect to the z-axis 
        on the Bloch sphere (0 <= _theta <= np.pi).
    _phi : float
        An angle in radians describing the tilt with respect to the x-axis 
        on the Bloch sphere (0 <= _phi <= 2.0 * np.pi).
    _round : integer
        The number of decimals to use while rounding the number (default is 0,
        i.e. the number is not rounded).

    Returns
    -------
    numpy ndarray
        A Bloch vector for a given spherical coordinates.
        
    Examples
    --------
    >>> print(qdcl.convert_spherical_coordinates_to_bloch_vector( 1, np.pi/2, np.pi, 2 ))
        [-1. 0.  0.]
    >>> print(qdcl.convert_spherical_coordinates_to_bloch_vector( 1, np.pi/2, np.pi/2, 4 ))
        [0. 1. 0.]

    """
    xcoord = _r * np.sin( _theta ) * np.cos( _phi )
    ycoord = _r * np.sin( _theta ) * np.sin( _phi )
    zcoord = _r * np.cos( _theta )
    
    if _round==0:
        return np.array([xcoord, ycoord, zcoord])
    else:
        return np.array([np.round(xcoord, _round), np.round(ycoord, _round), np.round(zcoord, _round)])

def convert_bloch_vector_to_spherical_coordinates( _x, _y, _z ):
    """
    Converts a Bloch vector to a spherical coordinates.

    Parameters
    ----------
    _x : float
        The coordinate of the Bloch vector with respect to the x-axis.
    _y : float
        The coordinate of the Bloch vector with respect to the y-axis.
    _z : float
        The coordinate of the Bloch vector with respect to the z-axis.

    Returns
    -------
    numpy ndarray
        A data of the spherical point: radius, theta, and phi angles.
        
    Examples
    --------
    >>> ...
    """
    
    if _x==0.0 and _y==0.0 and _z==0.0:
        raise ArgumentValueError("Zero values for _x, _y, _z arguments are not allowed!")
    
    if _x==0.0 and _y==0.0:
        r=np.sqrt( _z * _z )
        theta = np.arccos( _z / r )
        phi = np.arccos( 0.0 ) 
    else:
        r = np.sqrt( _x * _x + _y * _y + _z * _z )
        theta = np.arccos( _z / r )
        phi = np.sign(_y) *  np.arccos( _x / np.sqrt(_x*_x + _y*_y) ) 
    
    return np.array([r, theta, phi ])

def convert_spherical_coordinates_to_pure_state( _theta, _phi, _round=0):
    """
    Converts spherical coordinates of theta and phi angles to
    a pure quantum state.

    Parameters
    ----------
    _theta : float
        An angle in radians describing the tilt with respect to the z-axis 
        on the Bloch sphere (0 <= _theta <= np.pi).
    _phi : float
        An angle in radians describing the tilt with respect to the x-axis 
        on the Bloch sphere (0 <= _phi <= 2.0 * np.pi).
    _round : integer
        The number of decimals to use while rounding the number (default is 0,
        i.e. the number is not rounded).

    Returns
    -------
    pure_state_qubit : numpy vector
        The 1-qubit pure state vector.
        
    Examples
    --------
    >>> print( qdcl.convert_spherical_coordinates_to_pure_state( 0, 0) )
        [1.+0.j 0.+0.j]
    >>> print(qdcl.convert_spherical_coordinates_to_pure_state( np.pi, 0, 7))
        [0.+0.j 1.+0.j]
    >>> print(qdcl.convert_spherical_coordinates_to_pure_state( np.pi/2, 0))
        [0.70710678+0.j 0.70710678+0.j]
    >>> print(qdcl.convert_spherical_coordinates_to_pure_state( np.pi/2, np.pi/2, 4))
        [0.7071+0.j     0.    +0.7071j]

    """
    pure_state_qubit = create_zero_vector( 2 )
    
    if _round==0:
        pure_state_qubit[0] = np.cos( _theta / 2.0 )
        pure_state_qubit[1] = np.exp(1.0J * _phi) * np.sin( _theta / 2.0 )
    else:
        pure_state_qubit[0] = np.round(np.cos( _theta / 2.0 ), _round)
        pure_state_qubit[1] = np.round(np.exp(1.0J * _phi) * np.sin( _theta / 2.0 ), _round)
    
    return pure_state_qubit

def convert_bloch_vector_to_pure_state( _x, _y, _z ):
    """
    Converts a Bloch vector to a pure vector.

    Parameters
    ----------
    _x : float
        The coordinate of the Bloch vector with respect to the x-axis.
    _y : float
        The coordinate of the Bloch vector with respect to the y-axis.
    _z : float
        The coordinate of the Bloch vector with respect to the z-axis.

    Returns
    -------
    pure_state_qubit : numpy vector
        The 1-qubit pure state vector.
        
    Examples
    --------
    >>> ...
    """
    r, theta, phi = convert_bloch_vector_to_spherical_coordinates(_x, _y, _z)
    pure_state_qubit = convert_spherical_coordinates_to_pure_state( theta, phi  )
    
    return pure_state_qubit


def stereographic_projection_to_two_component_vector( _x, _y, _z ):
    """
    The stereographic projection of a Bloch vector to two-element vector.

    Parameters
    ----------
    _x : float
        The coordinate of the Bloch vector with respect to the x-axis.
    _y : float
        The coordinate of the Bloch vector with respect to the y-axis.
    _z : float
        The coordinate of the Bloch vector with respect to the z-axis.

    Returns
    -------
    two_component_vector : numpy vector
        The 1-qubit pure state vector.
        
    Examples
    --------
    >>> ...
    """
    two_component_vector = create_zero_vector( 2 )
    
    two_component_vector[0] = _x / (1.0 - _z)
    two_component_vector[1] = _y / (1.0 - _z)
    
    return two_component_vector
    
#
# TO DESC
#
def vector_data_encode_with_inverse_stereographic_projection( _v ):
    d = _v.shape[0]
    
    rsltvec = np.zeros( shape=(d+1,) )
    normv = np.linalg.norm( _v )

    for idx in range(d): 
        rsltvec[ idx ] = _v[ idx ] / (normv * np.sqrt( (normv ** 2) + 1.0))

    rsltvec[d]=(normv / np.sqrt( (normv ** 2) + 1.0))

    return rsltvec

def encode_probe_by_normalization( _qdX ):
    """
    The simple normalization of a given vector.

    Parameters
    ----------
    _qdX : numpy ndarray
        The data vector with an arbitrary number of elements.
    
    Returns
    -------
    x : numpy ndarray
        A normalized vector.
        
    Examples
    --------
    >>> print(qdcl.encode_probe_by_normalization( np.array([1/2,0] ) ))
        [1. 0.]
    >>> print(qdcl.encode_probe_by_normalization( np.array([1/2,1/3,1/4] ) ))
        [0.76822128 0.51214752 0.38411064]
    >>> print(qdcl.encode_probe_by_normalization( np.array([1/2,1/2,0,1/2] ) ))
        [0.57735027 0.57735027 0.         0.57735027]
        
    """
    nrm = np.linalg.norm( _qdX )
    _n_features = _qdX.shape[0]
    x = np.zeros( shape=(_n_features,)  )
    for k in range(_n_features):
        x[k] = (1.0/nrm) * _qdX[k]
        
    return x

def encode_probes_by_normalization( _qdX ):
    """
    The simple normalization in each row of the given array.

    Parameters
    ----------
    _qdX : numpy ndarray
        The data table with an arbitrary number of rows (the number of probes) 
        and columns (the number of variables).
    
    Returns
    -------
    _qdX : numpy ndarray
        An array of normalized row vectors.
        
    Examples
    --------
    >>> print(qdcl.encode_probes_by_normalization( np.array([[1/2,0], [1/4,1/3], [4,3], [1,8]]) ) )
        [[1.         0.        ]
         [0.6        0.8       ]
         [0.8        0.6       ]
         [0.12403473 0.99227788]]
        
    """
    for idx in range( _qdX.shape[0]):
        _qdX[idx] = encode_probe_by_normalization(_qdX[idx])
    
    return _qdX

#
# TO DESC
#
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


#
# TO DESC
#
def linear_kernel(x0, x1):
    v = np.dot(x0, x1)
    
    return v

#
# TO DESC
#
def polynomial_kernel( x0, x1, _const_val=1.0, _poly_degree=3):
    v = (np.dot(x0, x1) + _const_val) ** _poly_degree
    
    return v

#
# TO DESC
#
def gaussian_kernel( x0, x1, _sigma=0.5):
    v = np.exp( -_sigma * np.linalg.norm(x0 - x1) ** 2.0 )
    
    return v

#
# TO DESC
#
def create_kernel_matrix_for_training_data( _qdX, _sigma, _n_samples ):
    
    K = np.zeros( (_n_samples + 1, _n_samples + 1), dtype=complex )
    sigmaI = np.multiply( 1.0/_sigma, np.eye(_n_samples) )
    
    gram_matrix = np.zeros( (_n_samples, _n_samples), dtype=complex )
    
    for i in range( _n_samples ):
        for j in range( _n_samples ):
            gram_matrix[i,j] = np.dot( _qdX[i], _qdX[j] )

    K[0 , 1:] = 1.0
    K[1:, 0 ] = 1.0
    K[1:, 1:]  = gram_matrix + sigmaI
    
    return K

#
# TO DESC
#
def create_right_b_alpha_vector( _kernel_matrix, _labels, _n_samples ):
    
    tmpvec = np.zeros( (_n_samples + 1,), dtype=complex )
    tmpvec[1:] = _labels
    
    return np.linalg.inv(_kernel_matrix) @ tmpvec

#
# TO DESC
#
def create_b_c_and_alphas(_b_alpha_vector, _n_samples ):
    
    b = _b_alpha_vector[0]
    alphas=_b_alpha_vector[1:_n_samples+1]
    C = b*b + np.sum( _b_alpha_vector[1:_n_samples+1] * _b_alpha_vector[1:_n_samples+1] )
    
    return b, C, alphas

#
# TO DESC
#
def create_nu_coefficent(_qdX, _b, _alphas, _n_samples):
    vsum=0
    for idx in range(_n_samples):
        vx = _qdX[idx]
        va = _alphas[idx]
        vsum=vsum + (va ** 2.0) * (np.linalg.norm(vx) ** 2.0)
    
    return (_b ** 2.0) + vsum

#
# TO DESC
#
def create_nx_coefficent( _probe_x, _n_samples):
    return _n_samples * (np.linalg.norm( _probe_x ) ** 2.0) + 1

#
# TO DESC
#
def create_dot_ux_for_classification(_nu, _nx, _b, _alphas, _qdX, _probe_x, _n_samples):
    vsum=0
    norm_of_probe_x = np.linalg.norm( _probe_x )
    for idx in range(_n_samples):
        vx = _qdX[idx]
        va = _alphas[idx]
        vsum=vsum + va * np.linalg.norm(vx) * norm_of_probe_x * np.dot(vx, _probe_x)
    
    return ( 1.0/np.sqrt( _nx * _nu) ) * (_b + vsum)

# in preparation
#
# classic part is based on sklearn SVM classs
#  https://github.com/scikit-learn/scikit-learn/tree/main/sklearn/svm
#
# and another simple implementation
#   https://github.com/DrIanGregory/MachineLearning-SupportVectorMachines/
#

#
# TO DESC
#
class QuantumSVM:
    
    def __init__( self ):
        self.data_for_classification = [ ]
        self.data_labels = [ ] 
    
        self.q_data_for_classification = [ ]
        self.q_data_labels = [ ] 
        
        self._n_samples = -1
        self._n_features = -1
        
        self._sigma = 0.5
        self._degree = -1
        self._kernel = None
        self._kernel_type = LINEAR_KERNEL
        self._value_of_c = None
        self._show_progress = False
        self._absolute_tolerance = 1e-08
        self._relative_tolerance = 1e-08
        self._feasibility_tolerance = 1e-08
        self._alphas_tolerance  = 1e-5        
    
        self._bigN = -1
        self._nu = [ ]
        self._nx = [ ]
        self._b = 0
        self._C = 0
        self._alphas = None
        self._b_alpha_vector = None
        
        self.K = None
        
        self._quantum_kernel = None
        self._user_matrix_kernel_for_quantum_svm = False
    
    def reset( self ):
        pass

    def set_data(self, _qdX, _labels, _is_it_quantum=False):

        self.data_labels = _labels                
            
        self._n_samples = _qdX.shape[0]
        self._n_features = _qdX.shape[1]
        
        if _is_it_quantum == False:
            self.data_for_classification = _qdX
        else:
            self.data_for_classification = _qdX
            q_train_d = np.empty((0, self._n_features), dtype=complex)

            for d in _qdX:
                q=encode_probe_by_normalization( d )
                q_train_d = np.append(q_train_d, [[ q[0], q[1] ]], axis=0)

            self.q_data_for_classification = q_train_d
            self.q_data_labels = _labels
        
        self._bigN = int(2 ** round(np.log(self._n_samples)/np.log(2)))
        
    def prepare_quantum_objects(self):
        self.K = create_kernel_matrix_for_training_data( self.q_data_for_classification, 
                                                         self._sigma, 
                                                         self._n_samples )
        # Kinv=np.linalg.inv(K)
        # Id= qdcl.chop_and_round_for_array( Kinv @ K )
           
        self._b_alpha_vector = create_right_b_alpha_vector( self.K, 
                                                            self.q_data_labels, 
                                                            self._n_samples )
        
        self._b, self._C, self._alphas = create_b_c_and_alphas( self._b_alpha_vector, 
                                                                self._n_samples)
        
        # self._nu = create_nu_coefficent(self.data_for_classification, self._b, self._alphas, self._n_samples )
        # self._nx = create_nx_coefficent(self.data_for_classification[0], self._n_samples )
        
    def update_data_for_quantum_svm(self):
        self.prepare_quantum_objects()
        
    def set_kernel( self, _func_kernel ):
        self._kernel = _func_kernel

    def set_kernel_quantum_svm( self, _func_kernel ):
        self._quantum_kernel = _func_kernel
        self._user_matrix_kernel_for_quantum_svm = True

    def set_type_kernel( self, _t_kernel):
        self._kernel_type = _t_kernel

    
# implementation of this classical part 
# is inspired by:
#   https://github.com/DrIanGregory/MachineLearning-SupportVectorMachines/
#   https://github.com/BHARATHBN-123/MachineLearning-SupportVectorMachines
    def classic_fit( self ):
        
        gram_matrix = np.zeros( (self._n_samples, self._n_samples) )
        
        for i in range( self._n_samples ):
            for j in range( self._n_samples ):
                if self._kernel_type == LINEAR_KERNEL:
                    gram_matrix[i, j] = linear_kernel(self.data_for_classification[i], 
                                                      self.data_for_classification[j])
                if self._kernel_type == GAUSSIAN_KERNEL:
                    gram_matrix[i, j] = gaussian_kernel(self.data_for_classification[i],
                                                        self.data_for_classification[j], 
                                                        self._gamma)
                    self._value_of_c = None
                if self._kernel_type == POLYNOMIAL_KERNEL:
                    gram_matrix[i, j] = polynomial_kernel(self.data_for_classification[i],
                                                          self.data_for_classification[j], 
                                                          self._value_of_c, self._degree)
        
        P = cvxopt.matrix( np.outer(self.data_labels, self.data_labels) * gram_matrix )
        q = cvxopt.matrix( np.ones(self._n_samples) * -1.0 )
        A = cvxopt.matrix( self.data_labels, (1, self._n_samples) )
        b = cvxopt.matrix( 0.0 )
    
        if self._value_of_c == None or self._value_of_c==0:
            G = cvxopt.matrix( np.diag( np.ones(self._n_samples) * -1.0 ) )
            h = cvxopt.matrix( np.zeros( self._n_samples ) )
        else:
            
            tmp1 = np.diag( np.ones( self._n_samples) * -1.0 )
            tmp2 = np.identity(self._n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(self._n_samples)
            tmp2 = np.ones(self._n_samples) * self._value_of_c
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))
        
        cvxopt.solvers.options['show_progress'] = self._show_progress
        cvxopt.solvers.options['abstol']        = self._absolute_tolerance
        cvxopt.solvers.options['reltol']        = self._relative_tolerance
        cvxopt.solvers.options['feastol']       = self._feasibility_tolerance

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)        

        alphas = np.ravel( solution['x'] )

        
        sv = alphas > self._alphas_tolerance 
        ind = np.arange( len(alphas) )[sv]
        self.alphas = alphas[sv]
        self.sv = self.data_for_classification[sv]
        self.sv_labels = self.data_labels[sv]

        
        self.b = 0
        for n in range( len(self.alphas) ):
            self.b += self.sv_labels[n]
            self.b -= np.sum( self.alphas * self.sv_labels * gram_matrix[ind[n], sv] )
        self.b = self.b / len( self.alphas )

        if self._kernel_type == LINEAR_KERNEL:
            self.w = np.zeros(self._n_features)
            for n in range(len(self.alphas)):
                self.w += self.alphas[n] * self.sv_labels[n] * self.sv[n]
        else:
            self.w = None
            
    def classic_project ( self, _qdX ):
       
        if self.w is not None:
            return np.dot(_qdX, self.w) + self.b
        else:
            labels_predict = np.zeros(len(_qdX))
            for i in range(len(_qdX)):
                sum_tmp = 0
                for a, sv_labels, sv in zip(self.alphas, self.sv_labels, self.sv):

                    if self._kernel_type == LINEAR_KERNEL:
                        sum_tmp += a * sv_labels * linear_kernel(_qdX[i], sv)
                    
                    if self._kernel_type == GAUSSIAN_KERNEL:
                        sum_tmp += a * sv_labels * gaussian_kernel(_qdX[i], sv, self._sigma)
                        self._value_of_c = None 
                    
                    if self._kernel_type == POLYNOMIAL_KERNEL:
                        sum_tmp += a * sv_labels * polynomial_kernel(_qdX[i], sv, self._value_of_c, self._degree)

                labels_predict[i] = sum_tmp
                
        return labels_predict + self.b

    def classic_predict(self, _qdX):
        
        return np.sign( self.classic_project(_qdX) )
    
    def quantum_fit( self ):
        self.update_data_for_quantum_svm()
        self._nu = create_nu_coefficent( self.data_for_classification, 
                                         self._b, 
                                         self._alphas, 
                                         self._n_samples )

    def quantum_single_project ( self, _qdX ):
        # P =  0.5 * (1.0 - self.utildestate @ _qdX.T  )
        self._nx = create_nx_coefficent( _qdX, self._n_samples ) 
        P = 0.5 * ( 1.0 - create_dot_ux_for_classification( self._nu, 
                                                            self._nx, 
                                                            self._b, 
                                                            self._alphas, 
                                                            self.q_data_for_classification, 
                                                            _qdX, 
                                                            self._n_samples))
        return P

    def quantum_single_predict(self, _qdX):
        _label = 0
        _val = self.quantum_single_project( _qdX )
        if _val<0.5:
            _label = +1.0
        else:
            _label = -1.0
        
        return _label
    
    def quantum_project ( self, _qdX ):
        
        probabilities=np.zero( (_qdX.shape[0],) )
        
        idx=0
        for p in _qdX:
            probabilities[idx] = self.quantum_single_project( p )
            idx=idx+1
        
        return probabilities

    def quantum_predict( self, _qdX):

        q_test_d = np.empty((0,2), dtype=complex)
        for d in _qdX:
            q = encode_probe_by_normalization( d )
            q_test_d = np.append(q_test_d, [[ q[0], q[1] ]], axis=0)
        

        labels=np.zeros( (q_test_d.shape[0],) )

        idx=0
        for p in q_test_d:
            labels[idx] = self.quantum_single_predict( p )
            idx=idx+1

        
        return labels
        
# in preparation    
#
# TO DESC
#
class VQEClassification:
    
    def __init__( self ):
        self.params_filename = None, 
        self.save_params=0
        
        self.optymizer = None
        self.optimizer_type = OPT_COBYLA
        
        self._n_centers = -1
        self.centers = []
        
        self._n_layers = -1
        self._circuit_type_form = -1
        self._num_qubits = -1
        self._qubits = []
        

    def reset( self ):
        pass
    
    def set_qubits_table(self, _val_tab_qubits):
        self._qubits = _val_tab_qubits
    
    def set_number_of_qubits( self, _val_n_qubits):
        self._num_qubits = _val_n_qubits
    
    def create_n_centers( self, _val_n_centers ):
        if not self._num_qubits >= 1:
            raise ValueError("VQEClassification the number of qubits must be bigger or equal to one!")
            
        self._n_centers = _val_n_centers
        self.centers = np.zeros( shape=(2**self._num_qubits, self._n_centers ) )
        
    
    def set_center(self, _idx, _state):
        self.centers[:, _idx] = _state
    
    def get_center(self, _idx):
        return self.centers[:, _idx]

    def get_centers( self ):
        return self.centers
    
    def train_vqe_for_all_centers(self):
        pass
    
    def train_vqe( self, _initial_params, _state_for_train, _n_center):
        _method_name = None
        
        if self.optimizer_type==OPT_COBYLA:
            _method_name='COBYLA'
        
        result = scipy.optimize.minimize(
            fun=self.objective_function,
            x0=_initial_params,
            args=(self._qubits, _n_center, self._circuit_type_form, self._n_layers),
            method=_method_name )
        
        return result.x

   
    def objective_function( self, _parameters, *args):
        cost_value = 0.0
        
        _qubits, _n_center, _circuit_type, _n_layers, = args 
        
        q = self.perform_variational_circuit( _qubits, _parameters, _circuit_type, _n_layers )
        _state = q.ToNumpyArray()
        
        cost_value = sum( abs( self.centers[i, _n_center] - _state[i]) 
                         for i in range(2 ** self._num_qubits) )
    
        return cost_value
    
    def perform_variational_circuit( self, _qubits, _parameters, _formval, _layers):
         
         offsetidx=0
         
         q = qcs.QuantumReg( len( _qubits ) )
         q.Reset()
         
     # ----------------------------------- form 0
     #     
         if _formval == 0:
             for idx in range (0, len(_qubits)):
                 q.YRotN( _qubits[0 + idx], _parameters[offsetidx  + idx] )
          
             offsetidx = offsetidx + len(_qubits)
              
             q.CZ(0,1)
             q.CZ(2,0)
         
         
             for idx in range (0, len(_qubits)):
                 q.YRotN( _qubits[0 + idx], _parameters[offsetidx + idx] )
         
             offsetidx = offsetidx + len(_qubits)
         
         
             q.CZ(1,2)
             q.CZ(2,0)
         
             
             for idx in range (0, len(_qubits)):
                 q.YRotN( _qubits[0 + idx], _parameters[offsetidx + idx] )
         
             offsetidx=offsetidx+len(_qubits)
    
     # ----------------------------------- form 1    
     # linear entanglement
    
         if _formval == 1:
    
             for idx in range (0, len(_qubits)):
                 q.YRotN( _qubits[0 + idx], _parameters[offsetidx  + idx] )
    
             offsetidx=offsetidx+len(_qubits)
    
             for idx in range (0, len(_qubits)):
                 q.ZRotN( _qubits[0 + idx], _parameters[offsetidx  + idx] )
    
             offsetidx=offsetidx+len(_qubits)
            
             for idx in range (0, len(_qubits)-1):
                 q.CNot(idx, idx+1)
    
             for idx in range (0, len(_qubits)):
                 q.YRotN( _qubits[0 + idx], _parameters[offsetidx  + idx] )
    
             offsetidx=offsetidx+len(_qubits)
    
             for idx in range (0, len(_qubits)):
                 q.ZRotN( _qubits[0 + idx], _parameters[offsetidx  + idx])
    
             offsetidx=offsetidx+len(_qubits)
    
    
             for idx in range (0, len(_qubits)-1):
                 q.CNot(idx, idx+1)
    
     # ----------------------------------- form 2
     # full entanglement
         if _formval == 2:
    
             for idx in range (0, len(_qubits)):
                 q.YRotN(_qubits[0 + idx], _parameters[offsetidx  + idx])
    
             offsetidx=offsetidx+len(_qubits)
    
             for idx in range (0, len(_qubits)):
                 q.ZRotN(_qubits[0 + idx], _parameters[offsetidx  + idx])
    
             offsetidx=offsetidx+len(_qubits)
    
    
             for idx in range (0, len(_qubits)-1):
                 q.CNot(_qubits[idx], _qubits[idx+1])
    
             q.CNot(_qubits[0], _qubits[len(_qubits)-1])
    
    
             for idx in range (0, len(_qubits)):
                 q.YRotN( _qubits[0 + idx], _parameters[offsetidx  + idx])
    
             offsetidx=offsetidx+len(_qubits)
    
             for idx in range (0, len(_qubits)):
                 q.ZRotN( _qubits[0 + idx], _parameters[offsetidx  + idx])
    
             offsetidx=offsetidx+len(_qubits)
    
    
             for idx in range (0, len(_qubits)-1):
                 q.CNot(_qubits[idx], _qubits[idx+1])
    
             q.CNot(_qubits[0], _qubits[len(_qubits)-1])
    
    
     # ----------------------------------- form 3
     # 
         if _formval == 3:    
             for _ in range(0, _layers):
                 for idx in range (0, len(_qubits)):
                     q.XRotN( _qubits[0 + idx], _parameters[offsetidx  + idx] )
         
                 offsetidx=offsetidx+len(_qubits)
         
                 for idx in range (0, len(_qubits)):
                     q.ZRotN( _qubits[0 + idx], _parameters[offsetidx  + idx] )
         
                 offsetidx=offsetidx+len(_qubits)
         
                 for idx in range (0, len(_qubits)):
                     q.XRotN( _qubits[0 + idx], _parameters[offsetidx  + idx])
         
                 offsetidx=offsetidx+len(_qubits)
         
         
                 for idx in range (0, len(_qubits)-1):
                     q.CNot(idx, idx+1)
         
    
     # ----------------------------------- form 4
     #     
         if _formval == 4:
    
             for idx in range (0, len(_qubits)):
                 q.YRotN( _qubits[0 + idx], _parameters[offsetidx  + idx] )
      
             offsetidx=offsetidx+len(_qubits)
    
    
             for _ in range(0, _layers):
    
                 for idx in range (0, len(_qubits)):
                      q.YRotN( _qubits[0 + idx], _parameters[offsetidx  + idx] )
         
                 offsetidx=offsetidx+len(_qubits)
    
                 q.CZ(0, 1)
                       
                 for idx in range (0, len(_qubits)):
                      q.YRotN( _qubits[0 + idx], _parameters[offsetidx  + idx])
    
                 offsetidx=offsetidx+len(_qubits)
         
                 q.CZ(1, 2)
                 q.CZ(0, 2)
         
             for idx in range (0, len(_qubits)):
                 q.YRotN( _qubits[0 + idx], _parameters[offsetidx  + idx])
    
             offsetidx=offsetidx+len(_qubits)
    
     # ----------------------------------- form 5
     #     
         if _formval == 5:
             for idx in range (0, len(_qubits)):
                  q.YRotN( _qubits[0 + idx], _parameters[offsetidx  + idx] )
              
             offsetidx=offsetidx+len(_qubits)
             
             for _ in range(0, _layers):           
                 q.YRotN( _qubits[0], _parameters[offsetidx] )
                 offsetidx=offsetidx+1
                 
                 q.CNot(_qubits[0],_qubits[2])
                            
                 q.YRotN( _qubits[0], _parameters[offsetidx] )
                 offsetidx=offsetidx+1
                 
                 q.CNot(_qubits[0],_qubits[1])
                 
                 q.YRotN( _qubits[1], _parameters[offsetidx] )
                 offsetidx=offsetidx+1
                 
                 q.CNot( _qubits[1], _qubits[2] )
                 
             
             for idx in range (0, len(_qubits)):
                  q.YRotN(_parameters[offsetidx  + idx], _qubits[0 + idx])
              
             offsetidx=offsetidx+len(_qubits)        
             
    
         return q
       
    def set_angles_file_name( self, _fname ):
        self.params_filename = _fname
    
    def save_angles_to_file( self, _fname = None ):
        pass

    def load_angles_from_file( self, _fname = None ):
        pass

# in preparation
#
# TO DESC
#
class DistanceQuantumClassification:
    def __init__( self ):
        self._func_distance = None
        self._dimension = -1
        self._labels = None
        self._num_of_classes = -1
        self._centroids=None
            
    def reset( self ):
        pass

    def create_empty_centroids_for_n_classes(self, _n):
        self._num_of_classes = _n
        self._centroids = np.zeros( shape=( self._num_of_classes, 
                                            self._dimension, self._dimension ),
                                    dtype=complex )

    def create_centroid_for_class(self, _idx, _data_for_centroid):
        dm_for_centroid = create_quantum_centroid( _data_for_centroid )
        self._centroids[_idx] = dm_for_centroid
    
    def set_centroid(self, _idx, _centroid):
        self._centroids[_idx] = _centroid

    def get_centroid(self, _idx):
        return self._centroids[ _idx ]

    def set_distance(self, _f_dist):
        self._func_distance = _f_dist
            
    def set_dimension( self, _d ):
        self._dimension = _d
           
    def classify_all_probes( self, _qdX ):
        _labels = np.zeros( shape=( _qdX.shape[0]) ) 
        idx=0
        for r in _qdX:
            iclass = self.classify_probe( r )
            _labels[idx] = iclass
            idx = idx + 1
        
        return _labels
    
    def classify_probe( self, _qdX ):
        _dm_qdX = vector_state_to_density_matrix( _qdX )
        val_for_class = np.zeros( shape=(self._num_of_classes) )
        
        for iclass in range(0, self._num_of_classes):
            val_for_class[iclass] = self._func_distance( self._centroids[iclass], _dm_qdX )
        
        return np.argmin( val_for_class )
        


# in preparation
#
# TO DESC
#
class QuantumSpectralClustering:
    def __init__( self ):
        self._data_for_cluster = [ ]

        self._n_samples = -1
        self._n_clusters = -1
        self._n_features = -1
        
        self._distance_min = 0
        self._threshold = 0
        self._func_dist = None
        
        self._labels = None
        self._projectors = None

    
    def reset( self ):
        pass
    
    def set_data(self, _qdX):
        
        self._data_for_cluster = _qdX
        
        self._n_samples = _qdX.shape[0]
        self._n_features = _qdX.shape[1]
    
    def set_function_distance(self, _func):
        self._func_dist = _func
       
    def perform_classic_spectral_clustering(self, _val_n_clusters):
        self._n_clusters = _val_n_clusters
        
        self._labels = classic_spectral_clustering( self._data_for_cluster,
                                                    self._n_samples, 
                                                    self._n_clusters, 
                                                    self._threshold, 
                                                    _func_distance = self._func_dist )

    def perform_quantum_spectral_clustering(self, _val_n_clusters, _val_threshold):
        self._n_clusters = _val_n_clusters
        self._threshold = _val_threshold

        self._labels, self._projectors = quantum_spectral_clustering( 
                                              self._data_for_cluster, 
                                              self._n_samples, 
                                              self._n_clusters, 
                                              self._threshold, 
                                              _func_distance = self._func_dist )
    def classic_predicts(self):
        return self._labels
    
    def quantum_predicts(self):
        return self._labels
    
    def get_quantum_projectors(self):
        return self._projectors
    
    def quantum_get_rho(self):
        rho = create_rho_state_for_qsc( self._data_for_cluster,
                                        self._n_samples,
                                        self._n_clusters,
                                        self._threshold,
                                        _func_distance = self._func_dist )
        return rho
    

# in preparation
#
# TO DESC
#
class ClusteringByPotentialEnergy:
    def __init__( self ):
        self.dimension = -1
        self.bigE = 0
        self.data_for_cluster = [] 
        self._func_distance = None

    def reset( self ):
        pass
    
    def set_distance(self, _f_dist):
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
            for idx in range(self.dimension):
                dij2 = (_x[idx] - dval[idx]) ** 2.0
                evalue = np.exp( -1.0 * ( (dij2)/(two_sigma_sqr) ) )
                _psi = _psi + evalue
                sumval1 = sumval1 + dij2 * evalue
                sumval2 = sumval2 + evalue
        
        sumval = sumval + (sumval1/sumval2)
        
        coeff = 1.0 / ( 2.0 * (two_sigma_sqr) * _psi )
        
        rslt = _E - (self.dimension/2.0) + coeff * sumval
        
        return rslt

    def calc_V_with_distance( self, _x, _E, _sigma ):
        two_sigma_sqr = ( 2.0 * (_sigma ** 2.0) )
        
        _psi    = 0.0
        sumval  = 0.0
        sumval1 = 0.0
        sumval2 = 0.0

        for dval in self.data_for_cluster:
            dij2 = self._func_distance( _x, dval ) ** 2.0
            evalue = np.exp( -1.0 * ( (dij2)/(two_sigma_sqr) ) )
            _psi = _psi + evalue
            sumval1 = sumval1 + dij2 * evalue
            sumval2 = sumval2 + evalue
        
        sumval = sumval + (sumval1/sumval2)
        
        coeff = 1.0 / ( 2.0 * (two_sigma_sqr) * _psi )
        
        rslt = _E - (self.dimension/2.0) + coeff * sumval
        
        return rslt

    
    def calc_v_function_on_2d_mesh_grid(self, _sigma, _mesh_grid_x = 50, _mesh_grid_y = 50 ):

        minx = np.min( self.data_for_cluster[:, 0] )
        maxx = np.max( self.data_for_cluster[:, 0] )
        
        miny = np.min( self.data_for_cluster[:, 1] )
        maxy = np.max( self.data_for_cluster[:, 1] )    

        X, Y = np.mgrid[ minx:maxx:_mesh_grid_x*1J, miny:maxy:_mesh_grid_y*1J ]
        Z    = np.zeros( shape=X.shape )
        
        for idx in range( _mesh_grid_x ):
            for idy in range( _mesh_grid_y ):
                v=self.calc_V( [X[idx,idy], Y[idx,idy]], 0.0, _sigma)
                Z[idx,idy] = (-v)/(1.0+v)
        
        return Z

    def calc_v_function_with_distance_on_2d_mesh_grid(self, _sigma, _mesh_grid_x = 50, _mesh_grid_y = 50):
        
        if self._func_distance == None:
            raise  ValueError("Distance function has been not assigned!!!")
        
        minx = np.min( self.data_for_cluster[:, 0] )
        maxx = np.max( self.data_for_cluster[:, 0] )
        
        miny = np.min( self.data_for_cluster[:, 1] )
        maxy = np.max( self.data_for_cluster[:, 1] )    

        X, Y = np.mgrid[ minx:maxx:_mesh_grid_x*1J, miny:maxy:_mesh_grid_y*1J ]
        Z    = np.zeros( shape=X.shape )
        
        for idx in range( _mesh_grid_x ):
            for idy in range( _mesh_grid_y ):
                v=self.calc_V_with_distance( [X[idx,idy], Y[idx,idy]], 0.0, _sigma)
                Z[idx,idy] = (-v)/(1.0+v)
        
        return Z
    
    def find_clusters_centers(self):
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
    circle = None
    if _limits == None:
        ax.set_xlim( [-1.0, 1.0] )
        ax.set_ylim( [-1.0, 1.0] )
        circle = plt.Circle( (0,0), 1,  color='r', fill=False)
    else:
        ax.set_xlim( _limits[0], _limits[1] )
        ax.set_ylim( _limits[2], _limits[3] )
        if len(_limits) > 4:
            circle = plt.Circle( (0,0), _limits[4],  color='r', fill=False)

    ax.scatter( _qX[ :, _first_col ], _qX[ :, _second_col ])
    
    if circle is not None:
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
        If the first normalization results with an observation with all zero 
        values, then this observation will be deteled.

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
    #detection of an observation with all zero values
    h=0
    i=0
    aa=a
    while (h==0 and i<a):
        sum_all=0
        for j in range(b):
            sum_all+=KNtab[i,j]
        if sum_all==0:
            h=1
            aa-=1
        i+=1
    #quantum normalization
    if math.ceil(math.log2(b)) != math.floor(math.log2(b)):
        c=math.ceil(math.log2(b))
        c=2**c
        Qtab=np.zeros(shape=(aa,c))
    else:
        Qtab=np.ndarray(shape=(aa,b))
    k=0
    for i in range(a):
        sum_all=0
        for j in range(b):
            sum_all+=KNtab[i,j]
        if sum_all==0:
            print('Warning: an observation with all zero values occured and it will be deleted!')
        else:
            for j in range(b):
                Qtab[k,j]=sympy.sqrt(KNtab[i,j]/sum_all)
            k+=1
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
    dim=0
    if check==1:
        x=vector_check(uvector)
        y=vector_check(vvector)
    if (x==1) and (y==1):
        d=0.0
        if isinstance(uvector, np.ndarray):
            dim=uvector.shape[0]
            
        if isinstance(uvector, list):
            dim=len(uvector)
            
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


def cosine_distance_with_normalisation( uvector, vvector, r = 0 ):
    """
    Calculates a cosine distance between two vectors. If one or both entered 
    vectors are not normalized, the result will be returned after the normalisation. 

    Parameters
    ----------
    uvector, vvector : numpy array objects
        Vectors of complex numbers describing quantum states.
    r : integer
        The number of decimals to use while rounding the number (default is 0,
        i.e. the number is not rounded).

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
    >>> print(cosine_distance_with_normalisation(u, v))
        0.0
    A distance between the opposite states:
    >>> v=np.array([1,0])
    >>> u=np.array([-1,0])
    >>> print(cosine_distance_with_normalisation(u, v))
        2.0
    A distance between the orthogonal states:
    >>> v=np.array([1/math.sqrt(2),0 + 1j/math.sqrt(2)])
    >>> u=np.array([1/math.sqrt(2),0 - 1j/math.sqrt(2)])
    >>> print(cosine_distance_with_normalisation(u, v))
        (1+0j)
    >>> v=np.array([1/math.sqrt(2),1/math.sqrt(2)])
    >>> u=np.array([1/math.sqrt(2),-1/math.sqrt(2)])
    >>> print(cosine_distance_with_normalisation(u, v))
        1.0
    A distance between two examplary states:
    >>> v=np.array([1/math.sqrt(2),1/math.sqrt(2)])
    >>> u=np.array([1/math.sqrt(2),0 + 1j/math.sqrt(2)])
    >>> print(cosine_distance_with_normalisation(u, v, 3))
        (0.5+0.5j)
    If entered vector v is not a correct quantum state:
    >>> v=np.array([1,1])
    >>> u=np.array([1,0])
    >>> print(cosine_distance_with_normalisation(u, v, 4))
        0.2929

    """
    distance_value = 1.0 - np.vdot(uvector, vvector) / ( np.linalg.norm( uvector ) * np.linalg.norm( vvector ) )
    if r==0:
        return distance_value 
    else:
        return np.round(distance_value,r)

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

#
# TO DESC
#
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

#
# TO DESC
#
def probability_as_distance_case_qubit_alpha(uvector, vvector, r=0, check=0):
    return probability_as_distance(uvector, vvector, 0, r, check)

#
# TO DESC
#
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

def swap_test_value(uvector, vvector, r=0, check=0):
    """
    Calculates the probability of measuring |0> on the first qubit of the Swap-Test 
    circuit when two other inputs of this circuit serve to enter states uvector
    and vvector. If the mentioned probability of measuring |0> is 1, the states
    uvector and vvector are the same; for any othogonal states uvector and 
    vvector Pr(|0>)=0.5.

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
        The probability of measuring |0> on the first qubit.
    Examples
    --------
    A Swap-Test value for the same states:
    >>> v=np.array([1,0])
    >>> u=np.array([1,0])
    >>> print(swap_test_value(u, v))
        1.0
    A Swap-Test value for the orthogonal states:
    >>> v=np.array([1/math.sqrt(2),0 + 1j/math.sqrt(2)])
    >>> u=np.array([1/math.sqrt(2),0 - 1j/math.sqrt(2)])
    >>> print(swap_test_value(u, v))
        0.5
    A Swap-Test value for two examplary states:
    >>> v=np.array([1/math.sqrt(2),1/math.sqrt(2)])
    >>> u=np.array([1/math.sqrt(2),0 + 1j/math.sqrt(2)])
    >>> print(swap_test_value(u, v, 3))
        0.75
    If entered vector v is not a correct quantum state:
    >>> v=np.array([1,1])
    >>> u=np.array([1,0])
    >>> print(swap_test_value(u, v, 0, 1))
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
            rslt = float(0.5 + 0.5 * np.linalg.norm( (uvector @ (np.conj(vvector))) ) ** 2)
        else:
            rslt = round( float(0.5 + 0.5 * np.linalg.norm( (uvector @ (np.conj(vvector))) ) ** 2), r )
        return rslt
    else:
        return None

def swap_test_as_distance_p0(uvector, vvector, r=0, check=0):
    """
    Calculates the distance between two vectors based on the Swap-Test (according
    to the probability of measuring |0> on the first qubit).
    The distance between the same vectors is zero. Maximal distance value, in
    this case, is one (for othogonal states) - the Swap-Test value is subtracted 
    from one and projected into range [0;1].

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
        The distance based on the Swap-Test, i.e. a float in range [0.0;1.0].
    Examples
    --------
    A Swap-Test distance value for the same states:
    >>> v=np.array([1,0])
    >>> u=np.array([1,0])
    >>> print(swap_test_as_distance_p0(u, v))
        0.0
    A Swap-Test distance value for the orthogonal states:
    >>> v=np.array([1/math.sqrt(2),0 + 1j/math.sqrt(2)])
    >>> u=np.array([1/math.sqrt(2),0 - 1j/math.sqrt(2)])
    >>> print(swap_test_as_distance_p0(u, v))
        1.0
    A Swap-Test distance value for two examplary states:
    >>> v=np.array([1/math.sqrt(2),1/math.sqrt(2)])
    >>> u=np.array([1/math.sqrt(2),0 + 1j/math.sqrt(2)])
    >>> print(swap_test_as_distance_p0(u, v, 3))
        0.5
    If entered vector v is not a correct quantum state:
    >>> v=np.array([1,1])
    >>> u=np.array([1,0])
    >>> print(swap_test_as_distance_p0(u, v, 0, 1))
        ...
        ValueError: The vector is not normalized!
        
    """
    std=float(1.0 - swap_test_value(uvector, vvector, r, check))
    return float(std/0.5)

def swap_test_as_distance_p1(uvector, vvector, r=0, check=0):
    """
    Calculates the distance between two vectors based on the Swap-Test (according
    to the probability of measuring |1> on the first qubit).
    The Swap-Test value is subtracted from one and projected into range [0;1].

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
        The distance based on the Swap-Test, i.e. a float in range [0.0;1.0].
    Examples
    --------
    A Swap-Test distance value for the same states:
    >>> v=np.array([1,0])
    >>> u=np.array([1,0])
    >>> print(swap_test_as_distance_p1(u, v))
        1.0
    A Swap-Test distance value for the orthogonal states:
    >>> v=np.array([1/math.sqrt(2),0 + 1j/math.sqrt(2)])
    >>> u=np.array([1/math.sqrt(2),0 - 1j/math.sqrt(2)])
    >>> print(swap_test_as_distance_p1(u, v))
        0.0
    A Swap-Test distance value for two examplary states:
    >>> v=np.array([1/math.sqrt(2),1/math.sqrt(2)])
    >>> u=np.array([1/math.sqrt(2),0 + 1j/math.sqrt(2)])
    >>> print(swap_test_as_distance_p1(u, v, 3))
        0.5
    If entered vector v is not a correct quantum state:
    >>> v=np.array([1,1])
    >>> u=np.array([1,0])
    >>> print(swap_test_as_distance_p1(u, v, 0, 1))
        ...
        ValueError: The vector is not normalized!
        
    """
    return float(1.0 - swap_test_as_distance_p0(uvector, vvector, r, check))


def euclidean_distance_without_sqrt(uvector, vvector, r=0, check=0):
    """
    Calculates the distance between two pure states based on the Euclidean 
    distance (without final square root).

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
        The distance between given quantum states as value in range [0.0, 2.0].
    Examples
    --------
    A distance value for the same states:
    >>> v=np.array([1,0])
    >>> u=np.array([1,0])
    >>> print(euclidean_distance_without_sqrt(u, v))
        0.0
    A distance value for the orthogonal states:
    >>> v=np.array([1/math.sqrt(2),0 + 1j/math.sqrt(2)])
    >>> u=np.array([1/math.sqrt(2),0 - 1j/math.sqrt(2)])
    >>> print(euclidean_distance_without_sqrt(u, v, 3))
        2.0
    A distance value for two examplary states:
    >>> v=np.array([1/math.sqrt(2),1/math.sqrt(2)])
    >>> u=np.array([1/math.sqrt(2),0 + 1j/math.sqrt(2)])
    >>> print(euclidean_distance_without_sqrt(u, v, 3))
        1.0
    If entered vector v is not a correct quantum state:
    >>> v=np.array([1,1])
    >>> u=np.array([1,0])
    >>> print(euclidean_distance_without_sqrt(u, v, 0, 1))
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
            rslt = np.sum( ( np.abs( (uvector - vvector) ) ) ** 2.0 )
        else:
            rslt = round( np.sum( ( np.abs( (uvector - vvector) ) ) ** 2.0 ), r )
        return rslt
    else:
        return None

def euclidean_distance_with_sqrt(uvector, vvector, r=0, check=0):
    """
    Calculates the distance between two pure states based on the Euclidean 
    distance.

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
        The distance between given quantum states as value in range [0.0, 
        np.sqrt(2.0)].
    Examples
    --------
    A distance value for the same states:
    >>> v=np.array([1,0])
    >>> u=np.array([1,0])
    >>> print(euclidean_distance_with_sqrt(u, v))
        0.0
    A distance value for the orthogonal states:
    >>> v=np.array([1/math.sqrt(2),0 + 1j/math.sqrt(2)])
    >>> u=np.array([1/math.sqrt(2),0 - 1j/math.sqrt(2)])
    >>> print(euclidean_distance_with_sqrt(u, v, 3))
        1.414
    A distance value for two examplary states:
    >>> v=np.array([1/math.sqrt(2),1/math.sqrt(2)])
    >>> u=np.array([1/math.sqrt(2),0 + 1j/math.sqrt(2)])
    >>> print(euclidean_distance_with_sqrt(u, v, 3))
        1.0
    If entered vector v is not a correct quantum state:
    >>> v=np.array([1,1])
    >>> u=np.array([1,0])
    >>> print(euclidean_distance_with_sqrt(u, v, 0, 1))
        ...
        ValueError: The vector is not normalized!
        
    """
    rslt = euclidean_distance_without_sqrt(uvector, vvector, 0, check)
    if r==0:
        return np.sqrt(rslt)
    else:
        return round(np.sqrt(rslt), r)
    

#
# TO DESC
#
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

#
# TO DESC
#
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

def data_vertical_stack(d1, d2):
    """
    Uses numpy.vstack function. Combines the rows of arrays given as parameters
    to one output array.

    Parameters
    ----------
    d1, d2 : numpy arrays
        Two arrays with the same number of columns.

    Returns
    -------
    numpy array
        An array gathering the rows of arrays given as this function's 
        parameters.
        
    Examples
    --------
    >>> a = np.array([1/np.sqrt(2),1/np.sqrt(2)])
    >>> b = np.array([1/np.sqrt(2),0+1j/np.sqrt(2)])
    >>> qdcl.data_vertical_stack(a,b)
    [[0.70710678+0.j         0.70710678+0.j        ]
     [0.70710678+0.j         0.        +0.70710678j]]
    >>> a = np.array([[1,0,0],[0,0,1]])
    >>> b = np.array([0, 1, 0])
    >>> qdcl.data_vertical_stack(a,b)
    [[1 0 0]
     [0 0 1]
     [0 1 0]]

    """
    return np.vstack( (d1,d2) )

#
# TO DESC
#
def data_horizontal_stack(d1, d2):
    return np.hstack( (d1,d2) )


def create_blob_2d( _n_samples = 100, _center=None):
    """
    Uses numpy.random.multivariate_normal function with the unit matrix as 
    a covariance matrix. Generates samples as two-dimensional blobs.

    Parameters
    ----------
    _n_samples : integer, optional
        The number of samples. The default value is 100.
    _center : numpy array, optional
        An array pointing out the center around which blobs will be generated. 
        The default value is np.array([0, 0]).

    Returns
    -------
    d1 : numpy array
        Contains generated samples' coordinates.
        
    Example
    -------
    Creation of five samples:
    >>> qdcl.create_blob_2d(5)
    [[ 0.59011169 -1.59784931]
     [-1.52844339  0.19637419]
     [-1.32380436 -0.18730801]
     [ 1.20515547 -0.69428618]
     [ 0.70639911  2.31603014]]
    
    """   

    if _center is None:
        mean_for_d1 = np.array([0, 0])
    else:
        mean_for_d1 = _center

    cov = np.array([[1.0, 0.0], 
                    [0.0, 1.0]])
    
    d1 = np.random.multivariate_normal(mean_for_d1, cov, _n_samples)

    return d1


#
# TO DESC
#
def create_circles_data_set( _n_samples = 100, _factor = 0.75, _noise = None, _random_state = 1234):
    
    rslt_data = None
    rslt_labels = None
    
    np.random.seed( _random_state )
    
    data_for_circle_out = np.linspace(0.0, 2.0 * np.pi, _n_samples, endpoint=False)
    data_for_circle_in  = np.linspace(0.0, 2.0 * np.pi, _n_samples, endpoint=False)
  
    circ_coords_x_out = np.cos(data_for_circle_out)
    circ_coords_y_out = np.sin(data_for_circle_out)
    
    circ_coords_x_in = np.cos(data_for_circle_in) * _factor
    circ_coords_y_in = np.sin(data_for_circle_in) * _factor  
  
     
    rslt_data = np.transpose( data_vertical_stack( 
                                np.append(circ_coords_x_out, circ_coords_x_in),
                                np.append(circ_coords_y_out, circ_coords_y_in)
                              )
    )
    
    rslt_labels = data_horizontal_stack(
        np.zeros(_n_samples, dtype=np.int64), 
        np.ones(_n_samples, dtype=np.int64)
    )
    
    if _noise is not None:
        rslt_data += np.random.normal(scale=_noise, size=rslt_data.shape)
    
    return rslt_data, rslt_labels

#
# TO DESC
#
def create_moon_data_set( _n_samples = 100, _shuffle = True, _noise = None, _random_state = 1234 ):
    
    rslt_data = None
    rslt_labels = None
           
    np.random.seed( _random_state )
   
    circ_coords_x_out = np.cos(np.linspace(0.0, np.pi, _n_samples))
    circ_coords_y_out = np.sin(np.linspace(0.0, np.pi, _n_samples))
    circ_coords_x_in  = 1.0 - np.cos(np.linspace(0.0, np.pi, _n_samples))
    circ_coords_y_in  = 1.0 - np.sin(np.linspace(0.0, np.pi, _n_samples)) - 0.5 
    
    rslt_data = np.transpose( data_vertical_stack( 
                                np.append(circ_coords_x_out, circ_coords_x_in),
                                np.append(circ_coords_y_out, circ_coords_y_in)
                              )
    )
    
    rslt_labels = data_horizontal_stack(
        np.zeros(_n_samples, dtype=np.int64), 
        np.ones(_n_samples, dtype=np.int64)
    )

    if _shuffle == True:
        new_indices = np.random.permutation( 2 * _n_samples )
        rslt_data = rslt_data[new_indices]
        rslt_labels = rslt_labels[new_indices]

    if _noise is not None:
        rslt_data += np.random.normal(scale=_noise, size=rslt_data.shape)
            
    
    return rslt_data, rslt_labels

# add labels 
#
# TO DESC
#
def create_data_non_line_separated_four_lines( _n_samples = 50, _centers=None ):        
   
    mean1 = [-2,  2]
    mean2 = [ 1, -1]
    mean3 = [ 3, -3]
    mean4 = [-4,  4]

    cov = [[1.0, 0.9], 
           [0.9, 1.0]]
    
    d1 = np.random.multivariate_normal( mean1, cov, _n_samples )
    d1 = np.vstack( (d1, np.random.multivariate_normal(mean3, cov, _n_samples)) )
    d2 = np.random.multivariate_normal( mean2, cov, _n_samples )
    d2 = np.vstack( (d2, np.random.multivariate_normal(mean4, cov, _n_samples)) )
   
    line_data = None
    line_labels = None
    
    line_data=data_vertical_stack( d1, d2 )

    labels_d1 = np.ones( shape=(_n_samples * 2, ) )
    labels_d2 = np.multiply( np.ones( shape=(_n_samples * 2, ) ), -1.0)

    line_labels = data_horizontal_stack(labels_d1, labels_d2)

    return line_data, line_labels

# add labels 
#
# TO DESC
#
def create_data_separated_by_line( _n_samples = 100, _centers=None ):        
    
    if _centers == None:
        mean_for_d1 = np.array([0, 3])
        mean_for_d2 = np.array([3, 0])
    else:
        mean_for_d1 = np.array( _centers[0] )
        mean_for_d2 = np.array( _centers[1] )
    
    cov = np.array([[0.8, 0.5], 
                    [0.5, 0.8]])
    
    d1 = np.random.multivariate_normal(mean_for_d1, cov, _n_samples)
    d2 = np.random.multivariate_normal(mean_for_d2, cov, _n_samples)
    
    labels_d1 = np.ones( shape=(_n_samples, ) )
    labels_d2 = np.multiply( np.ones( shape=(_n_samples, ) ), -1.0)
    
    return d1, d2, labels_d1, labels_d2

#
# TO DESC
#
def split_data_and_labels( _qdX1, _labels1,  _qdX2, _labels2, _ratio ):
    
    idx_for_cutoff = int( _qdX1.shape[0] * _ratio )
    
    train_d1      =    _qdX1[:idx_for_cutoff]
    train_labels1 = _labels1[:idx_for_cutoff]
    train_d2      =    _qdX2[:idx_for_cutoff]
    train_labels2 = _labels2[:idx_for_cutoff]
    
    train_d      = np.vstack( (train_d1, train_d2) )
    train_labels = np.hstack( (train_labels1, train_labels2) )
    
    test_d1      =    _qdX1[idx_for_cutoff:]
    test_labels1 = _labels1[idx_for_cutoff:]
    test_d2      =    _qdX2[idx_for_cutoff:]
    test_labels2 = _labels2[idx_for_cutoff:]
    
    test_d      = np.vstack( (test_d1, test_d2) )
    test_labels = np.hstack( (test_labels1, test_labels2) )
        
    
    return train_d, train_labels, test_d, test_labels

def create_spherical_probes( _n_points, _n_dim=2):
    """
    Creates normalized points on a unit circle.
    This data can be regared as qubits where probability amplitudes are 
    real numbers.
    
    Parameters
    ----------
    _n_points : integer
        The number of probes to generate.
    _n_dim : integer, optional
        The number of probe's dimensions. The default value is 2.

    Returns
    -------
    numpy array
        The array containing probes.
    
    Example
    -------
    Creation of ten 2-dimensional probes.
    >>> qdcl.qdcl.create_spherical_probes(10)
    [[ 0.83199628 -0.5547812 ]
     [-0.69520821  0.71880842]
     [-0.83934711 -0.54359583]
     [ 0.95564325 -0.29452671]
     [ 0.50293973 -0.86432149]
     [ 0.14637115  0.98922974]
     [ 0.96195755  0.27319897]
     [ 0.26458467 -0.96436246]
     [-0.9540489  -0.29965096]
     [-0.31569086  0.9488621 ]]

    """
    
    if _n_points > 0:
        _unit_vectors = np.random.randn( _n_dim, _n_points )
        _unit_vectors /= np.linalg.norm( _unit_vectors, axis=0 )
    else:
        raise ValueError("The number of points must be positive integer number!")
        return None
    
    return _unit_vectors.T

#
# TO DESC
#
def create_focused_circle_probes( _n_points, _n_focus_points, _width_of_cluster=0.25 ):
    """
    Generates observations/probes as normalized 1-qubit states using function
    sklearn.datasets.make_blobs.
    
    Parameters
    ----------
    _n_points : integer
        The number of probes to generate.
    _n_focus_points : integer
        The number of centers (sklearn.datasets.make_blobs generates blobs 
        for clustering).
    _width_of_cluster : float, optional
        The standard deviation of the clusters. The default value is 0.25.

    Returns
    -------
    d : numpy array
        The array containing normalized probes.
    
    Example
    -------
    Creation of ten 2-dimensional probes using three cluster centers.
    >>> qdcl.create_focused_circle_probes( 10, 3 )
    [[ 0.44510238 -0.89547969]
     [-0.4555554   0.89020743]
     [ 0.52455015 -0.85137955]
     [ 0.43051802 -0.90258198]
     [-0.85803463 -0.51359184]
     [-0.87752778 -0.4795258 ]
     [-0.43812699  0.89891309]
     [-0.8647385  -0.50222239]
     [ 0.50517566 -0.86301654]
     [-0.44428403  0.89588599]]

    """
    d, _ = make_blobs( n_samples=_n_points,
                       n_features=2,
                       centers = _n_focus_points,
                       cluster_std=_width_of_cluster )

    for i in range(_n_points):
        d[i] = d[i] / np.linalg.norm(d[i])
    
    return d

#
# TO DESC
#
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

# to check
#
# TO DESC
#
def create_focused_qubits_probes( _n_points, _n_focus_points, _width_of_cluster=0.25 ):
    
    d, _ = make_blobs( n_samples=_n_points,
                       n_features=3,
                       centers = _n_focus_points,
                       cluster_std=_width_of_cluster )

    for i in range(_n_points):
        d[i] = d[i] / np.linalg.norm(d[i])
    
    return d

# to check
#
# TO DESC
#
def create_focused_qubits_probes_with_uniform_placed_centers( _n_points, _n_theta, _n_psi, _width_of_cluster=0.1, _return_labels = False, _return_centers = False ):
    
    centers_on_sphere = [ ]
    
    _theta = 0.0
    _psi = 0.0
    _theta_delta= (2.0*np.pi) / _n_theta
    _psi_delta= (np.pi) / _n_psi
    
    for i in range( _n_theta ):
        for j in range( _n_psi ):
            sp = convert_spherical_coordinates_to_bloch_vector(1.0, _theta, _psi)
            centers_on_sphere.append( ( sp[0], sp[1], sp[2]) ) 
            _theta = _theta +_theta_delta
            _psi = _psi + _psi_delta


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

#
# TO DESC
#
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
    # a jak nie, to wyjątkiem ;-) DimensionError, podobnie jak w EntDetectorze
    omega = np.arccos(np.dot(p0/np.linalg.norm(p0), p1/np.linalg.norm(p1)))
    so = np.sin(omega)
    
    return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1

#
# TO DESC
#
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

#
# TO DESC
#
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

#
# TO DESC
#
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


#
# TO DESC
#
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

#
# TO DESC
#
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

def calculate_distance(_qdX, _vector, _func_distance):
    """
    Calculates distances, according to a given function, between a vector state 
    and vectors in a data table.

    Parameters
    ----------
    _qdX : numpy ndarray
        A data array which contains vector states in each row.
    _vector : numpy ndarray
        A vector state.
    _func_distance : function
        The paramter pointing chosen distance function.

    Returns
    -------
    distance_table : numpy ndarray
        An array with distance values between _vector and subsequent rows of 
        _qdX array.
    
    Example
    -------
    Let data_tab be an array of three vectors. Function calculates distances,
    according to the swap_test_value, between these vectors and vector [1,0]:
    >>> data_tab=np.array( [[1,0], [0,1], [1/np.sqrt(2),1/np.sqrt(2)]] )
    >>> qdcl.calculate_distance(data_tab, np.array([1,0]), qdcl.swap_test_value)
        [1.   0.5  0.75]

    """
    distance_table=np.zeros( shape=(_qdX.shape[0] ) )
    idx=0
    for e in _qdX:
        distance_table[idx] = _func_distance(e, _vector)
        #distance_table[idx, 1] = l
        idx=idx+1
    
    return distance_table
    

def create_distance_table( _qdX, _centers, _labels, _n_clusters, _func_distance=None ):
    """
    Calculates a distance between each probe/observation in a given data table
    and the cluster to which it belongs.    

    Parameters
    ----------
    _qdX : numpy ndarray
        A data array which contains vector states in each row.
    _centers : numpy ndarray
        An array with coordinates of clusters centers.
    _labels : numpy ndarray
        An array of class labels for each observation from data table.
    _n_clusters : integer
        The number of clusters in the data set.
    _func_distance : function, optional
        The paramter pointing chosen distance function. The default is None.

    Returns
    -------
    distance_table : numpy ndarray
        Two-column array. In the first column, the distance between a probe from
        data table and the cluster center is calculated. In the second column, 
        the label of class to which the probe belongs is listed. 

    Example
    -------
    Let data_tab be a two-column array taken from the file CRABS.xlsx. Function 
    calculates distances for clusters generated by function kmedoids_quantum_states:
    >>> df = pd.read_excel(r'CRABS.xlsx')
    >>> data_tab=qdcl.convert_data_to_vector_states(df,2)
    >>> probes_no=data_tab.shape[0]
    >>> clusters_no=3
    >>> d = qdcl.create_focused_circle_probes_with_uniform_placed_centers(probes_no, clusters_no, _width_of_cluster=0.15)
    >>> labels, centers = qdcl.kmedoids_quantum_states( d, clusters_no, _func_distance=qdcl.COSINE_DISTANCE )
    >>> print(qdcl.create_distance_table(data_tab, centers, labels, clusters_no, qdcl.cosine_distance))
        [[0.35312776 0.        ]
         [0.33689475 0.        ]...
         [0.70764875 1.        ]
         [0.69604256 1.        ]...
         [1.97182289 2.        ]
         [1.97035611 2.        ]...
        
    """
    idx=0
    distance_table=np.zeros( shape=(_qdX.shape[0], 2) )
    for l in range(0, _n_clusters):
        cntr=_centers[l]
        for e in _qdX[_labels == l]:
            distance_table[idx, 0] = _func_distance(e, cntr)
            distance_table[idx, 1] = l
            idx=idx+1
    
    return distance_table

def get_distances_for_cluster( _dist_tab, _n_cluster ):
    """
    Displays distances of probes from the cluster's center for a given cluster.

    Parameters
    ----------
    _dist_tab : numpy ndarray
        Two-column array. In the first column, the distance between a probe from
        data table and the cluster center is calculated. In the second column, 
        the label of class to which the probe belongs is listed. A distance table
        may be calculated by function create_distance_table.
    _n_cluster : int
        A label of a cluster.

    Returns
    -------
    numpy ndarray
        An array containing distances of probes in a given cluster.

    Example
    -------
    Let data_tab be a two-column array taken from the file CRABS.xlsx. The 
    function create_distance_table calculates distances for all clusters 
    generated by the function kmedoids_quantum_states. The function 
    get_distances_for_cluster retuns distances only for cluster 1:
    >>> df = pd.read_excel(r'CRABS.xlsx')
    >>> data_tab=qdcl.convert_data_to_vector_states(df,2)
    >>> probes_no=data_tab.shape[0]
    >>> clusters_no=3
    >>> d = qdcl.create_focused_circle_probes_with_uniform_placed_centers(probes_no, clusters_no, _width_of_cluster=0.15)
    >>> labels, centers = qdcl.kmedoids_quantum_states( d, clusters_no, _func_distance=qdcl.COSINE_DISTANCE )
    >>> distances=qdcl.create_distance_table(data_tab, centers, labels, clusters_no, qdcl.cosine_distance)
    >>> print(qdcl.get_distances_for_cluster( distances, 1 ))
        [[0.3387702  1.        ]
         [0.32479836 1.        ]
         [0.32617641 1.        ]
         [0.32430024 1.        ]
         [0.33310496 1.        ]
         [0.34288569 1.        ]
         [0.33766788 1.        ]
         [0.35303781 1.        ]
         [0.34595025 1.        ]...
    
    """
    return _dist_tab[ _dist_tab[:, 1] == _n_cluster ]

def get_data_for_class(_qdX, _labels, _class):
    """
    Displays data from a required class/cluster.

    Parameters
    ----------
    _qdX : numpy ndarray
        A data array which contains vector states in each row.
    _labels : numpy ndarray
        An array of class labels.
    _class : int
        A value pointing out the class/cluster of data which will be returned 
        by this function.

    Returns
    -------
    numpy ndarray
        An array of probes for a class/cluster given in _class.

    Example
    -------
    Let data_tab be a two-column array taken from the file CRABS.xlsx. Function 
    kmedoids_quantum_states returns clusters' labels and centers. The function
    get_data_for_class uses data array and label array to return probes for 
    class/cluster pointed out by the parameter _class:
    >>> df = pd.read_excel(r'CRABS.xlsx')
    >>> data_tab=qdcl.convert_data_to_vector_states(df,2)
    >>> probes_no=data_tab.shape[0]
    >>> clusters_no=3
    >>> d = qdcl.create_focused_circle_probes_with_uniform_placed_centers(probes_no, clusters_no, _width_of_cluster=0.15)
    >>> labels, centers = qdcl.kmedoids_quantum_states( d, clusters_no, _func_distance=qdcl.COSINE_DISTANCE )
    >>> print(qdcl.get_data_for_class( data_tab, labels, 0 ))
        [[0.74256439 0.66977469]
         [0.72742559 0.68618658]
         [0.74346662 0.66877304]...
    >>> print(qdcl.get_data_for_class( data_tab, labels, 1 ))
        [[0.73979544 0.67283185]
         [0.73029674 0.68313005]
         [0.74199852 0.67040152]...
    >>> print(qdcl.get_data_for_class( data_tab, labels, 2 ))
        [[0.73979544 0.67283185]
         [0.74199852 0.67040152]
         [0.73854895 0.67419986]...
    
    """
    return _qdX[ _labels == _class ]

def get_min_label_class(_labels):
    """
    Calculates the minimal label value.    

    Parameters
    ----------
    _labels : numpy ndarray
        An array of class labels.

    Returns
    -------
    integer
        A value of the minimal label. 

    Example
    -------
    Let data_tab be a two-column array taken from the file CRABS.xlsx. Function 
    kmedoids_quantum_states returns clusters' labels and centers. The array of
    labels is a parameter for get_min_label_class - function returns 0 because
    it is the minimal value from the set of all labels (0, 1, 2):
    >>> df = pd.read_excel(r'CRABS.xlsx')
    >>> data_tab=qdcl.convert_data_to_vector_states(df,2)
    >>> probes_no=data_tab.shape[0]
    >>> clusters_no=3
    >>> d = qdcl.create_focused_circle_probes_with_uniform_placed_centers(probes_no, clusters_no, _width_of_cluster=0.15)
    >>> labels, centers = qdcl.kmedoids_quantum_states( d, clusters_no, _func_distance=qdcl.COSINE_DISTANCE )
    >>> print(qdcl.get_min_label_class( labels ))
        0
        
    """
    return np.min( _labels )

def get_max_label_class(_labels):
    """
    Calculates the maximal label value.    

    Parameters
    ----------
    _labels : numpy ndarray
        An array of class labels.

    Returns
    -------
    integer
        A value of the maximal label. 

    Example
    -------
    Let data_tab be a two-column array taken from the file CRABS.xlsx. Function 
    kmedoids_quantum_states returns clusters' labels and centers. The array of
    labels is a parameter for get_max_label_class - function returns 2 because
    it is the maximal value from the set of all labels (0, 1, 2):
    >>> df = pd.read_excel(r'CRABS.xlsx')
    >>> data_tab=qdcl.convert_data_to_vector_states(df,2)
    >>> probes_no=data_tab.shape[0]
    >>> clusters_no=3
    >>> d = qdcl.create_focused_circle_probes_with_uniform_placed_centers(probes_no, clusters_no, _width_of_cluster=0.15)
    >>> labels, centers = qdcl.kmedoids_quantum_states( d, clusters_no, _func_distance=qdcl.COSINE_DISTANCE )
    >>> print(qdcl.get_max_label_class( labels ))
        2
        
    """
    return np.max( _labels )

def get_vectors_for_label(l, _qdX, labels, _n_samples):
    """
    Returns vectors associated with a given label.    

    Parameters
    ----------
    l : integer
        A number of cluster.
    _qdX : numpy ndarray
        A data array.
    _labels : numpy ndarray
        An array of class labels.
    _n_samples : integer
        A number of samples to be taken into account from _qdX.

    Returns
    -------
    numpy ndarray
        Vectors associated with a given label. 
    
    Example
    -------
    Let data_tab be a two-column array taken from the file CRABS.xlsx. Function 
    kmedoids_quantum_states returns clusters' labels and centers. The function
    get_vectors_for_label returns probes for a given class/cluster:
    >>> df = pd.read_excel(r'CRABS.xlsx')
    >>> data_tab=qdcl.convert_data_to_vector_states(df,2)
    >>> probes_no=data_tab.shape[0]
    >>> clusters_no=3
    >>> d = qdcl.create_focused_circle_probes_with_uniform_placed_centers(probes_no, clusters_no, _width_of_cluster=0.15)
    >>> labels, centers = qdcl.kmedoids_quantum_states( d, clusters_no, _func_distance=qdcl.COSINE_DISTANCE )
    >>> print(qdcl.get_vectors_for_label( 0, data_tab, labels, probes_no ))
        [[0.72702918 0.68660656]
         [0.74667404 0.66519011]
         [0.74600385 0.66594163]...
    >>> print(qdcl.get_vectors_for_label( 1, data_tab, labels, probes_no ))
        [[0.73029674 0.68313005]
         [0.74667404 0.66519011]
         [0.74600385 0.66594163]...
    >>> print(qdcl.get_vectors_for_label( 2, data_tab, labels, probes_no ))
        [[0.73564697 0.67736514]
         [0.73854895 0.67419986]
         [0.72742559 0.68618658]...

    More sophisticated examples of use in a file vqe-for_credit_risk.py
        
    """
    outlist = []
    for idx in range(0, _n_samples):
        if l == labels[idx]:
            outlist.append( _qdX[idx] )
    
    return np.array( outlist )

def get_vector_of_idx_for_label(l, labels, _qdX, _n_samples):
    """
    Returns indexes of vectors associated with a given label.    

    Parameters
    ----------
    l : integer
        A number of cluster.
    _labels : numpy ndarray
        An array of class labels.
    _qdX : numpy ndarray
        A data array.   
    _n_samples : integer
        A number of samples to be taken into account from _qdX.

    Returns
    -------
    numpy ndarray
        Indexes of vectors associated with a given label from data table. 
    
    Example
    -------
    Let data_tab be a two-column array taken from the file CRABS.xlsx. Function 
    kmedoids_quantum_states returns clusters' labels and centers. The function
    get_vector_of_idx_for_label returns indexes of probes for a given class/cluster:
    >>> df = pd.read_excel(r'CRABS.xlsx')
    >>> data_tab=qdcl.convert_data_to_vector_states(df,2)
    >>> probes_no=data_tab.shape[0]
    >>> clusters_no=3
    >>> d = qdcl.create_focused_circle_probes_with_uniform_placed_centers(probes_no, clusters_no, _width_of_cluster=0.15)
    >>> labels, centers = qdcl.kmedoids_quantum_states( d, clusters_no, _func_distance=qdcl.COSINE_DISTANCE )
    >>> print(qdcl.get_vector_of_idx_for_label(0, labels, data_tab, probes_no ))
        [  3   7   8  12  14  16  21  23  24  25 ...
    >>> print(qdcl.get_vector_of_idx_for_label(1, labels, data_tab, probes_no ))
        [  1   5   9  11  28  39  40  42  44  49 ...
    >>> print(qdcl.get_vector_of_idx_for_label(2, labels, data_tab, probes_no ))
        [  0   2   4   6  10  13  15  17  18  19 ...

    More sophisticated examples of use in a file vqe-for_credit_risk.py
        
    """
    out_idx_list = [] 

    for idx in range(0, _n_samples):
        if l == labels[idx]:
            out_idx_list.append( idx )

    return np.array( out_idx_list )

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

def is_matrix_symmetric( _matrix, _rtolerance=1e-05, _atolerance=1e-08):
    """
    Checks if the square matrix is symmetric that is if a given matrix (M) 
    and its transposition (MT) are equal. The calculation is run including 
    two parameters of the tolerance: relative (rel) and absolute (ab); 
    according to the relation: absolute(M - MT) <= (ab + rel * absolute(MT)).

    Parameters
    ----------
    _matrix : numpy array object
        The matrix to be verified.
    _rtolerance : float
        The relative tolerance parameter. Default value: 1e-05.
    _atolerance : float
        The absolute tolerance parameter. Default value: 1e-08.
    
    Returns
    -------
    bool
        Returns True if the matrix is symmetric within the given tolerance; 
        False otherwise. If analyzed matrix is not square, the value error is
        raised.
    
    Examples
    --------
    When the analyzed matrix is symmetric:
    >>> a=np.array([[1,0],[0,1]])
    >>> print(is_matrix_symmetric(a))
        True
    When the analyzed matrix is not symmetric:
    >>> a=np.array([[1,1],[0,1]])
    >>> print(is_matrix_symmetric(a))
        False
    When the analyzed matrix is not square:
    >>> a=np.array([[1,0,1],[0,1,0]])
    >>> print(is_matrix_symmetric(a))
        ...
        ValueError: The given matrix is not square what is the necessary condition to be symmetric!

    """
    r,c = _matrix.shape
    if r==c:
        return np.allclose( _matrix, _matrix.T, rtol=_rtolerance, atol=_atolerance)
    else:
        raise ValueError("The given matrix is not square what is the necessary condition to be symmetric!")
        return None

def difference_matrix( _rho1, _rho2 ):
    """
    Calculates the difference between given matrices.
    
    Parameters
    ----------
    _rho1 : numpy array object
        The minuend matrix.
    _rho2 : numpy array object
        The subtrahend matrix.
    
    Returns
    -------
    _result_rho : numpy array object
        The difference matrix. If the given matrices (_rho1, _rho2) are not of 
        the same dimensions, the value error will be raised. 
    
    Examples
    --------
    For two matrices of real numbers:
    >>> a=np.array([[1,0],[0,1]])
    >>> b=np.array([[1,1],[1,1]])
    >>> print(difference_matrix(a,b))
        [[ 0 -1]
         [-1  0]]
    For two matrices of complex numbers:
    >>> a=np.array([[1,0,0+1j],[0,0,1]])
    >>> b=np.array([[0,1,0],[0,0+1j,0]])
    >>> print(difference_matrix(a,b))
        [[ 1.+0.j -1.+0.j  0.+1.j]
         [ 0.+0.j  0.-1.j  1.+0.j]]
    When matrices are not of the same dimensions:
    >>> a=np.array([[1,0],[0,1]])
    >>> b=np.array([[1,1,1],[1,1,1]])
    >>> print(difference_matrix(a,b))
        ...
        ValueError: The matrices are not of the same dimensions!

    """
    # dimension check for rho1, rho2
    r1,c1 = _rho1.shape
    r2,c2 = _rho2.shape
    if r1==r2 and c1==c2:
        _result_rho = np.zeros( _rho1.shape )
        _result_rho = _rho1 - _rho2
        return _result_rho
    else:
        raise ValueError("The matrices are not of the same dimensions!")
        return None

def create_covariance_matrix( _qdX ):  
    """
    Calculates the covariance matrix for the given data table with observations
    as normalized quantum states.
    
    Parameters
    ----------
    _qdX : numpy array object
        The data table - each row represents a normalized quantum state.
    
    Returns
    -------
    _covariance_matrix : numpy array object
        The covariance matrix of complex numbers.
    
    Examples
    --------
    The covariance matrix for the exemplary data set CRABS.xlsx limited to 
    4 variables during the data quantum normalization:
    >>> df = pd.read_excel(r'CRABS.xlsx')
    >>> data_tab=convert_data_to_vector_states(df,4)
    >>> print(create_covariance_matrix(data_tab))
        [[ 6.68684747e-05+0.j -1.48683687e-05+0.j  2.69459874e-06+0.j
          -3.73476665e-05+0.j]
         [-1.48683687e-05+0.j  1.88529350e-04+0.j -6.68424455e-05+0.j
          -3.84981355e-05+0.j]
         [ 2.69459874e-06+0.j -6.68424455e-05+0.j  2.89967696e-05+0.j
           1.03818431e-05+0.j]
         [-3.73476665e-05+0.j -3.84981355e-05+0.j  1.03818431e-05+0.j
           3.72404181e-05+0.j]]

    """
    _n_samples = np.shape( _qdX )[0]
    
    scale = (1.0 / (_n_samples - 1.0))
    
    _qXDiffWithMean = _qdX - _qdX.mean(axis=0)
    
    _covariance_matrix = scale * (( _qXDiffWithMean ).T.dot( _qXDiffWithMean ))

    return np.array( _covariance_matrix, dtype=complex )

def create_adjacency_matrix( _qdX, _threshold, _func_distance = None):
    """
    Calculates the adjacency matrix for the given data table with observations
    as normalized quantum states. User points out the type of a distance function
    and the distance threshold to asses if two quantum states are/are not adjacent.
    
    Parameters
    ----------
    _qdX : numpy array object
        The data table - each row represents a normalized quantum state.
    _threshold : float
        The critic distance value which groups observations to adjacent and not
        adjacent.
    _func_distance : function
        The given distance function according to which the distance between 
        quantum states is calculated.
    
    Returns
    -------
    adj_matrix : numpy array object
        The adjacency matrix.
    
    Examples
    --------
    The adjacency matrix for the exemplary data set CRABS.xlsx limited to 
    4 variables during the data quantum normalization; Manhattan distance used
    as the distance measure with the closeness treshold as 0.1:
    >>> df = pd.read_excel(r'CRABS.xlsx')
    >>> data_tab=convert_data_to_vector_states(df,4)
    >>> print(create_adjacency_matrix(data_tab, 0.1, manhattan_distance))
        [[0. 1. 1. ... 1. 1. 1.]
         [1. 0. 1. ... 1. 1. 1.]
         [1. 1. 0. ... 1. 1. 1.]
         ...
         [1. 1. 1. ... 0. 1. 1.]
         [1. 1. 1. ... 1. 0. 1.]
         [1. 1. 1. ... 1. 1. 0.]]

    """
    if _func_distance == None:
        raise ValueError("Distance function has been not assigned!!!") 
    
    rows = _qdX.shape[0]
    # cols = _qdX.shape[1]

    adj_matrix = np.zeros ( shape=(rows, rows) ) 

    for x in range(rows):
        for y in range(rows):
            if x != y and _func_distance(_qdX[x], _qdX[y]) < _threshold:
                adj_matrix[x,y] = 1
            else:
                adj_matrix[x,y] = 0

    return adj_matrix

def create_laplacian_matrix( _adj_matrix ):
    """
    Calculates the Laplacian matrix based on the adjacency matrix for the 
    analyzed data.
    
    Parameters
    ----------
    adj_matrix : numpy array object
        The adjacency matrix.
    
    Returns
    -------
    lap_matrix : numpy array object
        The Laplacian matrix.
    
    Examples
    --------
    The Laplacian matrix for the exemplary data set CRABS.xlsx limited to 
    4 variables during the data quantum normalization; Eucidean distance 
    (with square root) used as the distance measure with the closeness 
    treshold as 0.2 to calculate the adjacency matrix:
    >>> df = pd.read_excel(r'CRABS.xlsx')
    >>> data_tab=convert_data_to_vector_states(df,4)
    >>> adjm=create_adjacency_matrix(data_tab, 0.2, euclidean_distance_with_sqrt)
    >>> print(create_laplacian_matrix( adjm ))
        [[199.  -1.  -1. ...  -1.  -1.  -1.]
         [ -1. 199.  -1. ...  -1.  -1.  -1.]
         [ -1.  -1. 199. ...  -1.  -1.  -1.]
         ...
         [ -1.  -1.  -1. ... 199.  -1.  -1.]
         [ -1.  -1.  -1. ...  -1. 199.  -1.]
         [ -1.  -1.  -1. ...  -1.  -1. 199.]]

    """
    lap_matrix = np.zeros( shape=(_adj_matrix.shape[0], 
                                  _adj_matrix.shape[1]) ) 
    rows = _adj_matrix.shape[0]
    for x in range(rows):
        for y in range(rows):
            if _adj_matrix[x,y] == 1:
                lap_matrix[x,y] = -1
        lap_matrix[x,x] = np.sum(_adj_matrix[x])
    
    return lap_matrix

# similiar solution with upper traingular matrix
# can be also found at:
# https://www.mathworks.com/matlabcentral/fileexchange/24661-graph-adjacency-matrix-to-incidence-matrix        
def create_incidence_matrix( _adj_matrix ):
    """
    Calculates the incidence matrix based on the adjacency matrix for the 
    analyzed data.
    
    Parameters
    ----------
    adj_matrix : numpy array object
        The adjacency matrix.
    
    Returns
    -------
    m_incidence : numpy array object
        The incidence matrix.
    
    Examples
    --------
    The incidence matrix for the exemplary data set CRABS.xlsx limited to 
    4 variables during the data quantum normalization; Eucidean distance 
    (with square root) used as the distance measure with the closeness 
    treshold as 0.2 to calculate the mincidence matrix:
    >>> df = pd.read_excel(r'CRABS.xlsx')
    >>> data_tab=convert_data_to_vector_states(df,4)
    >>> adjm=create_adjacency_matrix(data_tab, 0.2, euclidean_distance_with_sqrt)
    >>> print(create_incidence_matrix( adjm ))
        [[1. 1. 0. ... 0. 0. 0.]
         [1. 0. 1. ... 0. 0. 0.]
         [1. 0. 0. ... 0. 0. 0.]
         ...
         [0. 0. 0. ... 1. 1. 0.]
         [0. 0. 0. ... 1. 0. 1.]
         [0. 0. 0. ... 0. 1. 1.]]

    """
    if is_matrix_symmetric(_adj_matrix) == True:
        n_vertices = _adj_matrix.shape[0]
        inds = np.argwhere( np.triu(_adj_matrix) )
        n_edges = inds[:,0].shape[0]
        edges_idx =  np.hstack( (np.arange(n_edges), np.arange(n_edges)) )
        vertices_idx = np.hstack( (inds[:,0], inds[:,1]) )
        
        m_incidence = np.zeros( shape=(n_edges, n_vertices) )
        
        vidx=0
        for eidx in edges_idx:
                m_incidence[eidx, vertices_idx[vidx]] = 1
                vidx = vidx + 1
        return m_incidence
    else:
        return None    
 
def create_float_table_zero_filled( _n_samples ):
    """
    Generates _n_samples-element array of zeros as float numbers.
    
    Parameters
    ----------
    _n_samples : integer
        The number of elements in the array.
    
    Returns
    -------
    ck_tbl : numpy array
        The array of floats, filled with zeros.
    
    Example
    -------
    The generation of a 10-element array:
    >>> qdcl.create_float_table_zero_filled( 10 )
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

    """
    ck_tbl = np.zeros( shape = (_n_samples,), dtype=float )
    
    return ck_tbl

def create_ck_table_zero_filled( _n_samples ):
    """
    Generates _n_samples-element array of zeros as integer numbers.
    
    Parameters
    ----------
    _n_samples : integer
        The number of elements in the array.
    
    Returns
    -------
    _ck_tbl : numpy array
        The array of integers, filled with zeros.
    
    Example
    -------
    The generation of a 10-element array:
    >>> qdcl.create_ck_table_zero_filled( 10 )
    [0 0 0 0 0 0 0 0 0 0]

    """
    _ck_tbl = np.zeros( shape = (_n_samples,), dtype=np.int64 )
    
    return _ck_tbl

#
# TO DESC
#
def random_assign_clusters_to_ck(_n_samples, _n_clusters):
    
    _rng = np.random.default_rng()
    
    _ck = _rng.integers(_n_clusters, size=_n_samples, dtype=np.int64)
    
    return _ck
 
#
# TO DESC
#
def create_initial_centroids(_qdX, _n_samples, _n_clusters):
    
    rng = np.random.default_rng()
    
    idx_table = rng.integers(_n_samples, size=_n_clusters)
    
    _local_ck = np.zeros( (_n_clusters, _qdX.shape[1]) )
    
    idx_local_ck=0
    for idx in idx_table:
        _local_ck[idx_local_ck] = _qdX[idx]
        idx_local_ck = idx_local_ck + 1
        

    return _local_ck
    
    
def get_indices_for_cluster_k(_ck, _k):
    """
    Calculates the coordinates of points belonging to the class _k.

    Parameters
    ----------
    _ck : numpy ndarray
        The array of elements as ordinal numbers of classes for each observation.
    _k : interger
        The number of the class.

    Returns
    -------
    tuple
        The coordinates of points belonging to the class _k.

    """
    return np.where( _ck == _k)


def is_probe_n_in_cluster_k( _ck, _n, _k, ):
    """
    Checks if an observation (probe) indexed as n, belongs to the cluster 
    denoted as _k.

    Parameters
    ----------
    _ck : numpy ndarray
        The array of elements as ordinal numbers of classes for each observation.
    _n : integer
        A number of the observation (probe).
    _k : interger
        The number of the class.

    Returns
    -------
    val: Boolean
        Returns True if observation n belongs to the cluster denoted as _k and 
        False otherwise.

    """
    val = None
    if _ck[_n] == _k:
        val = True
    else:
        val = False
    return val

def number_of_probes_in_cluster_k( _ck, _k ):
    """
    Calculates a number of observations (probes) belonging to the cluster 
    denoted as _k.

    Parameters
    ----------
    _ck : numpy ndarray
        The array of elements as ordinal numbers of classes for each observation.
    _k : interger
        The number of the class.

    Returns
    -------
    integer
        The number of observations (probes) belonging to the pointed cluster.

    """
    return (_ck == _k).sum() 

def number_of_probes_for_class( _labels, _class ):
    """
    A variant of the function number_of_probes_in_cluster_k. We calculate a number
    of observations (probes) belonging to the class denoted as the parameter _class.

    Parameters
    ----------
    _labels : numpy ndarray
        The array of elements as ordinal numbers of classes for each observation.
    _class : integer
        The number of the class.

    Returns
    -------
    integer
        The number of observations (probes) belonging to the pointed class.


    """
    return number_of_probes_in_cluster_k( _labels, _class)

#
# TO DESC
#
def quantum_kmeans_clusters_assignment(_qdX, _centroids, _n_samples, _n_clusters,  _func_distance=None):
    
    new_ck = create_ck_table_zero_filled( _n_samples )
    
    distance_table = np.zeros( shape=(_n_samples, _n_clusters))
    
    for _n in range(_n_samples):
        for _k in range(_n_clusters):
            if _func_distance==None:
                distance_table[ _n, _k] = np.linalg.norm( _qdX[_n] - _centroids[_k] ) ** 2.0
            else:
                distance_table[ _n, _k] = _func_distance( _qdX[_n], _centroids[_k] )
            
    for _n in range(_n_samples):
        new_ck[_n] = np.argmin(distance_table[_n])
            
    return distance_table, new_ck
#
# TO DESC
#
def quantum_kmeans_update_centroids(_qdX, _ck, _n_samples, _n_clusters):
    
    _centroids = np.zeros( shape=(_n_clusters, _qdX.shape[1]) )
            
    # centroid update
    chi_vector = np.zeros( shape=(_n_samples, 1) )
    
    idx=0
    for _k in range(_n_clusters):
        
        probes_in_cluster = get_indices_for_cluster_k(_ck, _k)[0].shape[0]
        
        for _n in range(_n_samples):
            if is_probe_n_in_cluster_k(_ck, _n, _k) == True:
                chi_vector[ _n ] = 1.0  / np.sqrt( probes_in_cluster )
            else:
                chi_vector[ _n ] = 0.0
        
        _centroids[idx] = ((_qdX.T @ chi_vector).T) 
        
        # we rescale _centroids
        _centroids[idx] = _centroids[idx] / np.linalg.norm(_centroids[idx])
        
        idx = idx+1
        
    return _centroids

#
# TO DESC
#
def quantum_kmeans_assign_labels( _qdX, _centroids, _n_samples, _n_clusters, _func_distance=None ):
    
    _, _labels = quantum_kmeans_clusters_assignment( _qdX,_centroids, 
                                                     _n_samples, 
                                                     _n_clusters, 
                                                     _func_distance)
    
    return _labels

#
# TO DESC
#
def quantum_kmeans( _qdX, _n_samples, _n_clusters, _max_iteration=128, _func_distance=None):
    
    _ck = random_assign_clusters_to_ck( _n_samples, _n_clusters )
   
    _centroids = create_initial_centroids( _qdX, _n_samples, _n_clusters)

    _iteration=0
    
    while _iteration < _max_iteration:    
        _old_ck = _ck.copy()        
        
        _, _ck = quantum_kmeans_clusters_assignment(_qdX,
                                                 _centroids, 
                                                 _n_samples, 
                                                 _n_clusters,
                                                 _func_distance )

        _centroids = quantum_kmeans_update_centroids( _qdX, _ck, _n_samples, _n_clusters )

        if all(_old_ck == _ck):
            break
  
        _iteration = _iteration + 1
    
    return _ck, _centroids

#
# TO DESC
#
def classic_spectral_clustering(_qdX, _n_samples, _n_clusters, _threshold, _func_distance=None ):
    """
    

    Parameters
    ----------
    _qdX : TYPE
        DESCRIPTION.
    _n_samples : TYPE
        DESCRIPTION.
    _n_clusters : TYPE
        DESCRIPTION.
    _threshold : TYPE
        DESCRIPTION.
    _func_distance : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    _labels : TYPE
        DESCRIPTION.

    """
    
    adj_matrix = create_adjacency_matrix( _qdX, _threshold, _func_distance )

    lap_matrix = create_laplacian_matrix( adj_matrix )

    evalues, evectors = np.linalg.eig( lap_matrix )
    
    A = np.zeros( shape=(_n_samples, _n_clusters) )

    indicies = np.argpartition(evalues,_n_clusters)[:_n_clusters]
    idx=0
    for i in indicies:
        A[:, idx] = evectors[:, i]
        idx = idx + 1
        
    kmeans = KMeans(n_clusters=_n_clusters, random_state=1234, n_init="auto").fit(A)
    _labels = kmeans.labels_
    # _centers = kmeans.cluster_centers_
    
    # rather for normalised data
    # _labels, _ = kmeans_spherical( A, _n_clusters, _func_distance=_func_distance )
    
    return _labels

#
# TO DESC
#
def create_rho_state_for_qsc(_qdX, _n_samples, _n_clusters, _threshold, _func_distance=None ):
    """
    

    Parameters
    ----------
    _qdX : TYPE
        DESCRIPTION.
    _n_samples : TYPE
        DESCRIPTION.
    _n_clusters : TYPE
        DESCRIPTION.
    _threshold : TYPE
        DESCRIPTION.
    _func_distance : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    rho : TYPE
        DESCRIPTION.

    """
    
    adj_matrix = create_adjacency_matrix( _qdX, _threshold, _func_distance )
   
    lap_matrix = create_laplacian_matrix( adj_matrix )
   
    evalues, evectors = np.linalg.eig( lap_matrix )
    
    A = np.zeros( shape=(_n_samples, _n_clusters) )
   
    indicies = np.argpartition( evalues, _n_clusters )[ :_n_clusters ]
    idx = 0
    for ind in indicies:
        A[:, idx] = evectors[:, ind]
        idx = idx + 1

    rho = (1.0/_n_clusters) * (A @ A.T)

    return rho

#
# TO DESC
#
def quantum_spectral_clustering(_qdX, _n_samples, _n_clusters, _threshold, _func_distance=None ):
    """
    

    Parameters
    ----------
    _qdX : TYPE
        DESCRIPTION.
    _n_samples : TYPE
        DESCRIPTION.
    _n_clusters : TYPE
        DESCRIPTION.
    _threshold : TYPE
        DESCRIPTION.
    _func_distance : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    labels : TYPE
        DESCRIPTION.
    projectors : TYPE
        DESCRIPTION.

    """
    
    rho = create_rho_state_for_qsc(_qdX, _n_samples, _n_clusters, _threshold, _func_distance)
    
    prj_evals, prj_evectors = np.linalg.eig( rho )
    
    eigenvalues_of_rho = np.zeros( shape=(_n_clusters, ) )
    projectors_of_rho = np.zeros( shape=(_n_samples, _n_clusters) )
    
    indicies = np.argpartition( prj_evals, -_n_clusters )[ -_n_clusters: ]
    idx = 0
    for i in indicies:
        eigenvalues_of_rho[idx] = prj_evals[i]
        projectors_of_rho[:, idx] = prj_evectors[:, i]
        idx = idx + 1
    
    projectors_of_rho_abs = abs(projectors_of_rho)

    projectors = np.zeros( shape=(_n_samples, _n_samples, _n_clusters) )

    labels=np.zeros( shape=(_n_samples,), dtype=int )
    for n in range(0, _n_samples):
            labels[n]=projectors_of_rho_abs[n].argmax()

    for cidx in range(0, _n_samples):
        for n in range(0, _n_samples):
            cind = labels[n]
            if cidx == cind:
                projectors[ n, n, cidx ] = 1.0
    
    return labels, projectors

#
# TO DESC
#
def hc_create_distance_matrix( _qdX, _n_samples, _func_distance=None ):
    pass

#
# TO DESC
#
def hc_complete_linkage_clustering(_qdX, _a, _b):
    pass

#
# TO DESC
#
def hc_single_linkage_clustering(_qdX, _a, _b):
    pass

#
# TO DESC
#
def hierarchical_clustering_for_quantum_data(_qdX, _n_samples, _n_clusters, _threshold, _func_distance=None ):
    pass



#
# TO DESC
#
def version():
    pass

#
# TO DESC
#
def about():
    pass

#
# TO DESC
#
def how_to_cite():
    pass
