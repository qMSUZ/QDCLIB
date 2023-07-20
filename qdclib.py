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

COSINE_DISTANCE    = 1000
DOT_DISTANCE       = 1001
FIDELITY_DISTANCE  = 1002
TRACE_DISTANCE     = 1003
MANHATTAN_DISTANCE = 1004
BURES_DISTANCE     = 1005
HS_DISTANCE        = 1006

POINTS_DRAW        = 2000
LINES_DRAW         = 2001


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
    if isinstance(expr, (int, float, complex)):
        return 0 if -delta <= expr <= delta else expr
    else:
        return [_internal_chop(x) for x in expr]


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
        
        0 <= _theta <= np.pi 
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

class BlochVectorsTable:
    pass

class PureStatesTable:
    pass

class BlochVisualization:

    def __init__( self ):
        
        self.additional_points = []
        self.additional_states = []
        self.additional_vectors = []
        
        self.radius = 2.0
        self.resolution_of_mesh = 31
        
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

        self.draw_mode = 0

    def make_figure( self ):
        
        self.prepare_mesh()
        f = self.render_bloch_sphere()
        
        return f
    
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

    def enable_draw_points( self ):
        self.draw_mode = POINTS_DRAW

    def set_points(self, _points=None):
        self.additional_points = _points.copy()
        
        for row in range(0, self.additional_points.shape[0]):
            # normalization points
            self.additional_points[row] /= np.linalg.norm(self.additional_points[row])
            self.additional_points[row] *= (self.radius + 0.01)
            
        # rescale to radius r
       
    def clear_points(self):
        self.additional_points = [ ]

    def add_points(self, _points=None, _color=None):
        # normalise points
        # rescale to radius r
        pass
    
    def set_vectors(self, _points=None):
        pass

    def clear_vectors(self):
        self.additional_vectors = [ ]

    def add_vectors(self, _points=None, _color=None):
        pass

    def set_pure_states(self, _states=None, _color=None):
        ptns = np.empty((0,3))
        for qstate in _states:
            qstateden = _internal_qdcl_vector_state_to_density_matrix( qstate )
            
            # change sign for x coords
            xcoord = - np.trace( _internal_pauli_x() @ qstateden )
            ycoord =   np.trace( _internal_pauli_y() @ qstateden )
            zcoord =   np.trace( _internal_pauli_z() @ qstateden )
        
            ptns = np.append( ptns, [[ xcoord, ycoord, zcoord]], axis=0)  # for state
    
        self.set_points( ptns )
        
    def enable_pure_states_draw( self ):
        self.draw_mode = POINTS_DRAW        

    def clear_pure_states(self):
        self.additional_states = [ ]

    def add_pure_states(self, _states=None):
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
        # axis needs reorganisation
        self.axes.scatter(
            np.real(self.additional_points[:,1]),
            np.real(self.additional_points[:,0]),
            np.real(self.additional_points[:,2]),
            s=200,
            alpha=1,
            edgecolor=None,
            zdir="z",
            color="green",
            marker=".",
        )
        #pass
    
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

        if self.draw_mode == POINTS_DRAW:
            self.render_points()


        self.render_equator_and_parallel()

        self.render_sphere_axes()

        self.render_labels_for_axes()


        return self.figure
    
    def save_to_file(self, filename = None):
        pass

def create_circle_plot_for_2d_data(_qX):
    # shape _qX to check

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    circle = plt.Circle( (0,0), 1,  color='r', fill=False)
    ax.scatter( _qX[:,0], _qX[:,1])
    ax.add_patch(circle)
    
    return fig

def create_circle_plot_with_centers_for_2d_data(_qX, _n_clusters, _centers, _labels):
    # shape _qX to check
    
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
    intervals=maxs-mins
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
            
        return distance_value
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
 

def bures_distance( uvector, vvector, r=0 ):
    """
    Caclutales the Bures distance between two pure states.

    Parameters
    ----------
    uvector, vvector : numpy array objects
        Vectors of complex numbers describing quantum states.
    r : integer
        The number of decimals to use while rounding the number (default is 0,
        i.e. the number is not rounded).

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
    >>> v=np.array([1/math.sqrt(2),0 + 1j/math.sqrt(2)]])
    >>> u=np.array([1/math.sqrt(2),0 - 1j/math.sqrt(2)]])
    >>> print(bures_distance(u, v))
        2.0
    A distance between two examplary states:
    >>> v=np.array([1/math.sqrt(2),1/math.sqrt(2)])
    >>> u=np.array([1/math.sqrt(2),0 + 1j/math.sqrt(2)])
    >>> print(bures_distance(u, v, 3))
        0.586

    """
    if r==0:
        rslt = 2.0 - 2.0 * math.sqrt( fidelity_measure(uvector, vvector, r) )
    else:
        rslt = round( ( 2.0 - 2.0 * math.sqrt( fidelity_measure(uvector, vvector, r) ) ), r )
    
    return rslt

def hs_distance( uvector, vvector, r=0 ):
    """
    Caclutales the Hilbert-Schmidt distance between two pure states.

    Parameters
    ----------
    uvector, vvector : numpy array objects
        Vectors of complex numbers describing quantum states.
    r : integer
        The number of decimals to use while rounding the number (default is 0,
        i.e. the number is not rounded).

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
    >>> v=np.array([1/math.sqrt(2),0 + 1j/math.sqrt(2)]])
    >>> u=np.array([1/math.sqrt(2),0 - 1j/math.sqrt(2)]])
    >>> print(hs_distance(u, v, 5))
        1.41421
    A distance between two examplary states:
    >>> v=np.array([1/math.sqrt(2),1/math.sqrt(2)])
    >>> u=np.array([1/math.sqrt(2),0 + 1j/math.sqrt(2)])
    >>> print(hs_distance(u, v, 3))
        1.0

    """
    qu=_internal_qdcl_vector_state_to_density_matrix(uvector)
    qv=_internal_qdcl_vector_state_to_density_matrix(vvector)
    rp=sympy.re(np.trace(np.subtract(qu,qv) @ np.subtract(qu,qv)))
    if r==0:
        rslt=math.sqrt(rp)
    else:
        rslt=round(math.sqrt(rp),r)
    return rslt


def trace_distance( uvector, vvector, r=0 ):
    """
    Calculates the distance based on density matrix trace of two pure states.

    Parameters
    ----------
    uvector, vvector : numpy array objects
        Vectors of complex numbers describing quantum states.
    r : integer
        The number of decimals to use while rounding the number (default is 0,
        i.e. the number is not rounded).

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

    """
    if r==0:
        rslt = np.sqrt( 1.0 - ( np.linalg.norm( np.vdot( vvector, uvector ) ) ** 2.0 ) )
    else:
        rslt = round( np.sqrt( 1.0 - ( np.linalg.norm( np.vdot( vvector, uvector ) ) ** 2.0 ) ), r )
    
    return rslt

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

def create_focused_qubits_probes_with_uniform_placed_centers( _n_points, _n_theta, _n_psi, _width_of_cluster=0.1, return_labels = False, return_centers = False ):
    centers_on_sphere = [ ]
    
    for i in range( _n_theta ):
        for j in range( _n_psi+1 ):
            _theta = 2.0 * np.pi * (i/_n_theta)
            _psi = (np.pi/2.0) - np.pi*(j/_n_psi)
            sp = convert_spherical_point_to_bloch_vector(1.0, _theta, _psi)
            centers_on_sphere.append( ( sp[0], sp[1], sp[2]) ) 

    d, labels = make_blobs( n_samples=_n_points,
                       n_features=3,
                       centers = centers_on_sphere,
                       cluster_std=_width_of_cluster )

    for i in range(_n_points):
        d[i] = d[i] / np.linalg.norm(d[i])
    
    
    if return_labels==True and return_centers==False:
        return d, labels

    if return_labels==True and return_centers==True:
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
        _funcdist = trace_distance

    if _func_distance==MANHATTAN_DISTANCE:
        _funcdist = manhattan_distance
    
    if _func_distance==BURES_DISTANCE:
        _funcdist =bures_distance
    
    if _func_distance==HS_DISTANCE:
        _funcdist = hs_distance 
    
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
    labels=np.zeros(samples)
    for l in range(_n_clusters):
        for v in current_clusters[l]:
            idx=np.where(v == _qX)[0][0]
            labels[idx]=l

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
        _funcdist = trace_distance

    if _func_distance==MANHATTAN_DISTANCE:
        _funcdist = manhattan_distance
    
    if _func_distance==BURES_DISTANCE:
        _funcdist =bures_distance
    
    if _func_distance==HS_DISTANCE:
        _funcdist = hs_distance 
    

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


def version():
    pass

def about():
    pass

def how_to_cite():
    pass
