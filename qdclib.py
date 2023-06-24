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

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

import numpy as np
import pandas as pd
import math as math
import sympy as sympy

COSINE_DISTANCE = 1000
DOT_DISTANCE = 1001
FIDELITY_DISTANCE = 1002
TRACE_DISTANCE = 1003

# klasy wyjątków z EntDetectora

class DimensionError(Exception):
    """DimensionError"""
    def __init__(self, message):
        self.message = message

class ArgumentValueError(Exception):
    """ArgumentValueError"""
    def __init__(self, message):
        self.message = message

class DensityMatrixDimensionError(Exception):
    """DensityMatrixDimensionError"""
    def __init__(self, message):
        self.message = message


class BlochVisualization:

    def __init__( self,  background=False, font_size=16 ):
        pass

    def show ( self ):
        pass
    
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



def cosine_distance( uvector, vvector ):
    """
    Calculate a cosine distance between two vectors

    Parameters
    ----------
    uvector : TYPE
        DESCRIPTION.
    vvector : TYPE
        DESCRIPTION.

    Returns
    -------
    distance_value : TYPE
        DESCRIPTION.

    """
    distance_value = 1.0 - np.dot(uvector, vvector) / ( np.linalg.norm(uvector) * np.linalg.norm(vvector) )
    distance_value = np.linalg.norm(distance_value)
    return distance_value

def dot_product_as_distance( uvector, vvector ):
    return 1.0 - np.linalg.norm( np.vdot( vvector, uvector ) )

def fidelity_as_distance( uvector, vvector ):
    """
    
    for two pure states

    Parameters
    ----------
    uvector : TYPE
        DESCRIPTION.
    vvector : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return 1.0 - ( np.linalg.norm( np.vdot( vvector, uvector ) ) ** 2.0 )

def trace_distance( uvector, vvector ):
    """
    for pure states

    Parameters
    ----------
    uvector : TYPE
        DESCRIPTION.
    vvector : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return np.sqrt( 1.0 - ( np.linalg.norm( np.vdot( vvector, uvector ) ) ** 2.0 ) )

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
    
    _vector_zero = np.zeros( (_n_dim) )
    
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
    
    _vector_one = np.zeros( (_n_dim) )
    _vector_one[ _axis ] = 1.0  
    
    return _vector_one


def create_spherical_probes( _n_points, _n_dim=3):
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

def create_focused_spherical_probes_2d( _n_points, _n_focus_points, _width_of_cluster=0.1):
    # a tu chodzi oto ze owszem losujemy punkty
    # ale już domylnie skupione wokól kilku puntków,
    # choć zakładamy że same punkty będą wylosowanane
    
    theta=0
    theta_delta = (2.0 * np.pi) / _n_focus_points
    centers_on_circle=[]
    for i in range(_n_focus_points):
        theta=theta+theta_delta
        x=np.sin(theta)
        y=np.cos(theta)  
        centers_on_circle.append((x,y))
    
    d, labels = make_blobs( n_samples=_n_points, n_features=2, centers=centers_on_circle, cluster_std=_width_of_cluster )
    
    for i in range(_n_points):
        d[i] = d[i] / np.linalg.norm(d[i])
    
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

    closest, centers = kmeans_spherical( _qX, _n_clusters, _max_iterations, _funcdist )
        
    return closest, centers 

def calculate_distance(_data, _vector, _func_distance):
    distance_table=np.zeros( shape=(_data.shape[0] ) )
    idx=0
    for e in _data:
        distance_table[idx] = _func_distance(e, _vector)
        #distance_table[idx, 1] = l
        idx=idx+1
    
    return distance_table
    

def create_distance_table( _data, _centers, _labels, _n_clusters, _func_distance=None ):
    idx=0
    distance_table=np.zeros( shape=(_data.shape[0], 2) )
    for l in range(0, _n_clusters):
        cntr=_centers[l]
        for e in _data[_labels==l]:
            distance_table[idx, 0] = _func_distance(e, cntr)
            distance_table[idx, 1] = l
            idx=idx+1
    
    return distance_table

def get_distances_for_cluster( _data, _n_cluster ):
    return _data[ _data[:,1] == _n_cluster ]

