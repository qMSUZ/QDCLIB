#! /usr/bin/env python3
# -*- coding: utf-8 -*-

#/***************************************************************************
# *   Copyright (C) 2020 -- 2022 by Marek Sawerwain                         *
# *                                  <M.Sawerwain@gmail.com>                *
# *                                  <M.Sawerwain@issi.uz.zgora.pl>         *
# *                                                                         *
# *                              by Joanna Wiśniewska                       *
# *                                  <Joanna.Wisniewska@wat.edu.pl>         *
# *                                                                         *
# *                              by Marek Wróblewski                        *
# *                                  <M.Wroblewski@issi.uz.zgora.pl>        *
# *                                                                         *
# *                              by Roman Gielerak                          *
# *                                  <R.Gielerak@issi.uz.zgora.pl>          *
# *                                                                         *
# *   Part of the EntDetector:                                              *
# *         https://github.com/qMSUZ/EntDetector                            *
# *                                                                         *
# *   This program is free software; you can redistribute it and/or modify  *
# *   it under the terms of the GNU General Public License as published by  *
# *   the Free Software Foundation; either version 3 of the License, or     *
# *   (at your option) any later version.                                   *
# *                                                                         *
# *   This program is distributed in the hope that it will be useful,       *
# *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
# *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
# *   GNU General Public License for more details.                          *
# *                                                                         *
# *   You should have received a copy of the GNU General Public License     *
# *   along with this program; if not, write to the                         *
# *   Free Software Foundation, Inc.,                                       *
# *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
# ***************************************************************************/


"""
First version created on Sat Nov 21 18:33:49 2020

@author: Marek Sawerwain, Joanna Wiśniewska, Marek Wróblewski, Roman Gielerak
"""

import numpy as np
import sympy
import scipy
import cvxpy

import random as rd

import math
import itertools

from exceptions_classes import *

# smallest value for entropy calculations
precision_for_entrpy_calc = 0.00001

ENTDETECTOR_version_major       = 0
ENTDETECTOR_version_minor       = 5
ENTDETECTOR_version_patch_level = 0

# code based on chop
# discussed at:
#   https://stackoverflow.com/questions/43751591/does-python-have-a-similar-function-of-chop-in-mathematica
def chop(expr, delta=10 ** -10):
    if isinstance(expr, (int, float, complex)):
        return 0 if -delta <= expr <= delta else expr
    else:
        return [chop(x) for x in expr]

#
# basic state creation
#
def create_base_state(d, n, base_state):
    """
        Create a base state

        Parameters
        ----------
        d : integer
            the number of degrees of freedom for the qudit d, d = 2

        n : integer
            number of qudits for the created state

        base_state: integer
            the base state that the created qubit will take

        Returns
        -------
        quantum state : numpy vector
            Numpy vector for quantum state

        Examples
        --------
        Create of register with one qubit that becomes zero state
        >>> q0=create_base_state(2, 1, 0)
        >>> print(q0)
        [1. 0.]
    """
    v = np.zeros(d ** n)
    v[base_state] = 1
    return v

def create_pure_state(d, n, base_state):
    """
        Create a pure state

        Parameters
        ----------
        d : integer
            the number of degrees of freedom for the qudit d, d = 2

        n : integer
            number of qudits for the created state

        base_state: integer
            the base state that the created qubit will take

        Returns
        -------
        quantum state : numpy vector
            Numpy vector for quantum state

        Examples
        --------
        Create of registry with one qubit that is a pure state
        >>> q0=create_pure_state(2, 1, 0)
        >>> print(q0)
        [1. 0.]
    """
    return create_base_state(d, n, base_state)

def create_qubit_zero_state():
    """
        Create a zero state qubit

        Parameters
        ----------
        None

        Returns
        -------
        quantum state : numpy vector
            Numpy vector for quantum state

        Examples
        --------
        Create of register with one qubit for zero state
        >>> q0=create_qubit_zero_state()
        >>> print(q0)
        [1. 0.]
    """
    v = np.zeros(2)
    v[0] = 1.0
    return v

def create_qubit_one_state():
    """
        Create a one state qubit

        Parameters
        ----------
        None

        Returns
        -------
        quantum state : numpy vector
            Numpy vector for quantum state

        Examples
        --------
        Create of register for one state qubit
        >>> q0=create_qubit_one_state()
        >>> print(q0)
        [0. 1.]
    """
    v = np.zeros(2)
    v[1] = 1.0
    return v

def create_qubit_plus_state():
    """
        Create a plus state qubit

        Parameters
        ----------
        None

        Returns
        -------
        quantum state : numpy vector
            Numpy vector for quantum state

        Examples
        --------
        Create of register with one qubit which becomes plus state
        >>> q0=create_qubit_plus_state()
        >>> print(q0)
        [0.70710678 0.70710678]
    """
    v = np.zeros(2)
    v[0] = 1.0 / np.sqrt(2)
    v[1] = 1.0 / np.sqrt(2)
    return v

def create_qubit_minus_state():
    """
        Create a minus state qubit

        Parameters
        ----------
        None

        Returns
        -------
        quantum state : numpy vector
            Numpy vector for quantum state

        Examples
        --------
        Create of register with one qubit which becomes minus state
        >>> q0=create_qubit_minus_state()
        >>> print(q0)
        [ 0.70710678 -0.70710678]
    """
    v = np.zeros(2)
    v[0] =   1.0 / np.sqrt(2)
    v[1] = - 1.0 / np.sqrt(2)
    return v

def create_qutrit_state(base_state):
    """
        Create a qutrit state

        Parameters
        ----------
        base_state: integer
            the base state that the created qubit will take

        Returns
        -------
        quantum state : numpy vector
            Numpy vector for quantum state

        Examples
        --------
        Create of register for qutrit state
        >>> q0=create_qutrit_state(0)
        >>> print(q0)
        [1. 0. 0.]
    """
    v = np.zeros(3)
    v[base_state] = 1.0
    return v

def create_qutrit_zero_state():
    """
        Create a qutrit zero state

        Parameters
        ----------
        None

        Returns
        -------
        quantum state : numpy vector
            Numpy vector for quantum state

        Examples
        --------
        Create of register for one qutrit which becomes zeros state
        >>> q0=create_qutrit_zero_state()
        >>> print(q0)
        [1. 0. 0.]
    """
    v = np.zeros(3)
    v[0] = 1.0
    return v

def create_qutrit_one_state():
    """
        Create a qutrit with state |1>

        Parameters
        ----------
        None

        Returns
        -------
        quantum state : numpy vector
            Numpy vector for quantum state

        Examples
        --------
        Create of state |1> register for one qutrit
        >>> q0=create_qutrit_one_state()
        >>> print(q0)
        [0. 1. 0.]
    """
    v = np.zeros(3)
    v[1] = 1.0
    return v

def create_qutrit_two_state():
    """
        Create a qutrit with state |2>

        Parameters
        ----------
        None

        Returns
        -------
        quantum state : numpy vector
            Numpy vector for quantum state

        Examples
        --------
        Create of state |2> register for one qutrit
        >>> q0=create_qutrit_two_state()
        >>> print(q0)
        [0. 0. 1.]
    """
    v = np.zeros(3)
    v[2] = 1.0
    return v

def create_qutrit_plus_state():
    """
        Create a qutrit plus state

        Parameters
        ----------
        None

        Returns
        -------
        quantum state : numpy vector
            Numpy vector for quantum state

        Examples
        --------
        Create of qutrit plus state register for one qutrit
        >>> q0=create_qutrit_plus_state()
        >>> print(q0)
        [0.57735027 0.57735027 0.57735027]
    """
    v = np.ones(3)
    v[0] = 1.0/np.sqrt(3.0)
    v[1] = 1.0/np.sqrt(3.0)
    v[2] = 1.0/np.sqrt(3.0)
    return v

def create_qubit_bell_state(minus=0):
    """
        Create a qubit bell state

        Parameters
        ----------
            minus : integer 
                additional parameters for minus amplitude
        Returns
        -------
        quantum state : numpy vector
            Numpy vector for quantum state

        Examples
        --------
        Create of qubit bell state register for one qubit:
        >>> q0=create_qubit_bell_state()
        >>> print(q0)
        [0.70710678  0.         0.         0.70710678]

        Create of qubit bell state (with minus amplitude)
        register for one qubit:
        >>> q0=create_qubit_bell_state(1)
        >>> print(q0)
        [0.70710678  0.          0.         -0.70710678]

    """
    d = 2
    n = 2
    v = np.zeros(d ** n)
    v[0] = 1.0 / np.sqrt(2)
    if minus == 1:
        v[(d ** n) - 1] = -1.0 / np.sqrt(2)
    else:
        v[(d ** n) - 1] =  1.0 / np.sqrt(2)
    return v

#
# TO DOC GEN
#
def create_two_qubit_bell_state_non_maximal( ratio = 0.5, minus=0 ):
    
    #  0.0 <= ratio <= 1.0
    
    d = 2
    n = 2

    last_idx = (d ** n) - 1
    q = create_qubit_bell_state( minus )

    q[ 0 ] = ratio * q[ 0 ]
    q[ last_idx ] = (1.0-ratio) * q[ last_idx ]
    
    q=q/np.linalg.norm(q)
    
    return q

def create_mixed_state(d,n):
    """
        Create a mixed state

        Parameters
        ----------
        d : integer
            the number of degrees of freedom for the qudit d, d = 2

        n : integer
            number of qudits for the created state

        base_state: integer
            the base state that the created qubit will take

        Returns
        -------
        density matrix
            Numpy array for quantum state
            expressed as density matrix

        Examples
        --------
        Create of mixed state register for one qubit
        >>> q0=create_mixed_state(2, 1)
        >>> print(q0)
        [[0.5 0. ]
         [0.  0.5]]
    """
    qden = np.eye(d ** n) / (d ** n)
    return qden

#"""
#state |00..0> +  |kkk...k>
#where k = d - 1 and d is demension of single qudit of quantum register
#with n equally dimensional qudits
#"""
def create_0k_stat(d, n):
    """
        Create a 0K state (internal function)

        Parameters
        ----------
        d : integer
            the number of degrees of freedom for the qudit d, d = 2

        n : integer
            number of qudits for the created state

        Returns
        -------
        quantum state : numpy vector
            Numpy vector for quantum state

        Examples
        --------
        Create of |+> for one qubit
        >>> q0=create_0k_stat(2, 1)
        >>> print(q0)
        [0.70710678 0.70710678]

        Create of 1.0/sqrt(2.0)(|00> + |11>) for one qubit
        >>> q0=create_0k_stat(2, 1)
        >>> print(q0)
        [0.70710678 0.         0.         0.70710678]
    """
    v = np.zeros(d ** n)
    v[0] = 1.0/np.sqrt(2)
    v[-1] = v[0]
    return v

def create_max_entangled_pure_state(d):
    """
        Create a maximum entangled of pure state

        Parameters
        ----------
        d : integer
            the number of degrees of freedom for the qudit d, d = 2

        Returns
        -------
        quantum state : numpy vector
            Numpy vector for quantum state

        Examples
        --------
        Create of register for maximum entangled of pure state
        for two qubits:
        >>> q0=create_max_entangled_pure_state(2)
        >>> print(q0)
        [0.70710678 0.         0.         0.70710678]
    """
    v = np.reshape( np.eye(d), d**2 )
    v = v / np.sqrt( d )
    return v

def create_bes_horodecki_24_state(b):
    """
        Create a Horodecki's 2x4 of entangled state

        Parameters
        ----------
        b : real
            the entangled state with a parameter b

        Returns
        -------
        density matrix : numpy array
            Numpy array gives the Horodecki's two-qudit states

        Examples
        --------
        Create a Horodecki's 2x4 of entangled state
        >>> qden=create_bes_horodecki_24_state(1)
        >>> print(qden)
        [[0.125 0.    0.    0.    0.    0.125 0.    0.   ]
         [0.    0.125 0.    0.    0.    0.    0.125 0.   ]
         [0.    0.    0.125 0.    0.    0.    0.    0.125]
         [0.    0.    0.    0.125 0.    0.    0.    0.   ]
         [0.    0.    0.    0.    0.125 0.    0.    0.   ]
         [0.125 0.    0.    0.    0.    0.125 0.    0.   ]
         [0.    0.125 0.    0.    0.    0.    0.125 0.   ]
         [0.    0.    0.125 0.    0.    0.    0.    0.125]]
    """
    x = np.array([b, b, b, b, b, b, b, b])
    rho = np.diag(x, k=0)
    rho[4][4] = (1.0 + b) / 2.0
    rho[7][7] = (1.0 + b) / 2.0
    rho[4][7] = np.sqrt(1.0 - b * b) / 2.0
    rho[7][4] = np.sqrt(1.0 - b * b) / 2.0
    rho[5][0] = b
    rho[6][1] = b
    rho[7][2] = b
    rho[0][5] = b
    rho[1][6] = b
    rho[2][7] = b
    rho = rho / (7.0 * b + 1.0)
    return rho

def create_bes_horodecki_33_state(a):
    """
        Create a Horodecki's 3x3 of entangled state

        Parameters
        ----------
        a : real
            the entangled state with a parameter a

        Returns
        -------
        density matrix : numpy array
            Numpy array for the Horodecki's two-qutrit state
            expressed as density matrix

        Examples
        --------
        Create a Horodecki's 3x3 of entangled state
        >>> qden=create_bes_horodecki_33_state(1)
        >>> print(qden)
        [[0.11111111 0.         0.         0.         0.11111111 0.          0.         0.         0.11111111]
         [0.         0.11111111 0.         0.         0.         0.          0.         0.         0.        ]
         [0.         0.         0.11111111 0.         0.         0.          0.         0.         0.        ]
         [0.         0.         0.         0.11111111 0.         0.          0.         0.         0.        ]
         [0.11111111 0.         0.         0.         0.11111111 0.          0.         0.         0.11111111]
         [0.         0.         0.         0.         0.         0.11111111  0.         0.         0.        ]
         [0.         0.         0.         0.         0.         0.          0.11111111 0.         0.        ]
         [0.         0.         0.         0.         0.         0.          0.         0.11111111 0.        ]
         [0.11111111 0.         0.         0.         0.11111111 0.          0.         0.         0.11111111]]
    """
    x = np.array([a, a, a, a, a, a, a, a, a])
    rho = np.diag(x, k=0)
    rho[6][6] = (1.0 + a) / 2.0
    rho[8][8] = (1.0 + a) / 2.0
    rho[8][6] = np.sqrt(1.0 - a * a) / 2.0
    rho[6][8] = np.sqrt(1.0 - a * a) / 2.0
    rho[4][0] = a
    rho[8][0] = a
    rho[4][8] = a
    rho[0][4] = a
    rho[0][8] = a
    rho[8][4] = a
    rho = rho / (8.0 * a + 1.0)
    return rho

def create_ghz_state(d, n):
    """
        Create a GHZ state 
 
        Parameters
        ----------
        d : integer
            the number of degrees of freedom for the qudit d,

        n : integer
            number of qudits for the created state

        Returns
        -------
        quantum state : numpy vector
            Numpy vector gives the d-partite GHZ state acting on local n dimensions

        Examples
        --------
        Create of register for a GHZ state two qubit state
        >>> q0=create_ghz_state(2, 2)
        >>> print(q0)
        [0.70710678 0.         0.         0.70710678]
    """
    g = np.zeros(d ** n)
    step = np.sum(np.power(d, range(n)))
    g[range(d) * step] = 1/np.sqrt(d)
    #g[0] = 1/np.sqrt(d)
    #g[-1] = 1/np.sqrt(d)
    return g

def create_ghz_alpha_qubit_state(n, alpha):
    """
        Create a GHZ state for N qubits register
            with alpha parameter 
 
        Parameters
        ----------
        d : integer
            the number of degrees of freedom for the qudit d,

        n : integer
            number of qudits for the created state

        Returns
        -------
        quantum state : numpy vector
            Numpy vector gives the d-partite GHZ state acting on local n dimensions

        Examples
        --------
        Create of register for a GHZ state two qubit state
        >>> q0=create_ghz_alpha_qubit_state(2, np.pi/4)
        >>> print(q0)
        [0.70710678 0.         0.         0.70710678]
    """
    d=2
    g = np.zeros(d ** n)
    g[0] = np.sin(alpha)
    g[-1] = np.cos(alpha)
    return g

def create_generalized_n_qutrit_ghz_state(N, alpha):
    #d=3
    q = np.zeros( 3 ** N)
    q[0]=np.sin( alpha )
    
    val=''
    for i in [1] * N:
        val = val + str(i)   
    q[ int(val, 3) ] = 1.0/np.sqrt(2) * np.cos(alpha)
    
    val=''
    for i in [2] * N:
        val = val + str(i)   
    q[ int(val, 3) ] = 1.0/np.sqrt(2) * np.cos(alpha)
    
    return q

def create_noon_state(d, N, theta):
     g = np.zeros(d * d, dtype=complex)
     g[N*d] = 1.0/np.sqrt(2)
     g[N] = np.exp(1j * N * theta) *  (1.0/np.sqrt(2))
     return g


def create_wstate(n):
    """
        Create a W-state

        Parameters
        ----------
        n : integer
            number of qudits for the created state

        Returns
        -------
        quantum state : numpy vector
            Numpy vector gives the n-qubit W-state

        Examples
        --------
        Create of register for a w state
        >>> q0=create_wstate(2)
        >>> print(q0)
        [0.         0.70710678 0.70710678 0.        ]
    """
    w = np.zeros(2 ** n)
    for i in range (n):
        w[2 ** i] = 1 / np.sqrt(n)
    return w

def create_isotropic_qubit_state(p):
    """
        Create a isotropic of qubit state
        Parameters
        ----------
        p : real
           The parameter of the isotropic state

        Returns
        -------
        density matrix : numpy array
           The isotropic state expressed
           as density matrix

        Examples
        --------
        Create of register for a isotropic of qubit state
        >>> q0=create_isotropic_qubit_state(0.25)
        >>> print(q0)
        [[0.3125 0.     0.     0.125 ]
         [0.     0.1875 0.     0.    ]
         [0.     0.     0.1875 0.    ]
         [0.125  0.     0.     0.3125]]
    """
    q = create_qubit_bell_state()
    qdentmp = np.outer(q, q)
    qden = (p * qdentmp) + ((1-p) * 0.25 * np.eye(4))
    return qden

def create_mixed_and_entangled_two_qubit_state(p, state="Bell+"):
    """
        Create a mixed and entangled state for two qubit

        Parameters
        ----------
        p : real
           The parameter of the mixed 
        state : string
           The name of quantum state: Bell+, Bell-, W.
           Default value is Bell+.

        Returns
        -------
        density matrix : numpy array
            The Werner state expressed
            as density matrix.

        Examples
        --------
        Create of register for two qubit to a mixed and entanglement state
        between max entangled state and mixed state
        >>> q0=create_mixed_and_entangled_two_qubit_state(0.25, state="Bell+")
        >>> print(q0)
        [[0.3125 0.     0.     0.125 ]
         [0.     0.1875 0.     0.    ]
         [0.     0.     0.1875 0.    ]
         [0.125  0.     0.     0.3125]]
    """
    if state=="Bell+":
        q = create_qubit_bell_state()
    if state=="Bell-":
        q = create_qubit_bell_state(minus=1)
    if state=="W":
        q = create_wstate(2)
    qdentmp = np.outer(q, q)
    qden = (p * qdentmp) + ((1-p) * 0.25 * np.eye(4))
    return qden

def create_two_qubit_werner_state(p):
    werner_state =np.array(
                [[p/3.0,            0,            0,     0],
                 [    0, ( 3-2*p)/6.0, (-3+4*p)/6.0,     0],
                 [    0, (-3+4*p)/6.0, ( 3-2*p)/6.0,     0],
                 [    0,            0,            0, p/3.0]])
    return werner_state

def create_werner_state(d, p):    
    eye = np.eye(d*d)
    fab = np.zeros((d*d, d*d))
    for i in range(d):
        for j in range(d):
            fa=np.zeros( (d, d) )
            fb=np.zeros( (d, d) )
            fa[i,j]=1;
            fb[j,i]=1;
            fab=fab+np.kron(fa, fb)
    
    psym = 0.5 * ( eye + fab )
    pas  = 0.5 * ( eye - fab )
    werner_state = ((p*(2.0))/(d*(d+1.0)))*psym + (((1.0-p)*(2.0))/(d*(d-1.0)))*pas
    
    return werner_state

def create_four_qubit_cluster_state():
    q = np.zeros(16)
    
    q[0] = q[2] = q[12] = q[15] = 0.5
    
    return q

def create_four_qubit_singlet_state():
    q = np.zeros(16)
    
    q[3] = q[12] = 1.0/np.sqrt(3.0)
    q[5] = q[6] = q[9] = q[10] = -1.0/np.sqrt(12.0)
    
    return q
    
def create_four_qubit_higuchi_sudbery_state():
    q = np.zeros(2 ** 4, dtype=complex)
    
    omega = -0.5  + (np.sqrt(3)/2.0)*1j
    omega_square = omega * omega
    
    # 1/√6( |1100> + |0011> + ω(|1001> + |0110>) + ω^2(|1010> + |0101>))
    
    q[12] = 1
    q[3] = 1
    
    q[9] = omega
    q[6] = omega
    
    q[10] = omega_square
    q[5] = omega_square
    
    q = q * 1.0/np.sqrt(6)  
    
    return q

def create_dicke_state(evalue, Nvalue):
    coff = 1.0 / math.sqrt(math.comb(Nvalue, evalue))
    q = np.zeros(2 ** Nvalue)
    # not effective, rewritten required
    for p in itertools.permutations( [0]*(Nvalue - evalue) + [1]*evalue ):
        val=''
        for it in p:
            val=val + str(it)
        q[int(val, 2)] = coff
    return q

def create_three_qutrit_singlet_state():
    pass

def create_chessboard_state(a,b,c,d,m,n):
    """
        Create a Chessboard state

        Parameters
        ----------
        a,b,c,d,m,n : integer
            The real arguments

        Returns
        -------
        density matrix
            Numpy array for quantum state
            expressed as density matrix


        Examples
        --------
        Create a Chessboard state
        >>> q0=create_chessboard_state(0.25, 0.5, 0.5, 0.1, 0.2, 0.8)
        >>> print(q0)
            [[ to fix  ]]


        """
    s = a * np.conj(c) / np.conj(n)
    t = a * d / m

    v1 = np.array([m, 0, s, 0, n, 0, 0, 0, 0])
    v2 = np.array([0, a, 0, b, 0, c, 0, 0, 0])
    v3 = np.array([(np.conj(n)), 0, 0, 0, (-np.conj(m)), 0, t, 0, 0])
    v4 = np.array([0, (np.conj(b)), 0, (-np.conj(a)), 0, 0, 0, d, 0])

    rho = np.outer(np.transpose(v1), v1) + np.outer(np.transpose(v2), v2) + np.outer(np.transpose(v3), v3) + np.outer(np.transpose(v4), v4)

    rho = rho/np.trace(rho)

    return rho

def create_gisin_state(lambdaX, theta):
    """
            Create a gisin state

            Parameters
            ----------
            lambdaX: float
                The real argument in 0 between 1 (closed interval)
            theta: float
                The real argument

        
            Returns
            -------
            density matrix
                Numpy array for Gisin state
                expressed as density matrix

            Examples
            --------
            Create a Gisin state
            >>> q0=create_gisin_state(0.25, 2)
            >>> print(q0)            
                [[0.375      0.         0.         0.        ]
                 [0.         0.20670545 0.09460031 0.        ]
                 [0.         0.09460031 0.04329455 0.        ]
                 [0.         0.         0.         0.375     ]]
    """
    rho_theta = np.array([[0, 0, 0, 0],
                [0, (np.sin(theta) ** 2), (-np.sin(2 * theta) / 2), 0],
                [0, (-np.sin(2 * theta) / 2), (np.cos(theta) ** 2), 0],
                [0, 0, 0, 0]])

    rho_uu_dd =np.array(
                [[1, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 1]])

    gisin_state = lambdaX * rho_theta + (1- lambdaX) * rho_uu_dd / 2

    return gisin_state

def create_x_two_qubit_random_state():
    antydiagval = np.random.rand(2)

    diagval = np.random.rand(4)
    diagval = (diagval / np.linalg.norm(diagval)) ** 2

    leftVal0=diagval[1] * diagval[2]
    rightVal0=np.abs(antydiagval[1]) ** 2

    leftVal1=diagval[0] * diagval[3]
    rightVal1=np.abs(antydiagval[0]) ** 2

    while not (leftVal0 >= rightVal0 and leftVal1 >= rightVal1):
        antydiagval = np.random.rand(2)

        diagval = np.random.rand(4)
        diagval = (diagval / np.linalg.norm(diagval)) ** 2

        leftVal0=diagval[1] * diagval[2]
        rightVal0=np.abs(antydiagval[1]) ** 2

        leftVal1=diagval[0] * diagval[3]
        rightVal1=np.abs(antydiagval[0]) ** 2

    qden = np.zeros( 16 )
    qden = np.reshape( qden, (4, 4) )

    qden[0,0] = diagval[0]
    qden[1,1] = diagval[1]
    qden[2,2] = diagval[2]
    qden[3,3] = diagval[3]

    qden[0,3] = antydiagval[0]
    qden[1,2] = antydiagval[1]
    qden[2,1] = antydiagval[1].conj()
    qden[3,0] = antydiagval[0].conj()

    return qden



# (1) new function to create quantum states
def create_random_pure_state( _d, _n, complex_no=False ):
    """
        Creates a random pure quantum state.

        Parameters
        ----------
        _d : int
            Freedom level of generated state.
        _n : int
            The number of qudits.
        complex_no : Boolean
            Default as False, what means that amplitudes are complex number 
            with imaginary part always equal to zero. If this parameter is
            True, then aplitudes may have non-zero imaginary part (the 
            probability that pointed out amplitude has non-zero imaginary part
            is 0.5).

        Returns
        -------
        _tab : numpy array
            A normalized quantum state.

        Examples
        --------
        Generation of 1-qubit states with probable non-zero imaginary parts in
        amplitudes' values:
        >>> ent.create_random_pure_state(2, 1, True)
        [0.-0.53545756j 0.+0.84456214j]
        >>> ent.create_random_pure_state(2, 1, True)
        [-0.87414796+0.j  0.4856597 +0.j]
        >>> ent.create_random_pure_state(2, 1, True)
        [0.        +0.92040475j 0.39096686+0.j        ]
        Generation of 1-qubit states only with imaginary parts equal to zero:
        >>> ent.create_random_pure_state(2, 1)
        [0.77860131+0.j 0.62751893+0.j]
        Generation of 1-qutrit states only with imaginary parts equal to zero:
        >>> ent.create_random_pure_state(3, 1)
        [-0.46514564+0.j  0.63426271+0.j -0.61753571+0.j]
        Generation of 1-qutrit states with probable non-zero imaginary parts:
        >>> ent.create_random_pure_state(3, 1, True)
        [0.65804958+0.j         0.24362108+0.j         0.        -0.71247422j]
        
    """
    _x = _d ** _n
    _tab = np.ndarray(shape=(_x),dtype=complex)
    _tab_final = np.ndarray(shape=(_x),dtype=complex)
    _list1 = [0, 1]
    
    if complex_no==False:
        for i in range(_x):
            _tab[i] = rd.uniform(-1,1) + 0j
        print(_tab)
    elif complex_no==True:
        for i in range(_x):
            _compl=rd.choice(_list1)
            if _compl==0:
                _tab[i] = rd.uniform(-1,1) + 0j
            else:
                _tab[i] = 0 + rd.uniform(-1,1)*1j
        print(_tab)
    else:
        raise ValueError("The parameter's _complex value has to be True or False!")
        return None
        
    sum_all=0
    for i in range(_x):
        sum_all += abs(sympy.re(_tab[i])) + abs(sympy.im(_tab[i]))
    print(sum_all)  
    for i in range(_x):
        if sympy.re(_tab[i]) >= 0 and sympy.im(_tab[i]) == 0:
            re_pos=sympy.re(_tab[i])
            re_pos=sympy.sqrt(re_pos/sum_all)
            _tab_final[i]=re_pos
        elif sympy.re(_tab[i]) < 0 and sympy.im(_tab[i]) == 0:
            re_neg=abs(sympy.re(_tab[i]))
            re_neg=sympy.sqrt(re_neg/sum_all)
            _tab_final[i]=re_neg*(-1)
        elif sympy.re(_tab[i]) == 0 and sympy.im(_tab[i]) > 0:
            im_pos=sympy.im(_tab[i])
            im_pos=sympy.sqrt(im_pos/sum_all)
            _tab_final[i]=im_pos*1j
        elif sympy.re(_tab[i]) == 0 and sympy.im(_tab[i]) < 0:
            im_neg=abs(sympy.im(_tab[i]))
            im_neg=sympy.sqrt(im_neg/sum_all)
            _tab_final[i]=im_neg*(-1j)
        else:
            print('Should never happen')
    
    return _tab_final

# (2) new function to create quantum states
def create_random_1qubit_pure_state():
    """
        Creates a random 1-qubit (d=2) pure state.

        Returns
        -------
        _tab : numpy array
            A normalized quantum state with amplitudes as complex numbers.

        Examples
        --------
        >>> ent.create_random_1qubit_pure_state()
        [0.33570732+0.j         0.        -0.94196635j]
        >>> ent.create_random_1qubit_pure_state()
        [0.93620949+0.j 0.35144243+0.j]
        >>> ent.create_random_1qubit_pure_state()
        [ 0.75643296+0.j -0.65407123+0.j]
        
    """
    _tab=np.ndarray(shape=(2),dtype=complex)
    for i in range(2):
        _tab[i]=rd.uniform(0,1)
    sum_all=0
    for i in range(2):
        sum_all+=_tab[i]
    for i in range(2):
        _tab[i]=sympy.sqrt(_tab[i]/sum_all)
    _list1 = [0, 1]
    sign=rd.choice(_list1)
    compl=rd.choice(_list1)
    if sign==1:
        _tab[1]*=-1
    if compl==1:
        pom=_tab[1]*sympy.I
        _tab[1]=pom
    return _tab

#
# TO DOC GEN
#
# (3) new function to create quantum states
def create_random_2qubit_pure_state():
    """
        Creates a random 2-qubit (d=2) pure state. The probability of 
        entanglement occurence is 1/2.

        Returns
        -------
        _tab : numpy array
            A normalized quantum state with amplitudes as complex numbers.

        Examples
        --------
        >>> ent.create_random_2qubit_pure_state()
        [ 0.35399885+0.j          0.        -0.06455057j -0.91788041+0.j
          0.        +0.1673726j ]
        >>> ent.create_random_2qubit_pure_state()
        [ 0.42914302+0.j          0.        +0.71644913j -0.28263805+0.j
         -0.        -0.47186083j]
        >>> ent.create_random_2qubit_pure_state()
        [0.70710678+0.j 0.        +0.j 0.        +0.j 0.70710678+0.j]
        
    """
    list1 = [0, 1]
    ent = rd.choice(list1)
    if ent==1:
        _tab=np.zeros(shape=(4),dtype=complex)
        list2 = [0, 1, 2, 3]
        r=rd.choice(list2)
        if r==0:
            _tab[0]=1
            _tab[3]=1
        elif r==1:
            _tab[0]=1
            _tab[3]=-1
        elif r==2:
            _tab[1]=1
            _tab[2]=1
        elif r==3:
            _tab[1]=1
            _tab[2]=-1
        _tab = _tab * 1.0/np.sqrt(2.0)
    else:
        _tab=np.ndarray(shape=(4),dtype=complex)
        s1=create_random_1qubit_pure_state()
        s2=create_random_1qubit_pure_state()
        _tab[0]=s1[0]*s2[0]
        _tab[1]=s1[0]*s2[1]
        _tab[2]=s1[1]*s2[0]
        _tab[3]=s1[1]*s2[1]
    return _tab


#
# TO DOC GEN
#
# (4) new function to create quantum states
def create_random_2qubit_separable_pure_state():
    _tab=np.ndarray(shape=(4),dtype=complex)
    
    _s1=create_random_1qubit_pure_state()
    _s2=create_random_1qubit_pure_state()
    
    _tab[0]=_s1[0]*_s2[0]
    _tab[1]=_s1[0]*_s2[1]
    _tab[2]=_s1[1]*_s2[0]
    _tab[3]=_s1[1]*_s2[1]
    
    return _tab

#
#
#

def create_standard_base_matrix(d, x, y):
    
    if x>d or y>d:
        raise DimensionError("X or Y values is bigger than d! (x={0}, y={1}, d={2})".format(x,y,d))
    
    mat = np.zeros( (d, d) )
    mat[x,y] = 1
    return mat

#
#
#

def vector_state_to_density_matrix(q):
    return np.outer(q, np.transpose(q.conj()))

def create_density_matrix_from_vector_state(q):
    return vector_state_to_density_matrix(q)

#
# Spectral decomposition of density matrix
#

def eigen_decomposition(qden):
    """
        Create a eigen decomposition

        Parameters
        ----------
        qden : numpy array
            The parameter qden represents a density matrix

        Returns
        -------
        eigval : numpy array
        eigvec : numpy array
            The vector and array of a eigenvalues and eigenvectors

        Examples
        --------
        Create eigen decomposition of given quantum state:
        >>> qden=create_werner_two_qubit_state(0.75)
        >>> ed=eigen_decomposition(qden)
        >>> print(ed)
        (array([0.0625, 0.0625, 0.0625, 0.8125]), array([[-0.70710678,  0.        ,  0.        , -0.70710678],
               [ 0.        ,  0.        , -1.        ,  0.        ],
               [ 0.        ,  1.        ,  0.        ,  0.        ],
               [ 0.70710678,  0.        ,  0.        , -0.70710678]]))
    """
    eigval, eigvec = np.linalg.eigh(qden)
    return eigval, eigvec

def eigen_decomposition_for_pure_state(q):
    """
        Create a eigen decomposition for pure state

        Parameters
        ----------
        q : numpy vector
            The parameter q represents the vector state.
            The input vector is converted to density matrix.

        Returns
        -------
        A two element tuple (eigval,eigvec) where:
            eigval : is a numpy array of a eigenvalues,
            eigvec : is a numpy array of a eigenvectors.

        Examples
        --------
        Create of register for eigen decomposition for pure state
        >>> q = create_qubit_bell_state()
        >>> ed=eigen_decomposition_for_pure_state(q)
        >>> print(ed)
        (array([0., 0., 0., 1.]), array([[-0.70710678,  0.        ,  0.        , -0.70710678],
               [ 0.        ,  0.        , -1.        ,  0.        ],
               [ 0.        ,  1.        ,  0.        ,  0.        ],
               [ 0.70710678,  0.        ,  0.        , -0.70710678]]))
    """
    qden = np.outer(q,q)
    eigval,eigvec = np.linalg.eigh(qden)
    return eigval, eigvec

def reconstruct_density_matrix_from_eigen_decomposition(eigval, eigvec):
    """
        Reconstruction of density matrix from a eigen decomposition

        Parameters
        ----------
        eigval : numpy array
        eigvec : numpy array
            The vector and array of a eigenvalues and eigenvectors

        Returns
        -------
        density matrix : numpy array
            Numpy array for reconstructed quantum state

        Examples
        --------
        Reconstruction of density matrix from eigen decomposition:
        >>> q = create_qubit_bell_state()
        >>> qden = vector_state_to_density_matrix(q)
        >>> ev,evec = eigen_decomposition(qden)
        >>> qdenrecon = reconstruct_density_matrix_from_eigen_decomposition(ev, evec)
        >>> print( qdenrecon )
        [[0.5 0.  0.  0.5]
         [0.  0.  0.  0. ]
         [0.  0.  0.  0. ]
         [0.5 0.  0.  0.5]]
     """

    i = 0
    qden = np.zeros([eigval.shape[0],eigval.shape[0]])
    for ev in eigval:
        qden = qden + np.outer(eigvec[:, i], ev * eigvec[:, i])
        i = i + 1
    return qden

#
# Schmidt decomposition of vector state
#

def schmidt_decomposition_for_vector_pure_state(q, decomposition_shape):
    """
        Create a Schmidt decomposition for vector pure state

        Parameters
        ----------
        q : numpy vector
            The parameter q represents the vector state

        decomposition_shape : tuple of two integers
            Dimensions of two subsystems

        Returns
        -------
        A three element tuple (s,u, vh) where:
           s  : numpy vector containing Schmidt coefficients,
           u  : arrays of left Schmidt vectors,
           vh : arrays of right Schmidt vectors.

        Examples
        --------
        Create of register to Schmidt decomposition for vector pure state
        >>> q = create_qubit_bell_state()
        >>> decomposition_shape=(2, 2)
        >>> sd=schmidt_decomposition_for_vector_pure_state(q, decomposition_shape)
        >>> print(sd)
        (array([0.70710678, 0.70710678]), array([[1., 0.],
               [0., 1.]]), array([[1., 0.],
               [0., 1.]]))
    """
    d1,d2 = decomposition_shape
    m = q.reshape(d1, d2)
    u, s, vh = np.linalg.svd(m, full_matrices=True)
    
    return s, u, vh

def schmidt_decomposition_operator(qden, decomposition_shape):
    schmidt_shape=decomposition_shape
    schmidt_shape_np=np.zeros((1,2))
    schmidt_shape_np[0,0] = schmidt_shape[0]; schmidt_shape_np[0,1] = schmidt_shape[1]
    schmidt_shape_np = np.concatenate((schmidt_shape_np,schmidt_shape_np))
    qqden = qden.reshape(int(np.prod(schmidt_shape_np)), 1)

    qqden_after_swap=np.reshape( np.transpose( np.reshape(qqden, schmidt_shape * 2), (0,2,1,3) ), (int(np.prod(schmidt_shape_np)), 1))

    schmidt_shape_final_decomp = np.prod(schmidt_shape_np, axis=0)
    s, e, f = schmidt_decomposition_for_vector_pure_state(qqden_after_swap, (int(schmidt_shape_final_decomp[0]), int(schmidt_shape_final_decomp[1])))

    elist=[None] * len(s)
    flist=[None] * len(s)
    for idx in range(len(s)):
        elist[idx] = e[idx].reshape(schmidt_shape[0], schmidt_shape[0])
    for idx in range(len(s)):
        flist[idx]=f[idx].reshape(schmidt_shape[1], schmidt_shape[1])
    return s,elist,flist

def schmidt_rank_for_vector_pure_state(q, decomposition_shape):
    """
        Calculate a Schmidt rank for vector pure state

        Parameters
        ----------
        q : numpy array
            The parameter of the vector state

        decomposition_shape : tuple of two integers
            Dimensions of two subsystems

        Returns
        -------
        sch_rank : integer
            Schmidt rank value as integer number

        Examples
        --------
        Calculate of Schmidt rank for vector pure state
        >>> q = create_qubit_bell_state()
        >>> decomposition_shape=(2, 2)
        >>> sr=schmidt_rank_for_vector_pure_state(q, decomposition_shape)
        >>> print(sr)
            2
    """
    d1,d2 = decomposition_shape
    m = q.reshape(d1, d2)
    sch_rank = np.linalg.matrix_rank(m)
    return sch_rank

def schmidt_rank_for_operator(qden, decomposition_shape):
    schmidt_shape=decomposition_shape
    schmidt_shape_np=np.zeros((1,2))
    schmidt_shape_np[0,0] = schmidt_shape[0]; schmidt_shape_np[0,1] = schmidt_shape[1]
    schmidt_shape_np = np.concatenate((schmidt_shape_np,schmidt_shape_np))
    qqden = qden.reshape(int(np.prod(schmidt_shape_np)), 1)

    qqden_after_swap=np.reshape( np.transpose( np.reshape(qqden, schmidt_shape * 2), (0,2,1,3) ), (int(np.prod(schmidt_shape_np)), 1))

    schmidt_shape_final_decomp = np.prod(schmidt_shape_np, axis=0)
    schrank = schmidt_rank_for_vector_pure_state(qqden_after_swap, (int(schmidt_shape_final_decomp[0]), int(schmidt_shape_final_decomp[1])))

    return schrank

def reconstruct_state_after_schmidt_decomposition(s, e, f):
    """
        Reconstruction state after Schmidt decomposition

        Parameters
        ----------
        s : numpy array
            The values of Schmidt coefficients

        e, f : numpy arrays
            The basis vectors from Schmidt decomposition

        Returns
        -------
        quantum state : numpy vector
            Numpy vector for quantum state

        Examples
        --------
        Reconstruction state after Schmidt decomposition:
        >>> q = create_qubit_bell_state()
        >>> schmidt_shp=(2, 2)
        >>> s,e,f = schmidt_decomposition_for_vector_pure_state(q,schmidt_shp)
        >>> q0=reconstruct_state_after_schmidt_decomposition(s, e, f)
        >>> print(q0)
            [0.70710678 0.         0.         0.70710678]
    """

    dfin = s.shape[0] * e.shape[0]
    v = np.zeros(dfin)

    idx = 0
    for sv in s:
        v = v + np.kron(sv * e[idx], f[idx])
        idx = idx + 1
    return v

#
# TO DOC GEN
#
def is_entangled_vector_2q_state( q ):
    rslt = False
    
    decomposition_shape = (2, 2)
    sr = schmidt_rank_for_vector_pure_state(q, decomposition_shape)
    
    if sr>1:
        rslt = True
        
    return rslt

#
# Creation of spectral table of given quantum state
# expressed as density matrix
#

def create_spectral_and_schmidt_table(qden, schmidt_shape):
    ev,evec = eigen_decomposition(qden)
    #idxs = [i for i, e in enumerate(ev) if e != 0.0]
    idxs = range(len(ev))
    evtbl=[]
    for ii in idxs:
        evtbl.append( (ev[ii], evec[:, ii]) )
    schmdtbl=[]
    for evt in evtbl:
        s, e, f = schmidt_decomposition_for_vector_pure_state(evt[1], schmidt_shape)
        schmdtbl.append( (s,e,f) )
    return evtbl, schmdtbl

def create_spectral_and_schmidt_table_data(qden, schmidt_shape):
    evtbl, schmdtbl = create_spectral_and_schmidt_table( qden, schmidt_shape)
    return (evtbl, schmdtbl)

def create_sas_table_data(qden, schmidt_shape):
    evtbl, schmdtbl = create_spectral_and_schmidt_table( qden, schmidt_shape)
    return (evtbl, schmdtbl)

def calculate_statistic_for_sas_table(e,s):
    idx=len(e)-1;
    vtbl0=[]
    vtbl1=[]
    while idx >=0:
        vtbl0.append(s[idx][0][0])
        vtbl1.append(s[idx][0][1])
        idx=idx-1
    return ( np.var(vtbl0), np.var(vtbl1), np.std(vtbl0), np.std(vtbl1) )

def print_sas_table( sas_table, statistics=0):
    e,s = sas_table
    idx=len(e)-1;
    vtbl0=[]
    vtbl1=[]
    while idx >=0:
        vtbl0.append(s[idx][0][0])
        vtbl1.append(s[idx][0][1])
        print(chop(s[idx][0]), "|", chop(e[idx][0]))
        idx=idx-1
    if statistics==1:
        print("var=", np.var(vtbl0), np.var(vtbl1))
        print("std=", np.std(vtbl0), np.std(vtbl1))

#
# Routines for Entropy calculation
#

def entropy(qden, logbase="e"):
    """
        Computes the entropy of a density matrix
        Parameters
        ----------
        qden : numpy array
            The parameter qden represents a density matrix
        logbase : string
            A string represents the base of the logarithm: 
               "e", "2", and "10".

        Returns
        -------
        entropy_val : float
            The value of entropy
        Examples
        --------
        Calculate the value of entropy of the density matrix
        >>> qden=create_x_two_qubit_random_state()
        >>> q0=entropy(qden, "10")
        >>> print(q0)
            0.5149569745101069
    """
    eigval,evec = eigen_decomposition(qden)
    entropy_val = 0.0
    for eval in eigval:
        e=eval
        if chop(e) >= precision_for_entrpy_calc:
            if logbase == "e":
                entropy_val = entropy_val + e * np.log(e)
            if logbase == "2":
                entropy_val = entropy_val + e * np.log2(e)
            if logbase == "10":
                entropy_val = entropy_val + e * np.log10(e)
    return chop(-entropy_val)

#
# Negativity
#

def negativity( qden, d=2, n=2 ):
    """
        Computes a negativity of bipartite density matrix
        Parameters
        ----------
        qden : numpy array
            The parameter qden represents a density matrix
        d : integer
            the number of degrees of freedom for the qudit d,
        n : integer
            number of qudits for the created state

        Returns
        -------
        negativity_value : float
            The value of negativity of bipartite density matrix

        Examples
        --------
        Calculate the value for a negativity of bipartite density matrix
        >>> q = create_wstate(d)
        >>> qden = vector_state_to_density_matrix( q )
        >>> q0=negativity(qden)
        >>> print(q0)
            0.4999999999999998
    """
    dim = int(np.log(d ** n)/np.log(d))
    #dim = int((d ** n)/(d))
    qdentmp = partial_transpose(qden, [[dim,dim], [dim,dim]], [0, 1])
    negativity_value = (np.linalg.norm(qdentmp, 'nuc') - 1.0)/2.0
    return chop(negativity_value)

#
# Concurrence
#

def concurrence( qden ):
    """
        Computes value a concurrence for a two-qubit state
        Parameters
        ----------
        qden : numpy array
            The parameter qden represents a density matrix
        Returns
        -------
        c : float
            The value of concurrence of two-qubit state

        Examples
        --------
        Calculate the value of concurrence for a two-qubit state
        >>> qden=create_werner_two_qubit_state(0.79)
        >>> q0=concurrence(qden)
        >>> print(q0)
            0.6849999999999994
    """
    pauliy=np.array([0.0, -1.0J, 1.0J, 0.0]).reshape(2,2)
    qden=np.matrix(qden)
    R = qden * np.kron(pauliy, pauliy) * qden.getH() * np.kron(pauliy, pauliy)
    e,v=np.linalg.eig(R)
    evalRealList = [float(ev.real) for ev in e]
    
    evallist = []
    for v in evalRealList:
        if v>0:
            evallist.append(np.sqrt(v))
        else:
            evallist.append(chop(v))
    evallist=-np.sort(-np.array(evallist))
    c=np.max([evallist[0]-evallist[1]-evallist[2]-evallist[3], 0.0])
    
    return c

#
#
#

# reference implementation directly based on 
# https://github.com/qutip/qutip/blob/master/qutip/partial_transpose.py
# 
def partial_transpose_main_routine(rho, dims, mask):
    mask = [int(i) for i in mask]
    nsys = len(mask)
    pt_dims = np.arange(2 * nsys).reshape(2, nsys).T
    pt_idx = np.concatenate( [ [pt_dims[n, mask[n]] for n in range(nsys)],
                               [pt_dims[n, 1 - mask[n]] for n in range(nsys)] ] )
    data = rho.reshape(np.array(dims).flatten()).transpose(pt_idx).reshape(rho.shape)

    return data

def partial_transpose(rho, dims, no_transpose):
    """
        Computes a partial transpose of a given density matrix rho.
        Implementation directly based on
        https://github.com/qutip/qutip/blob/master/qutip/partial_transpose.py
        Parameters
        ----------
        rho : numpy array
            The parameter rho represents a density matrix
        dims : list
            A list of lists where each sublist describes dimensions of each
            subsystem
        no_transpose : list
            A list of boolean values (0,1) where the number of elements is equal to
            the number of subsystems. If value is 1 then the partial
            transposition is performed on the pointed out subsystem
        Returns
        -------
        data : numpy array
            The matrix rho after the partial transposition operation
        Examples
        --------
        Calculate the partial transpose of matrix rho_AB sized 6x6. Let us
        assume that rho_AB is a density matrix calculated for bipartite system
        of one qubit (freedom level d=2) and one qutrit (d=3). The partial
        transposition is calculated for the first and then for the second
        subsystem
        >>> print(rho_AB)
            [[11 12 13 14 15 16]
             [21 22 23 24 25 26]
             [31 32 33 34 35 36]
             [41 42 43 44 45 46]
             [51 52 53 54 55 56]
             [61 62 63 64 65 66]]
        >>> rho_AB_transposed = partial_transpose(rho_AB, [[2,2],[3,3]], [1, 0])
        >>> print(rho_AB_transposed)
            [[11 12 13 41 42 43]
             [24 25 26 54 55 56]
             [14 15 16 44 45 46]
             [31 32 33 61 62 63]
             [21 22 23 51 52 53]
             [34 35 36 64 65 66]]
        >>> rho_AB_transposed = partial_transpose(rho_AB, [[2,2],[3,3]], [0, 1])
        >>> print(rho_AB_transposed)
            [[11 24 14 31 21 34]
             [12 25 15 32 22 35]
             [13 26 16 33 23 36]
             [41 54 44 61 51 64]
             [42 55 45 62 52 65]
             [43 56 46 63 53 66]]
    """
    return partial_transpose_main_routine(rho, dims, no_transpose)

def partial_transpose_for_qubit(rho, no_transpose):
    pass

def partial_transpose_for_qutrits(rho, no_transpose):
    pass

def partial_trace_main_routine(rho, dims, axis=0):
    """
        Computes a partial trace of a given density matrix rho.
        Implementation directly based on
            https://github.com/cvxgrp/cvxpy/issues/563

        Parameters
        ----------
        rho : numpy array
            The parameter rho represents a density matrix
        dims : list
            A list containing dimensions of systems from which matrix rho
            was created, e.g. if dims=[3, 2] then rho is a density matrix of
            a state constructed of one qutrit (freedom level = 3) and one
            qubit (freedom level = 2)
        axis : integer
            The parameter axis points out the subsystem to be traced out
            (the subsystems are numbered from 0)

        Returns
        -------
        data : numpy array
            The density matrix after tracing out pointed subsystem  

        Examples
        --------
        Calculate the partial trace of matrix rho_{ABCD} by extracting its 
        subsystems
        >>> rho_A = np.random.rand(4, 4) + 1j*np.random.rand(4, 4)
        >>> rho_A /= np.trace(rho_A)
        >>> rho_B = np.random.rand(2, 2) + 1j*np.random.rand(2, 2)
        >>> rho_B /= np.trace(rho_B)
        >>> rho_C = np.random.rand(2, 2) + 1j*np.random.rand(2, 2)
        >>> rho_C /= np.trace(rho_C)
        >>> rho_D = np.random.rand(3, 3) + 1j*np.random.rand(3, 3)
        >>> rho_D /= np.trace(rho_D)
        >>> rho_AB = np.kron(rho_A, rho_B)
        >>> rho_ABC = np.kron(rho_AB, rho_C)
        >>> rho_ABCD = np.kron(rho_ABC, rho_D)
        >>> rho_ABC_test = partial_trace_main_routine(rho_ABCD, [4, 2, 2, 3], axis=3)
        >>> rho_AB_test = partial_trace_main_routine(rho_ABC_test, [4, 2, 2], axis=2)
        >>> rho_A_test = partial_trace_main_routine(rho_AB_test, [4, 2], axis=1)
        >>> rho_B_test = partial_trace_main_routine(rho_AB_test, [4, 2], axis=0)
        >>> print("rho_ABC test correct? ", np.allclose(rho_ABC_test, rho_ABC))
            rho_ABC test correct?  True
        >>> print("rho_AB test correct? ", np.allclose(rho_AB_test, rho_AB))
            rho_AB test correct?  True
        >>> print("rho_A test correct? ", np.allclose(rho_A_test, rho_A))
            rho_A test correct?  True
        >>> print("rho_B test correct? ", np.allclose(rho_B_test, rho_B))
            rho_B test correct?  True
    """
    dims_tmp = np.array(dims)
    reshaped_rho = rho.reshape(np.concatenate((dims_tmp, dims_tmp), axis=None))

    reshaped_rho = np.moveaxis(reshaped_rho, axis, -1)
    reshaped_rho = np.moveaxis(reshaped_rho, len(dims) + axis - 1, -1)

    return_trc_out_rho = np.trace(reshaped_rho, axis1=-2, axis2=-1)

    dims_untraced = np.delete(dims_tmp, axis)
    rho_dim = np.prod(dims_untraced)
    
    return return_trc_out_rho.reshape([rho_dim, rho_dim])

def partial_trace(rho, ntrace_out):
    dimensions = []
    single_dim = int(np.log2(rho.shape[0]))
    for _ in range(int(single_dim)):
        dimensions.append(single_dim)

    densitytraceout = partial_trace_main_routine(rho, dimensions, axis = ntrace_out)
    return densitytraceout


def swap_subsystems_for_bipartite_system(rho, dims):
    finaldim=rho.shape
    rshp_dims=(dims, dims)
    axesno=len([e for l in rshp_dims for e in l])
    axesswap=list(range(axesno))[::-1]
    
    rrho = rho.reshape(np.concatenate(rshp_dims, axis=None))
    rrho = np.transpose(rrho, axes=axesswap )
    orho=rrho.reshape( finaldim ).T
    
    return orho

def permutation_of_subsystems(rho, dims, perm):
    nsys=len(dims)
    finaldim=rho.shape
    rshp_dims=(dims, dims)
    axesno=len([e for l in rshp_dims for e in l])
    axesswap=list(range(axesno))[::-1]
    permaxesswap = np.zeros(axesno)
    parts=int(axesno/nsys)
    idx=0
    while idx<parts:
        bidx=(idx*nsys)
        bendidx=bidx+parts+1
        for ii in range(nsys):
            permaxesswap[bidx:bendidx][ii] = axesswap[bidx:bendidx][perm[-ii]]
        idx=idx+1
    rrho = rho.reshape(np.concatenate(rshp_dims, axis=None))
    rrho = np.transpose(rrho, axes=permaxesswap.astype(int) )
    orho=rrho.reshape( finaldim ).T
    return orho


#
# Gram matrices
#

def gram_right_of_two_qubit_state(v):
    m = np.zeros((2,2))
    m[0,0] = np.abs(v[0])**2 + np.abs(v[1])**2;                m[0,1] = v[0].conjugate()*v[2] + v[1].conjugate()*v[3];
    m[1,0] = v[2].conjugate()*v[0] + v[3].conjugate()*v[1];    m[1,1] = np.abs(v[2])**2 + np.abs(v[3])**2;
    
    return m

def gram_left_of_two_qubit_state(v):
    m = np.zeros((2,2))
    m[0,0] = np.abs(v[0])**2.0 + np.abs(v[2])**2.0;            m[0,1] = v[0].conjugate()*v[1] + v[2].conjugate()*v[3];
    m[1,0] = v[1].conjugate()*v[0] + v[3].conjugate()*v[2];    m[1,1] = np.abs(v[1])**2.0 + np.abs(v[3])**2.0;
    
    return m

def full_gram_of_two_qubit_state(v):
    A = np.abs(v[0])**2.0 + np.abs(v[1])**2.0
    B = np.abs(v[2])**2.0 + np.abs(v[3])**2.0
    C = np.abs(v[0])**2.0 + np.abs(v[2])**2.0
    D = np.abs(v[1])**2.0 + np.abs(v[3])**2.0
    C13 = v[0].conjugate()*v[2] + v[1].conjugate()*v[3]
    C12 = v[0].conjugate()*v[1] + v[2].conjugate()*v[3]
    C31 = v[2].conjugate()*v[0] + v[3].conjugate()*v[1]
    C21 = v[1].conjugate()*v[0] + v[3].conjugate()*v[2]

    m = np.zeros((4,4))

    m[0,0] = A * C;     m[0,1] = A * C12;   m[0,2] = C * C13;   m[0,3] = C13 * C12;
    m[1,0] = A * C21;   m[1,1] = A * D;     m[1,2] = C13 * C21; m[1,3] = D * C13;
    m[2,0] = C31 * C;   m[2,1] = C31 * C12; m[2,2] = B * C;     m[2,3] = B * C12;
    m[3,0] = C31 * C21; m[3,1] = D * C31;   m[3,2] = B * C21;   m[3,3] = B * D;

    return m


def gram_matrices_of_vector_state(v, d1, d2):
    dl = np.zeros((d1,d2))
    for i in range(d1):
        ii=0;
        for j in range(d2):
            idx=(i)*d2+j
            dl[ii,i]= dl[ii,i] + v[idx]
            ii=ii+1
    
    dr = np.zeros((d2,d1))
    for j in range(d2):
        ii=0;
        for i in range(d1):
            idx=(i)*d2+j
            dr[ii,j]= dr[ii,j] + v[idx]
            ii=ii+1
    
    dRprime = np.zeros((d2,d1))
    for i in range(0, d1):
        for j in range(0, d2):
            dRprime[i,j] = dr[i] @ dr[j]
    
    dLprime = np.zeros((d1,d2))
    for i in range(0,d1):
        for j in range(0,d2):
            dLprime[i,j] = dl[i] @ dl[j]
    
    return dRprime, dLprime, np.kron(dRprime, dLprime)

#
#
#

def monotone_for_two_qubit_system(rho):
    # S(1) + S(2) − S(12)
    qr1=partial_trace(rho, 1)
    qr2=partial_trace(rho, 0)
    monotone12 = entropy(qr1) + entropy(qr2) - entropy(rho)
    return monotone12


def monotone_for_three_qubit_system(rho):
    pass

def monotone_for_four_qubit_system(rho):
    pass

def monotone_for_five_qubit_system(rho):
    pass

#
#
#

def create_random_qudit_pure_state(d, n, o=0):
    """
    Computes random qudit vector (pure) quantum state.
    Parameters
    ----------
    d : integer
        Describes the freedom level
    n : integer
        Describes the number of qudits
    o : interval
        Specifies elements of the computed vector. If o=0 (default value) the
        vector state is filled with complex numbers, but the imaginary part
        always equals 0. If o=1 the complex numbers are generated (there is
        a small propability that the imaginary part of the element is equals 0).
        When o=2, about half of generated elements, i.e. complex numbers, have
        the imaginary part equal to zero.
    Returns
    -------
    psi : numpy array
        The state vector of n qudits (where the freedom level is specified by d).  
    Raises
    --------
    ValueError
        If o is not 0, 1 or 2.
    Examples
    --------
    Generation of an arbitrary vector state with two qutrits (all imaginary
    parts equal zero):
    >>> print(create_random_qudit_pure_state(3, 2))
        [-0.29096037+0.j  0.26164289+0.j -0.39797311+0.j -0.4512239 +0.j
         -0.46182827+0.j  0.01962066+0.j  0.4003028 +0.j -0.14642441+0.j
         0.29924355+0.j]
    Generation of an arbitrary vector state with three qubits (imaginary
    parts not equal zero):
    >>> print(create_random_qudit_pure_state(2, 3, 1))
        [-0.17483161+0.22494299j -0.30446292+0.43240484j -0.36044787-0.14708236j
         0.1740339 +0.19282974j  0.11754922+0.11299262j -0.12301861+0.06155706j
         -0.23005978-0.18387321j -0.43584704+0.31293522j]
    Generation of an arbitrary vector state with two ququats (about half of
    imaginary parts equal zero):
    >>> print(create_random_qudit_pure_state(4, 2, 2))
        [ 0.31112823-0.17808906j -0.09969305+0.j          0.25818887+0.j
         -0.31093147+0.j          0.21543078+0.08418701j  0.15115571+0.11414964j
         -0.27394187+0.09819806j -0.18150951-0.28072925j -0.17691299+0.03188155j
         -0.0496484 -0.11691207j -0.02743185+0.j          0.31295478+0.j
         -0.07818034+0.j         -0.22111164+0.j         -0.27007445+0.19900792j
         -0.25301898+0.18352255j]
    The attempt to generate a state with an uncorrect parameter:
    >>> print(create_random_qudit_pure_state(4, 2, 3))
        Traceback (most recent call last): ... ValueError: Option has to be: 0, 1 or 2
    """
    ampNumber = d ** n
    psi = np.ndarray(shape=(ampNumber),dtype=complex)
    F = np.ndarray(shape=(ampNumber),dtype=complex)
    if o == 0:
        for i in range(ampNumber):
            F[i] = complex(rd.uniform(-1,1),0)
    elif o == 1:
        for i in range(ampNumber):
            a = rd.uniform(-1,1)
            b = rd.uniform(-1,1)
            F[i] = complex(a,b)
    elif o == 2:
        for i in range(ampNumber):
            a = rd.uniform(-1,1)
            x = rd.randint(0,1)
            if x == 0:
                b = 0
            else:
                b = rd.uniform(-1,1)
            F[i] = complex(a,b)
    else:
        raise ValueError('Option has to be: 0, 1 or 2')
        return None
    #normalization
    con = np.matrix.conjugate(F)
    norm = np.inner(con,F)
    norm = np.sqrt(norm)
    for i in range(ampNumber):
        psi[i] = F[i] / norm
    return psi

def create_random_pure_state_as_density_matrix(d, n, o=0):
    """
    Computes random qudit pure quantum state as density matrix.
    Parameters
    ----------
    d : integer
        Describes the freedom level
    n : integer
        Describes the number of qudits
    o : interval
        Specifies elements of the computed matrix. If o=0 (default value) the
        matrix state is filled with complex numbers, but the imaginary part
        always equals 0. If o=1 the complex numbers are generated (there is
        a small propability that the imaginary part of the element equals 0,
        if the element is not on the main diagonal of the matrix).
        When o=2, the probability of obtaining element with non-zero imaginary
        part is between 1/2 and 3/4.
    Returns
    -------
    rho : numpy array
        The density matrix of n qudits (where the freedom level is specified by d).  
    Raises
    --------
    ValueError
        If o is not 0, 1 or 2.
    Examples
    --------
    Generation of an arbitrary density matrix representing 1-qutrit state (all
    imaginary parts equal zero):
    >>> print(create_random_pure_state_as_density_matrix(3, 1))
        [[0.40559756+0.j 0.11370605+0.j 0.47766004+0.j]
         [0.11370605+0.j 0.03187659+0.j 0.1339082 +0.j]
         [0.47766004+0.j 0.1339082 +0.j 0.56252585+0.j]]
    Generation of an arbitrary density matrix representing 1-qutrit state
    (imaginary parts outside the main diagonal not equal zero):
    >>> print(create_random_pure_state_as_density_matrix(3, 1, 1))
        [[ 0.30476925+0.j         -0.25416467+0.33175818j  0.1177049 -0.15286381j]
         [-0.25416467-0.33175818j  0.57309971+0.j         -0.26456161-0.00064634j]
         [ 0.1177049 +0.15286381j -0.26456161+0.00064634j  0.12213104+0.j        ]]
    Generation of an arbitrary density matrix for 2-qubit state (more then half
    of imaginary parts equal zero):
    >>> print(create_random_pure_state_as_density_matrix(2, 2, 2))
        [[ 0.13354882+0.j         -0.10835538-0.15421297j  0.02296095+0.j
          -0.16818875-0.2266635j ]
         [-0.10835538+0.15421297j  0.26598909+0.j         -0.01862946+0.02651373j
          0.39819601-0.01030842j]
         [ 0.02296095+0.j         -0.01862946-0.02651373j  0.00394766+0.j
          -0.02891657-0.0389701j ]
         [-0.16818875+0.2266635j   0.39819601+0.01030842j -0.02891657+0.0389701j
          0.59651444+0.j        ]]
    The attempt to generate a state with an uncorrect parameter:
    >>> print(create_random_pure_state_as_density_matrix(2, 4, 3))
        Traceback (most recent call last): ... ValueError: Option has to be: 0, 1 or 2
    """
    if o not in (0,1,2):
        raise ValueError('Option has to be: 0, 1 or 2')
        return None
    else:
        vs = create_random_qudit_pure_state(d,n,o)
        rho = np.outer(vs,np.matrix.conjugate(vs))
        return rho

def create_random_density_matrix_for_mixed_state(st_no, d, n, o=0):
    """
    Computes a random qudit mixed quantum state in a form of density matrix.
    Parameters
    ----------
    st_no : integer
        Number of states included in the final mixed state
    d : integer
        Describes the freedom level
    n : integer
        Describes the number of qudits
    o : interval
        Specifies elements of pure states what influences the elements of the
        final matrix. If o=0 (default value) the pure states are filled with
        complex numbers, but the imaginary part always equals 0. If o=1 the
        complex numbers fill pure states (there is a small propability that
        the imaginary part of the element equals 0, if the element is not on
        the main diagonal of the matrix). When o=2, the probability of obtaining
        element of a pure state with non-zero imaginary part is between 1/2 and 3/4.
    Returns
    -------
    rho : numpy array
        The density matrix of n-qudit mixed state (where the freedom level is
        specified by d).
    pr_list : list
        A list containing probalitities of appearing in the final mixed state
        for all pure states.
    state_list : list
        A list of pure (vector) states generating the final mixed state.
    Raises
    --------
    ValueError
        If o is not 0, 1 or 2.
    Examples
    --------
    Generation of an arbitrary density matrix representing 2-qubit mixed state
    obtained from three pure states (all imaginary parts equal zero):
    >>> print(create_random_density_matrix_for_mixed_state(3, 2, 2, 0))
        (array([[ 0.34287729+0.j, -0.27228613+0.j, -0.06334613+0.j,  -0.20054115+0.j],
                [ -0.27228613+0.j,  0.40277631+0.j,  0.01218101+0.j,  0.30655126+0.j],
                [ -0.06334613+0.j,  0.01218101+0.j,  0.02037116+0.j,  0.00752357+0.j],
                [-0.20054115+0.j,  0.30655126+0.j,  0.00752357+0.j,  0.23397524+0.j]]),
         [0.10348, 0.80429, 0.09223],
         [array([-0.23012278+0.j, -0.72641643+0.j,  0.16157678+0.j, -0.62710096+0.j]),
          array([ 0.6342675 +0.j, -0.61565374+0.j, -0.09272093+0.j, -0.45834271+0.j]),
          array([-0.38730957+0.j, -0.68536093+0.j,  0.34148274+0.j, -0.51347953+0.j])])
    Generation of an arbitrary density matrix representing 2-qubit mixed state
    obtained from two pure states (non-zero imaginary parts in pure states):
    >>> print(create_random_density_matrix_for_mixed_state(2, 2, 2, 1))
        (array([[ 0.09215768+0.j        , -0.10766898-0.04075363j,
                 0.10086219-0.12932354j, -0.15320487-0.09903645j],
                [-0.10766898+0.04075363j,  0.19914306+0.j        ,
                 -0.0905114 +0.23242563j,  0.24724033+0.06356434j],
                [ 0.10086219+0.12932354j, -0.0905114 -0.23242563j,
                 0.332369  +0.j        , -0.03153456-0.34803899j],
                [-0.15320487+0.09903645j,  0.24724033-0.06356434j,
                 -0.03153456+0.34803899j,  0.37633026+0.j]]),
         [0.02846, 0.97154],
         [array([ 0.40880063-0.31569002j,  0.53031417-0.46528147j,
                 -0.25197582+0.19943566j, -0.24288121+0.27063667j]),
          array([-0.02954542-0.29354636j,  0.18495111+0.39520847j,
                 0.41182811-0.4116769j ,  0.39211458+0.47929829j])])
    The attempt to generate a state with an uncorrect parameter:
    >>> print(create_random_density_matrix_for_mixed_state(3, 2, 2, 3))
        Traceback (most recent call last): ... ValueError: Option has to be: 0, 1 or 2
    """
    if o not in (0,1,2):
        raise ValueError('Option has to be: 0, 1 or 2')
        return None
    else:
        # generation of probabilities
        pr_list=[]
        i=0
        while i<(st_no-1):
            x=round(rd.uniform(0,1),5)
            if (x!=0 and x!=1):
                pr_list.append(x)
                i+=1
            else:
                continue
        pr_list.append(1.0)
        pr_list.sort()
        # normalisation
        i=1
        while i<(st_no):
            s=0
            for j in range(i):
                s+=pr_list[j]
            pr_list[i]=round(pr_list[i]-s,5)
            i+=1
        # pure states generation
        vec_state_list=[]
        m_state_list=[]
        for i in range(st_no):
            tmp = create_random_qudit_pure_state(d, n, o)
            vec_state_list.append(tmp)
            tmp2 = np.outer(tmp,np.matrix.conjugate(tmp))
            m_state_list.append(tmp2)
        # mixed state calculation
        rho = np.multiply(m_state_list[0],pr_list[0])
        i=1
        while i<st_no:
            rho+=np.multiply(m_state_list[i],pr_list[i])
            i+=1
    return rho, pr_list, vec_state_list

#o=0 - only real numbers, 1 - complex numbers, 2 - mixed numbers (~ 1/2 complex numbers)
def create_random_unitary_matrix(dim, o):
    F=np.zeros((dim,dim),dtype=complex)
    #Q=np.zeros((dim,dim),dtype=complex)
    if o==0:
        for i in range(dim):
            for j in range(dim):
                F[j,i]=complex(rd.uniform(0,1)/np.sqrt(2),0)
    elif o==1:
        for i in range(dim):
            for j in range(dim):
                F[j,i]=complex(rd.uniform(0,1),rd.uniform(0,1))/np.sqrt(2)
    elif o==2:
        for i in range(dim):
            for j in range(dim):
                a=rd.uniform(0,1)
                x=rd.randint(0,1)
                if x==0:
                    b=0
                else:
                    b=rd.uniform(0,1)
                F[j,i]=complex(a,b)/np.sqrt(2)
    else:
        raise ArgumentValueError('Option has to be: 0, 1 or 2')
        return None
    Q,R=np.linalg.qr(F)
    d=np.diagonal(R)
    ph=d/np.absolute(d)
    U=np.multiply(Q,ph,Q)
    return U


#
# small routine for better
# matrix display
#


def pretty_matrix_print(x, _pprecision=4):
    with np.printoptions(precision = _pprecision, suppress=True):
        print(x)

#
# partitions generators
#

def partititon_initialize_first(kappa,M):
    for i in range(0, len(kappa)):
        kappa[i]=0
        M[i]=0

def partititon_initialize_last(kappa,M):
    for i in range(0, len(kappa)):
        kappa[i]=i
        M[i]=i

def partititon_p_initialize_first(kappa, M, p):
    n=len(kappa)
    for i in range(0, n-p+1):
        kappa[i]=0
        M[i]=0
    for i in range(n-p+1, n, 1):
        kappa[i]=i-(n-p)
        M[i]=i-(n-p)

def partititon_p_initialize_last(kappa, M, p):
    n=len(kappa)
    for i in range(0, p):
        kappa[i]=i
        M[i]=i
    for i in range(p, n, 1):
        kappa[i]=p-1
        M[i]=p-1

def partition_size(M):
    n=len(M)
    return M[n-1]-M[0]+1

def partititon_disp(kappa):
        n=len(kappa)
        m=max(kappa)
        fstr=""
        for j in range(0, m+1):
                string='('
                for i in range(0,n):
                        if kappa[i]==j:
                                string=string+str(i)+','
                string=string[0:len(string)-1]
                string=string+')'
                fstr=fstr +string
        return '{'+fstr+'}'

def make_partititon_as_list(kappa):
        n=len(kappa)
        m=max(kappa)
        fstr=[]
        for j in range(0, m+1):
                string=[]
                for i in range(0,n):
                        if kappa[i]==j:
                                string=string+[i]
                fstr=fstr + [string]
        return fstr

def partition_next(kappa, M):
    n=len(kappa)
    for i in range(n-1, 0, -1):
        if kappa[i] <= M[i-1]:
            kappa[i]=kappa[i]+1
            M[i]=max(M[i], kappa[i])
            for j in range(i+1, n, 1):
                kappa[j]=kappa[0]
                M[j]=M[i]
            return True
    return False

def partition_p_next(kappa, M, p):
        n=len(kappa)
        p=partition_size(M)
        for i in range(n-1,0,-1):
                if kappa[i]<p-1 and kappa[i]<=M[i-1]:
                        kappa[i]=kappa[i]+1
                        M[i]=max(M[i], kappa[i])
                        for j in range(i+1, n-(p-M[i])+1):
                                kappa[j]=0
                                M[j]=M[i]
                        for j in range(n-(p-M[i])+1, n):
                                kappa[j]=p-(n-j)
                                M[j]=p-(n-j)
                        return True
        return False

def gen_all_k_elem_subset(k,n):
    A=[0]*(k+1)
    for i in range(1,k+1):
        A[i]=i
    if k >= n:
        return A[1:]
    output=[]
    p=k
    while p>=1:
        output = output +[A[1:k+1]]
        if A[k]==n:
            p=p-1
        else:
            p=k
        if p>=1:
            for i in range(k,p-1,-1):
                A[i]=A[p]+i-p+1
    return output

# integer l must be a power of 2 
def gen_two_partion_of_given_number(l):
    d = l / 2
    part = []
    r=l/2
    while r!=1:
            part=part + [[int(l/r), int(r)]]
            r=r/2
    return part

def filtered_data_for_paritition_division( r, idxtoremove ):
    rr = [ ]
    schmidtnumbers = [ ]
    #
    #  remove qubits that are not entangled
    #
    for i in r:
        for idx in idxtoremove:
            if idx in i[1][0]:
                i[1][0].remove(idx)
            #end if
            if idx in i[1][1]:
                i[1][1].remove(idx)
            #end if
        #end for
        #if len(i[1][0])>1 and len(i[1][1])>1:
        rr=rr+[i]
        if i[0] not in schmidtnumbers:
            schmidtnumbers=schmidtnumbers + [i[0]]
        #end if
    #end for
    
    # print 'schmidt numbers', schmidtnumbers
    #
    # sort by Schmidt rank
    #
    rr=sorted(rr)
    # print("sorted partitions")
    # for i in rr:
    #     print(i)
    #end for
    #
    # building a set of partitions
    #
    finalpart=set()
    if 1 in schmidtnumbers:
        for i in rr:
            if i[0]==1:
                if len(i[1][0])>1:
                    finalpart.add(tuple(i[1][0]))
                #end of if
                if len(i[1][1])>1:
                    finalpart.add(tuple(i[1][1]))
                #end of if
            #end of if
        #end of for i in rr
    #end of if
    
    if 1 not in schmidtnumbers:
        finalcluster=[]
        for i in rr:
            if i[0]==2:
                for e in i[1][0]:
                    if e not in finalcluster:
                        finalcluster = finalcluster + [ e ]
                for e in i[1][1]:
                    if e not in finalcluster:
                        finalcluster = finalcluster + [ e ]
        finalpart.add(tuple(finalcluster))
        #print('final cluster', finalcluster)
        #print('final part', finalpart)
    return finalpart

def bin2dec(s):
    return int(s, 2)

def ent_detection_by_paritition_division( q, nqubits, verbose = 0 ):
    #s = q.size
    s = nqubits
    # generation of all two partitite divisions of given
    # set which is build from the quantum register q
    res = [ ]
    idxtoremove = [ ]
    k = [0] * s
    M = [0] * s
    p = 2
    partititon_p_initialize_first(k, M, p)
    lp = []
    lp = lp + [make_partititon_as_list(k)]
    while partition_p_next(k, M, p):
        lp = lp + [make_partititon_as_list(k)]
    for i in lp:
            if verbose==1 or verbose==2:
                    print(i[0], i[1])
            mxv=2**len(i[0])
            myv=2**len(i[1])
            if verbose==1:
                    print(mxv,"x",myv)
            #m=qcs.Matrix(mxv, myv)
            m  = np.zeros((mxv, myv), dtype=complex)
            #mt=qcs.Matrix(mxv, myv)
            mt = np.zeros((mxv, myv), dtype=complex)
            for x in range(0,mxv):
                    for y in range(0, myv):
                            xstr=bin(x)[2:]
                            ystr=bin(y)[2:]
                            xstr='0'*(len(i[0])-len(xstr)) + xstr
                            ystr='0'*(len(i[1])-len(ystr)) + ystr
                            cstr=[0]*s
                            for xidx in range(0, len(xstr)):
                                    idx = i[0][xidx]
                                    cstr[idx]=xstr[xidx]
                            for yidx in range(0, len(ystr)):
                                    idx = i[1][yidx]
                                    cstr[idx]=ystr[yidx]
                            cidx=""
                            for c in cstr:
                                    cidx=cidx+c
                            dcidx=bin2dec(cidx)
                            dxidx=bin2dec(xstr)
                            dyidx=bin2dec(ystr)
                            if verbose==1:
                                    print("D("+xstr+","+ystr+")","D(",dxidx,dyidx,") C",dcidx,cidx,cstr)
                            #m.AtDirect(dxidx,dyidx, q.GetVecStateN(dcidx).Re(), q.GetVecStateN(dcidx).Im())
                            m[dxidx,dyidx] = q[dcidx]
                            #mt.AtDirect(dxidx,dyidx, q.GetVecStateN(dcidx).Re(), q.GetVecStateN(dcidx).Im())
                            mt[dxidx,dyidx] = q[dcidx]
            if verbose==1:
                    #m.PrMatlab()
                    print("m matrix")
                    print(m)
            #mf=m.Calc_D_dot_DT() # D * D'
            mf = m @ m.transpose()
            #sd=mf.SpectralDecomposition()
            #sd.eigenvalues.Chop()
            #ev_count=sd.eigenvalues.NonZeros()
            (ev,evec)=eigen_decomposition(mf)
            ev=chop(ev)
            ev_count = np.count_nonzero(ev)

            if verbose==1 or verbose==2:
                    print("non zero:", ev_count)
                    print("ev=",ev)
            if (ev_count==1) and (len(i[0])==1):
                idxtoremove=idxtoremove+i[0]
                
            if (ev_count==1) and (len(i[1])==1):
                idxtoremove=idxtoremove+i[1]
                
            res=res + [[ev_count, [i[0], i[1]]]]
            if verbose==1 or verbose==2:
                print()
    return res,idxtoremove

def detection_entanglement_by_paritition_division( q, nqubits, verbose = 0 ):
    [r,idxtoremove]=ent_detection_by_paritition_division( q, nqubits, verbose )
    #print("idx to remove", idxtoremove)
    #print("all partitions")
    # for i in r:
    #     print(i)
    fp = filtered_data_for_paritition_division( r, idxtoremove )
    # if len(fp)==0:
    #     print("register is fully separable")
    # else:
    #     print("raw final filtered data")
    #     for i in fp:
    #         print(i)
    
    cfp = set(fp)
    ffp = set(fp)
    
    for i in fp:
        if i in cfp:
            cfp.remove(i)
        for e in cfp:
            if (set(i) < set(e)) and (len(i)!=len(e)):
                if e in ffp:
                    ffp.remove(e)
    # print("final filtered data")
    # for i in ffp:
    #     print(i)

    return ffp

def entropy_by_paritition_division( q, nqubits, verbose = 0 ):
    #s = q.size
    s = nqubits
    # generation of all two partitite divisions of given
    # set which is build from the quantum register q
    total_entropy_val = 0.0
    total_negativity_val = 0.0
    entropy_val = 0.0
    res = [ ]
    idxtoremove = [ ]
    k = [0] * s
    M = [0] * s
    p = 2
    partititon_p_initialize_first(k, M, p)
    lp = []
    lp = lp + [make_partititon_as_list(k)]
    while partition_p_next(k, M, p):
        lp = lp + [make_partititon_as_list(k)]
    for i in lp:
            if verbose==1 or verbose==2:
                    print(i[0], i[1])
            mxv=2**len(i[0])
            myv=2**len(i[1])
            if verbose==1:
                    print(mxv,"x",myv)
            #m=qcs.Matrix(mxv, myv)
            m  = np.zeros((mxv, myv), dtype=complex)
            #mt=qcs.Matrix(mxv, myv)
            mt = np.zeros((mxv, myv), dtype=complex)
            for x in range(0,mxv):
                    for y in range(0, myv):
                            xstr=bin(x)[2:]
                            ystr=bin(y)[2:]
                            xstr='0'*(len(i[0])-len(xstr)) + xstr
                            ystr='0'*(len(i[1])-len(ystr)) + ystr
                            cstr=[0]*s
                            for xidx in range(0, len(xstr)):
                                    idx = i[0][xidx]
                                    cstr[idx]=xstr[xidx]
                            for yidx in range(0, len(ystr)):
                                    idx = i[1][yidx]
                                    cstr[idx]=ystr[yidx]
                            cidx=""
                            for c in cstr:
                                    cidx=cidx+c
                            dcidx=bin2dec(cidx)
                            dxidx=bin2dec(xstr)
                            dyidx=bin2dec(ystr)
                            if verbose==1:
                                    print("D("+xstr+","+ystr+")","D(",dxidx,dyidx,") C",dcidx,cidx,cstr)
                            #m.AtDirect(dxidx,dyidx, q.GetVecStateN(dcidx).Re(), q.GetVecStateN(dcidx).Im())
                            m[dxidx,dyidx] = q[dcidx]
                            #mt.AtDirect(dxidx,dyidx, q.GetVecStateN(dcidx).Re(), q.GetVecStateN(dcidx).Im())
                            mt[dxidx,dyidx] = q[dcidx]
            if verbose==1:
                    #m.PrMatlab()
                    print("m matrix")
                    print(m)
            #mf=m.Calc_D_dot_DT() # D * D'
            mf = m @ m.transpose()
            #sd=mf.SpectralDecomposition()
            #sd.eigenvalues.Chop()
            #ev_count=sd.eigenvalues.NonZeros()
            (ev,evec)=eigen_decomposition(mf)
            ev=chop(ev)
            ev_count = np.count_nonzero(ev)
            if (ev_count > 1):
                entropy_val=0.0
                negativity_val=0.0
                for ee in ev:
                    #e=np.sqrt(ee)
                    e=ee
                    if e != 0.0:
                        entropy_val = entropy_val + (e ** 2) * np.log((e ** 2))
                        #entropy_val = entropy_val + e * np.log2(e)
                        #entropy_val = entropy_val + e * np.log10(e)
                        negativity_val=negativity_val+e
                total_negativity_val += negativity_val
                total_entropy_val += -entropy_val
            
            if verbose==1 or verbose==2:
                    print("non zero:", ev_count)
                    print("ev=",ev)
            if (ev_count==1) and (len(i[0])==1):
                idxtoremove=idxtoremove+i[0]
                
            if (ev_count==1) and (len(i[1])==1):
                idxtoremove=idxtoremove+i[1]
                
            res=res + [[ev_count, [i[0], i[1]]]]
            if verbose==1 or verbose==2:
                print()
    return res,idxtoremove, total_entropy_val, 0.5*((total_negativity_val ** 2)-1.0)


def calculate_trace_norm(qden):
    pass

def calculate_l1norm_coherence(qden):
    pass

def bravyi_theorem_for_two_qubit_mixed_state_check(qden, rho_A, rho_B):
    """
    Function checks if qden represents 2-qubit mixed state rho_(AB) for
    margins rho_A and rho_B, according to the theorem from: S. Bravyi,
    Requirments for compatibility between local and multipartite quantum
    states, quant-ph/0301014

    Parameters
    ----------
    qden, rho_A, rho_B : numpy arrays
        The parameter qden represents a density matrix, and rho_A, rho_B
        are potential margins of qden

    Raises
    ------
    ValueError
        If state is not mixed.
    DimensionError
        If state is not 2-qubit state.

    Returns
    -------
    Boolean value
    eigenvalues_Bravyi: list
            The boolean value informs if rho_A and rho_B are margions of qden
            according to Bravyi theorem. In eigenvalues_Bravyi first two values
            are minimal eigenvalues of rho_A and rho_B, and the next four are
            eigenvalues of qden in descending order
    Examples
    --------
    Let rho_A and rho_B be arbitrary margins, and rho_C is 2-qubit mixed state
    >>> a = np.ndarray(shape=(2),dtype=complex)
    >>> b = np.ndarray(shape=(2),dtype=complex)
    >>> a[0]=1/math.sqrt(2)
    >>> a[1]=1/math.sqrt(2)
    >>> b[0]=0
    >>> b[1]=1
    >>> rho_A = vector_state_to_density_matrix(a)
    >>> rho_B = vector_state_to_density_matrix(b)
    >>> rho_C = create_mixed_state(2,2)
    >>> print(bravyi_theorem_for_two_qubit_mixed_state_check(rho_C,rho_A,rho_B))
        (False, [[0.0, 0.0], [0.25, 0.25, 0.25, 0.25]])
    Let rho_C be a 2-qubit pure state and rho_A, rho_B its margins
    >>> C = create_wstate(2)
    >>> rho_C = vector_state_to_density_matrix(C)
    >>> rho_A = partial_trace_main_routine(rho_C, [2, 2], axis=1)
    >>> rho_B = partial_trace_main_routine(rho_C, [2, 2], axis=0)
    >>> print(bravyi_theorem_for_two_qubit_mixed_state_check(rho_C, rho_A, rho_B))
        Traceback (most recent call last): ... ValueError: Not mixed state!
    Let rho_C be a 3-qubit state and rho_A, rho_B its margins
    >>> C = create_random_qudit_state(2, 3)
    >>> rho_C = vector_state_to_density_matrix(C)
    >>> rho_AB = partial_trace_main_routine(rho_C, [2, 2, 2], axis=2)
    >>> rho_B = partial_trace_main_routine(rho_AB, [2, 2], axis=0)
    >>> rho_A = partial_trace_main_routine(rho_AB, [2, 2], axis=1)
    >>> print(bravyi_theorem_for_two_qubit_mixed_state_check(rho_C, rho_A, rho_B))
        Traceback (most recent call last): ... DimensionError: Not 2-qubit state!
    Let rho_C be a 2-qubit mixed state and rho_A, rho_B its margins
    >>> rho_C = create_mixed_state(2, 2)
    >>> rho_A = partial_trace_main_routine(rho_C, [2, 2], axis=1)
    >>> rho_B = partial_trace_main_routine(rho_C, [2, 2], axis=0)
    >>> print(bravyi_theorem_for_two_qubit_mixed_state_check(rho_C, rho_A, rho_B))
        (True, [[0.5, 0.5], [0.25, 0.25, 0.25, 0.25]])
    """
    x=calculate_purity(qden)
    if (len(qden)==4 and len(qden[0])==4 and x[0]==False):
        eigenval, eigenvec = eigen_decomposition(qden)
        eigenval_A, eigenvec_A = eigen_decomposition(rho_A)
        eigenval_B, eigenvec_B = eigen_decomposition(rho_B)
        lambda_A = eigenval_A[0]
        lambda_B = eigenval_B[0]
        lambda_1 = eigenval[3]
        lambda_2 = eigenval[2]
        lambda_3 = eigenval[1]
        lambda_4 = eigenval[0]
        eigenvalues_Bravyi=[[lambda_A, lambda_B], [lambda_1, lambda_2, lambda_3, lambda_4]]
        if (lambda_A>=lambda_3+lambda_4 and lambda_B>=lambda_3+lambda_4 and lambda_A+lambda_B>=lambda_2+lambda_3+2*lambda_4 and abs(lambda_A-lambda_B<=min(lambda_1-lambda_3,lambda_2-lambda_4))):
            return True, eigenvalues_Bravyi
        else:
            return False, eigenvalues_Bravyi
    else:
        if not(len(qden)==4 and len(qden[0])==4):
            raise DimensionError("Not 2-qubit state!")
        else:
            raise ValueError("Not mixed state!")
        return False, None

def density_matrix_trace_check(qden):
    """
        Checks if qden is a correct density matrix (its trace equals one)

        Parameters
        ----------
        qden : numpy array

        Returns
        -------
        bool
            If the function returns 1, then qden is a density matrix (and 0 otherwise).
        
        Examples
        --------
        Check the correctness of a density matrix for the correct pure quantum state d
        >>> d = np.ndarray(shape=(2),dtype=float)
        >>> d[0]=1/math.sqrt(2)
        >>> d[1]=1/math.sqrt(2)
        >>> rho_D = vector_state_to_density_matrix(d)
        >>> print(density_matrix_trace_check(rho_D))
            True
        Check the correctness of a density matrix for the not correct quantum state d
        >>> d = np.ndarray(shape=(2),dtype=float)
        >>> d[0]=1/math.sqrt(2)
        >>> d[1]=0
        >>> rho_D = vector_state_to_density_matrix(d)
        >>> print(density_matrix_trace_check(rho_D))
            False
        Check the correctness of the density matrix rho_C for the correct mixed quantum state
        >>> rho_C = create_mixed_state(2,2)
        >>> print(density_matrix_trace_check(rho_C))
            True
    """
    if not (np.ndim(qden) == 2):
        raise DensityMatrixDimensionError("Argument is not two dimensional matrix!")
        return None
    x = np.trace(qden)
    if (math.isclose(np.real(x), 1, abs_tol=0.000001) and math.isclose(np.imag(x), 0, abs_tol=0.000001)):
        return True
    else:
        return False


def calculate_purity(qden):
    """
        Checks if qden is a density matrix of a pure state

        Parameters
        ----------
        qden : numpy array

        Returns
        -------
        t : tuple
            The first element of the tuple is Boolean and answers the question,
            if qden is a pure state density matrix. The second element of the tuple
            is the trace of squared qden (if its value equals 1, then qden represents
            a pure state; if its value is lower than 1, then qden represents
            a mixed state).
        
        Examples
        --------
        Check if rho_A is a density matrix of a pure state
        >>> A = create_max_entangled_pure_state(3)
        >>> rho_A = vector_state_to_density_matrix(A)
        >>> print(calculate_purity(rho_A))
            (True, 1)
        Check if rho_B is a density matrix of a pure state
        >>> rho_B = create_mixed_state(2,2)
        >>> print(calculate_purity(rho_B))
            (False, 0.25)
    """
    if density_matrix_trace_check(qden):
        x = np.matmul(qden,qden)
        y = np.trace(x)
        if (math.isclose(np.real(y), 1, abs_tol=0.000001) and math.isclose(np.imag(y), 0, abs_tol=0.000001)):
            t = True, 1
            return t
        else:
            t = False, y
            return t
    else:
        raise ArgumentValueError("The matrix is not a correct density matrix!")
        return None

# function to rewrite
def dec_to_base(num,base):  #Maximum base - 36
    base_num = ""
    while num>0:
        dig = int(num%base)
        if dig<10:
            base_num += str(dig)
        else:
            base_num += chr(ord('A')+dig-10)  #Using uppercase letters
        num //= base

    base_num = base_num[::-1]  #To reverse the string
    return base_num

def pr_state(q, N, d):
    idx=0
    for v in q:
        print(dec_to_base(idx, d).zfill(N), v)
        idx=idx+1

def version():
    pass

def about():
    pass

def how_to_cite():
    pass
