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
# *                                                                         *
# *   Part of the VQEClassification:                                        *
# *         https://github.com/qMSUZ/VQEClassification                      *
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

import numpy as np
import entdetector as ed

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import Aer
from qiskit.algorithms.optimizers import SPSA

import pandas as pd
from sympy import sqrt, I

import matplotlib.pyplot as plt

from sklearn import decomposition
from sklearn.cluster import KMeans

import time


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
            Numpy array of noemalized quantum states.

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
            Qtab[i,j]=sqrt(KNtab[i,j]/sum_all)
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
        Qvec[i]=sqrt(Qvec[i]/sum_all)
    return Qvec

def read_data():
    df = pd.read_excel (r'data/train-data.xlsx')
    
    j=1
    K=np.ndarray(shape=(80,8))
    while(j<5):
        x_pom=df["X"+str(j)]
        min1=x_pom[0]
        max1=x_pom[0]
        for i in range(80):
            if x_pom[i] < min1:
                min1=x_pom[i]
            if x_pom[i] > max1:
                max1=x_pom[i]
        zakres=max1-min1
        for i in range(80):
            K[i,(j-1)]=(x_pom[i]-min1)/zakres
        j+=1
    x5_pom=df["X5"]
    x6_pom=df["X6"]
    x7_pom=df["X7"]
    x8_pom=df["X8"]
    for i in range(80):
        K[i,4]=x5_pom[i]
        K[i,5]=x6_pom[i]
        K[i,6]=x7_pom[i]
        K[i,7]=x8_pom[i]
    
    Q=np.ndarray(shape=(80,8))
    labels_for_Q = np.ndarray(shape=(80,))
    Q0=np.ndarray(shape=(1,8))
    Q1=np.ndarray(shape=(1,8))
    for i in range(80):
        sum_all=0
        for j in range(8):
            sum_all+=K[i,j]
        for j in range(8):
            Q[i,j]=sqrt(K[i,j]/sum_all)
    for i in range(80):
        suma=0
        for j in range(8):
            suma+=Q[i,j]*Q[i,j]
    for i in range(80):  
        labels_for_Q[i]=df.Y[i]
        if df.Y[i] == 0:
            Q0=np.vstack((Q0, Q[i]));
        if df.Y[i] == 1:
            Q1=np.vstack((Q1, Q[i]));
    
    Q0 = np.delete(Q0, (0), axis=0)
    Q1 = np.delete(Q1, (0), axis=0)
     
    return df, Q, labels_for_Q, Q0, Q1

def vector_to_distro(v):
    map_output_distr = {
                0: v[0] ** 2,
                1: v[1] ** 2,
                2: v[2] ** 2,
                3: v[3] ** 2,
                4: v[4] ** 2,
                5: v[5] ** 2,
                6: v[6] ** 2,
                7: v[7] ** 2}

    return map_output_distr

def array_to_dict(v):
    map_output_distr = {
                0: v[0],
                1: v[1],
                2: v[2],
                3: v[3],
                4: v[4],
                5: v[5],
                6: v[6],
                7: v[7]}

    return map_output_distr

            

def counts_to_distr(counts):
    num_of_shots = sum(counts.values())
    return {int(k, 2): v/num_of_shots for k, v in counts.items()}

def create_variational_circuit(qubits, parameters, formval, layers):
    
    circ = QuantumCircuit(len(qubits), len(qubits))
    
    offsetidx=0

# ----------------------------------- form 0
# 
    if formval == 0:
        for idx in range (0, len(qubits)):
            circ.ry(parameters[offsetidx  + idx], qubits[0 + idx])
     
        offsetidx=offsetidx+len(qubits)
        
        circ.barrier()
        
        circ.cz(0,1)
        circ.cz(2,0)
    
        circ.barrier()
    
    
        for idx in range (0, len(qubits)):
            circ.ry(parameters[offsetidx + idx], qubits[0 + idx])
    
        offsetidx=offsetidx+len(qubits)
    
        circ.barrier()
    
        circ.cz(1,2)
        circ.cz(2,0)
    
        circ.barrier()
    
        for idx in range (0, len(qubits)):
            circ.ry(parameters[offsetidx + idx], qubits[0 + idx])
    
        offsetidx=offsetidx+len(qubits)
    
        circ.barrier()
    
        for idx in range (0, len(qubits)):
            circ.measure(qubits[idx], qubits[idx])

# ----------------------------------- form 1    
# linear entanglement
    if formval == 1:

        for idx in range (0, len(qubits)):
            circ.ry(parameters[offsetidx  + idx], qubits[0 + idx])

        offsetidx=offsetidx+len(qubits)

        for idx in range (0, len(qubits)):
            circ.rz(parameters[offsetidx  + idx], qubits[0 + idx])

        offsetidx=offsetidx+len(qubits)

        circ.barrier()

        for idx in range (0, len(qubits)-1):
            circ.cx(idx, idx+1)

        circ.barrier()

        for idx in range (0, len(qubits)):
            circ.ry(parameters[offsetidx  + idx], qubits[0 + idx])

        offsetidx=offsetidx+len(qubits)

        for idx in range (0, len(qubits)):
            circ.rz(parameters[offsetidx  + idx], qubits[0 + idx])

        offsetidx=offsetidx+len(qubits)

        circ.barrier()

        for idx in range (0, len(qubits)-1):
            circ.cx(idx, idx+1)

        circ.barrier()


        for idx in range (0, len(qubits)):
            circ.measure(qubits[idx], qubits[idx])
        
# ----------------------------------- form 2
# full entanglement
    if formval == 2:

        for idx in range (0, len(qubits)):
            circ.ry(parameters[offsetidx  + idx], qubits[0 + idx])

        offsetidx=offsetidx+len(qubits)

        for idx in range (0, len(qubits)):
            circ.rz(parameters[offsetidx  + idx], qubits[0 + idx])

        offsetidx=offsetidx+len(qubits)

        circ.barrier()

        for idx in range (0, len(qubits)-1):
            circ.cx(qubits[idx], qubits[idx+1])

        circ.cx(qubits[0], qubits[len(qubits)-1])

        circ.barrier()

        for idx in range (0, len(qubits)):
            circ.ry(parameters[offsetidx  + idx], qubits[0 + idx])

        offsetidx=offsetidx+len(qubits)

        for idx in range (0, len(qubits)):
            circ.rz(parameters[offsetidx  + idx], qubits[0 + idx])

        offsetidx=offsetidx+len(qubits)

        circ.barrier()

        for idx in range (0, len(qubits)-1):
            circ.cx(qubits[idx], qubits[idx+1])

        circ.cx(qubits[0], qubits[len(qubits)-1])

        circ.barrier()

        for idx in range (0, len(qubits)):
            circ.measure(qubits[idx], qubits[idx])

# ----------------------------------- form 3
# 
    if formval == 3:    
        for _ in range(0, layers):
            for idx in range (0, len(qubits)):
                circ.rx(parameters[offsetidx  + idx], qubits[0 + idx])
    
            offsetidx=offsetidx+len(qubits)
    
            for idx in range (0, len(qubits)):
                circ.rz(parameters[offsetidx  + idx], qubits[0 + idx])
    
            offsetidx=offsetidx+len(qubits)
    
            for idx in range (0, len(qubits)):
                circ.rx(parameters[offsetidx  + idx], qubits[0 + idx])
    
            offsetidx=offsetidx+len(qubits)
    
            circ.barrier()
    
            for idx in range (0, len(qubits)-1):
                circ.cx(idx, idx+1)
    
            circ.barrier()

        for idx in range (0, len(qubits)):
            circ.measure(qubits[idx], qubits[idx])

    if formval == 4:

        for idx in range (0, len(qubits)):
             circ.ry(parameters[offsetidx  + idx], qubits[0 + idx])
 
        offsetidx=offsetidx+len(qubits)

        circ.barrier()

        for _ in range(0, layers):

            for idx in range (0, len(qubits)):
                 circ.ry(parameters[offsetidx  + idx], qubits[0 + idx])
    
            offsetidx=offsetidx+len(qubits)

            circ.cz(0, 1)
                  
            for idx in range (0, len(qubits)):
                 circ.ry(parameters[offsetidx  + idx], qubits[0 + idx])

            offsetidx=offsetidx+len(qubits)
    
            circ.cz(1, 2)
            circ.cz(0, 2)
    
            circ.barrier()

        for idx in range (0, len(qubits)):
             circ.ry(parameters[offsetidx  + idx], qubits[0 + idx])

        offsetidx=offsetidx+len(qubits)

        circ.barrier()

        for idx in range (0, len(qubits)):
            circ.measure(qubits[idx], qubits[idx])
            
    if formval == 5:
        for idx in range (0, len(qubits)):
             circ.ry(parameters[offsetidx  + idx], qubits[0 + idx])
         
        offsetidx=offsetidx+len(qubits)
        
        circ.barrier()
        
        for _ in range(0, layers):           
            circ.ry(parameters[offsetidx], qubits[0])
            offsetidx=offsetidx+1
            
            circ.cx(qubits[0],qubits[2])
            
            circ.barrier()
            
            circ.ry(parameters[offsetidx], qubits[0])
            offsetidx=offsetidx+1
            
            circ.cx(qubits[0],qubits[1])
            
            circ.ry(parameters[offsetidx], qubits[1])
            offsetidx=offsetidx+1
            
            circ.cx(qubits[1],qubits[2])
            
            circ.barrier()
        
        for idx in range (0, len(qubits)):
             circ.ry(parameters[offsetidx  + idx], qubits[0 + idx])
         
        offsetidx=offsetidx+len(qubits)        
        
        for idx in range (0, len(qubits)):
            circ.measure(qubits[idx], qubits[idx])

            
    return circ

def get_vectors_for_label(l, labels, data, n_data):
    outlist = []
    for idx in range(0, n_data):
        if l == labels[idx]:
            outlist.append(data[idx] )
    
    return np.array( outlist )


def get_vector_of_idx_for_label(l, labels, data, n_data):
    out_idx_list = [] 

    for idx in range(0, n_data):
        if l == labels[idx]:
            out_idx_list.append( idx )
    
    return np.array( out_idx_list )


def objective_function(parameters):
    
    global backend
    
    cost_value = 0
    qc = create_variational_circuit(qubits, parameters, circuit_type, layers)
    result = backend.run(qc).result()
    output_distr = counts_to_distr(result.get_counts())

    cost_value = sum(abs(target_distr.get(i, 0) - output_distr.get(i, 0))
            for i in range(2**qc.num_qubits) )

    return cost_value

def create_vqe_for_state(qubits, target_state, params, circuit_type, layers, verbose=0):
    
    global target_distr
    global backend
    
    target_distr = {0: target_state[0] ** 2,
                    1: target_state[1] ** 2,
                    2: target_state[2] ** 2,
                    3: target_state[3] ** 2,
                    4: target_state[4] ** 2,
                    5: target_state[5] ** 2,
                    6: target_state[6] ** 2,
                    7: target_state[7] ** 2}
    
    circ = create_variational_circuit(qubits, params, circuit_type, layers)
    #circ.draw()

    #optimizer = COBYLA(maxiter=500, tol=0.000001)
    optimizer = SPSA(maxiter=200)
    #optimizer = SLSQP(maxiter=500)
    #optimizer = POWELL( maxiter = 500 )

    result = optimizer.minimize(
        fun=objective_function,
        x0=params)

    if verbose>0:
        print("Parameters Found:", result.x)
        print("")
    
    return result.x


def train_vqe(params, state_for_train, filename, save_params=1):
    global backend
    
    target_params = create_vqe_for_state(qubits, 
                                         state_for_train, 
                                         params, 
                                         circuit_type, layers)
    
    #qc = create_variational_circuit(qubits, result.x, circuit_type, layers)
    
    qc = create_variational_circuit(qubits, target_params, circuit_type, layers)
    #qc.draw()
    #qc.draw(output='mpl')
    #qc.draw(output='latex')
    #qc.draw(output='latex_source')
    
    counts = backend.run(qc, shots=10000).result().get_counts()
    output_distribution = counts_to_distr(counts)
    
    output_distribution_as_array=np.zeros( 2 ** qc.num_qubits) 
    
    for i in range(2**qc.num_qubits):
        output_distribution_as_array[i] = output_distribution.get(i,0)
    
    if save_params==1:
        np.save( filename, target_params )
        np.save( filename+"_od",  output_distribution_as_array )

    return qc, target_params, output_distribution


def train_data_and_save_angles_to_file():
    #
    # VQE for class 0 case 0
    # 
    
    filename='angles_Q0_case_0_vqe_q{}_l{}_type{}'.format(len(qubits),layers, circuit_type)
    target_distr=vector_to_distro(Q0_cluster0_mean)
    qc, target_params, output_distribution = train_vqe(
                                    params, 
                                    Q0_cluster0_mean, 
                                    filename)
    
    #
    # VQE for class 0 case 1
    # 
    filename='angles_Q0_case_1_vqe_q{}_l{}_type{}'.format(len(qubits),layers, circuit_type)
    target_distr=vector_to_distro(Q0_cluster1_mean)
    qc, target_params, output_distribution = train_vqe(
                                    params, 
                                    Q0_cluster1_mean, 
                                    filename)
    
    #
    # VQE for class 0 case 2
    # 
    filename='angles_Q0_case_2_vqe_q{}_l{}_type{}'.format(len(qubits),layers, circuit_type)
    target_distr=vector_to_distro(Q0_cluster2_mean)
    qc, target_params, output_distribution = train_vqe(
                                    params, 
                                    Q0_cluster2_mean, 
                                    filename)
    
    #
    # VQE for class 0 case 3
    # 
    filename='angles_Q0_case_3_vqe_q{}_l{}_type{}'.format(len(qubits),layers, circuit_type)
    target_distr=vector_to_distro(Q0_cluster3_mean)
    qc, target_params, output_distribution = train_vqe(
                                    params, 
                                    Q0_cluster3_mean, 
                                    filename)
    
    #
    # VQE for class 0 case 4
    # 
    filename='angles_Q0_case_4_vqe_q{}_l{}_type{}'.format(len(qubits),layers, circuit_type)
    target_distr=vector_to_distro(Q0_cluster4_mean)
    qc, target_params, output_distribution = train_vqe(
                                    params, 
                                    Q0_cluster4_mean, 
                                    filename)
    
    #
    # VQE for class 0 case 5
    #
    filename='angles_Q0_case_5_vqe_q{}_l{}_type{}'.format(len(qubits),layers, circuit_type)
    target_distr=vector_to_distro(Q0_cluster5_mean)
    qc, target_params, output_distribution = train_vqe(
                                    params, 
                                    Q0_cluster5_mean, 
                                    filename)

    #
    # VQE for class 0 case 6
    #
    filename='angles_Q0_case_6_vqe_q{}_l{}_type{}'.format(len(qubits),layers, circuit_type)
    target_distr=vector_to_distro(Q0_cluster6_mean)
    qc, target_params, output_distribution = train_vqe(
                                    params, 
                                    Q0_cluster6_mean, 
                                    filename)

    
    #
    # VQE for class 1 case 0
    #
    filename='angles_Q1_case_0_vqe_q{}_l{}_type{}'.format(len(qubits),layers, circuit_type)
    target_distr=vector_to_distro(Q1_cluster0_mean)
    qc, target_params, output_distribution = train_vqe(
                                    params, 
                                    Q1_cluster0_mean, 
                                    filename)
    
    
    #
    # VQE for class 1 case 1
    #
    filename='angles_Q1_case_1_vqe_q{}_l{}_type{}'.format(len(qubits),layers, circuit_type)
    target_distr=vector_to_distro(Q1_cluster1_mean)
    qc, target_params, output_distribution = train_vqe(
                                    params,
                                    Q1_cluster1_mean,
                                    filename)
    
    
    #
    # VQE for class 1 case 2
    #
    filename='angles_Q1_case_2_vqe_q{}_l{}_type{}'.format(len(qubits),layers, circuit_type)
    target_distr=vector_to_distro(Q1_cluster2_mean)
    qc, target_params, output_distribution = train_vqe(
                                    params,
                                    Q1_cluster2_mean,
                                    filename)
    
    #
    # VQE for class 1 case 3
    #
    filename='angles_Q1_case_3_vqe_q{}_l{}_type{}'.format(len(qubits),layers, circuit_type)
    target_distr=vector_to_distro(Q1_cluster3_mean)
    qc, target_params, output_distribution = train_vqe(
                                    params,
                                    Q1_cluster3_mean,
                                    filename)

    #
    # VQE for class 1 case 4
    #
    filename='angles_Q1_case_4_vqe_q{}_l{}_type{}'.format(len(qubits),layers, circuit_type)
    target_distr=vector_to_distro(Q1_cluster4_mean)
    qc, target_params, output_distribution = train_vqe(
                                    params,
                                    Q1_cluster4_mean,
                                    filename)

    #
    # VQE for class 1 case 5
    #
    filename='angles_Q1_case_5_vqe_q{}_l{}_type{}'.format(len(qubits),layers, circuit_type)
    target_distr=vector_to_distro(Q1_cluster5_mean)
    qc, target_params, output_distribution = train_vqe(
                                    params,
                                    Q1_cluster5_mean,
                                    filename)

    #
    # VQE for class 1 case 6
    #
    filename='angles_Q1_case_6_vqe_q{}_l{}_type{}'.format(len(qubits),layers, circuit_type)
    target_distr=vector_to_distro(Q1_cluster6_mean)
    qc, target_params, output_distribution = train_vqe(
                                    params,
                                    Q1_cluster6_mean,
                                    filename)


def load_angles(file_name):
    tp_q0c0 = np.load( file_name+".npy" )
    tp_q0c0_od = np.load( file_name+"_od.npy")

    return tp_q0c0, tp_q0c0_od


def test_data(_class, _case, _n_case, verbose=0):

    _ngoodprobe=0
    _nfalseprobe=0   

    filename='angles_Q{}_case_{}_vqe_q{}_l{}_type{}'.format(_class, _case, len(qubits),layers, circuit_type)
        
    target_params, output_distribution_loaded = load_angles(filename)
    output_distribution = array_to_dict(output_distribution_loaded)
    
    
    vq0=np.zeros(40)
    
    for idx in range(0, 40):
        val=0
        current_probe=vector_to_distro(Q0[idx])
        for i in range(0, 2**len(qubits)):
            val += abs(current_probe.get(i,0) - output_distribution.get(i,0))    
        vq0[idx]=val
    
    vq1=np.zeros(40)
    
    for idx in range(0, 40):
        val=0
        current_probe=vector_to_distro(Q1[idx])
        for i in range(0, 2**len(qubits)):
            val += abs(current_probe.get(i,0) - output_distribution.get(i,0))    
        vq1[idx]=val
        
    if verbose>=2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(vq0, 'b.')
        ax.plot(vq1, 'r.')
        #ax.set_ylim(0, max(maxp0, maxp1)+0.25)
        
        plt.show()
    
    # prediction
    
    if verbose>=1:
        print()

    _threshold_one = 0.5
    _threshold_two = 0.5

         
    vq0_predict=np.zeros(40)
    vq1_predict=np.zeros(40)
    
    for idx in range(0, 40):
        if vq0[idx]<_threshold_one:
            vq0_predict[idx]=1
    
    vq0_predict_idx = np.where( vq0_predict == 1 )


    for idx in range(0, 40):
        if vq1[idx]<_threshold_two:
            vq1_predict[idx]=1
    
    vq1_predict_idx = np.where( vq1_predict == 1 )

    if  _class==0:
        _ngoodprobe=sum(vq0_predict)

    if  _class==1:
        _ngoodprobe=sum(vq1_predict)
   
    if verbose>=1 and _class==0:
        print("Accuracy in class %d for case %d: %0.1f%% (%d/%d)" % ( _class, _case, (_ngoodprobe / _n_case)*100, sum(vq0_predict),_n_case)) 

    if verbose>=1 and _class==1:
        print("Accuracy in class %d for case %d: %0.1f%% (%d/%d)" % ( _class, _case, (_ngoodprobe / _n_case)*100, sum(vq1_predict),_n_case)) 
        
    if verbose>=1:
        print("predicted indexes:")
        if _class==0:
            print("vq0_predict_idx=",vq0_predict_idx)

        if _class==1:
            print("vq1_predict_idx=",vq1_predict_idx)

        if _class==0 and _case==0:
            print("Q0_cluster0_idx=",Q0_cluster0_idx)
    
        if _class==0 and _case==1:
            print("Q0_cluster1_idx=",Q0_cluster1_idx)
    
        if _class==0 and _case==2:
            print("Q0_cluster2_idx=",Q0_cluster2_idx)
    
        if _class==0 and _case==3:
            print("Q0_cluster3_idx=",Q0_cluster3_idx)
    
        if _class==0 and _case==4:
            print("Q0_cluster4_idx=",Q0_cluster4_idx)
    
        if _class==0 and _case==5:
            print("Q0_cluster5_idx=",Q0_cluster5_idx)
            
        if _class==0 and _case==6:
            print("Q0_cluster6_idx=",Q0_cluster6_idx)
    
    
    
        if _class==1 and _case==0:
            print("Q1_cluster0_idx=",Q1_cluster0_idx)
    
        if _class==1 and _case==1:
            print("Q1_cluster1_idx=",Q1_cluster1_idx)
    
        if _class==1 and _case==2:
            print("Q1_cluster2_idx=",Q1_cluster2_idx)
    
        if _class==1 and _case==3:
            print("Q1_cluster3_idx=",Q1_cluster3_idx)

        if _class==1 and _case==4:
            print("Q1_cluster4_idx=",Q1_cluster4_idx)

        if _class==1 and _case==5:
            print("Q1_cluster5_idx=",Q1_cluster5_idx)

        if _class==1 and _case==6:
            print("Q1_cluster6_idx=",Q1_cluster6_idx)
    
        print("")

        if  _class==0:
            _nfalseprobe=sum(vq1_predict)

        if  _class==1:
            _nfalseprobe=sum(vq0_predict)


        if verbose>=1 and _class==0:    
            print("False probes in all class 1: %0.1f%% (%d/40)" % ((sum(vq1_predict) / 40)*100, sum(vq1_predict))) 
    
        if verbose>=1 and _class==1:    
            print("False probes in all class 0: %0.1f%% (%d/40)" % ((sum(vq0_predict) / 40)*100, sum(vq0_predict))) 
    
        if _class==1:
            print("predicted indexes for all class 0")
            print("vq1_predict_idx=",vq0_predict_idx)
    
        if _class==0:
            print("predicted indexes for all class 1")
            print("vq1_predict_idx=",vq1_predict_idx)
    
    return _ngoodprobe, _nfalseprobe


def test_data_tuned(_class, _case, _n_case, verbose=0):

    _ngoodprobe=0
    _nfalseprobe=0   

    filename='angles_Q{}_case_{}_vqe_q{}_l{}_type{}'.format(_class, _case, len(qubits),layers, circuit_type)
        
    target_params, output_distribution_loaded = load_angles('data/'+filename)
    output_distribution = array_to_dict(output_distribution_loaded)
    
    
    vq0=np.zeros(40)
    
    for idx in range(0, 40):
        val=0
        current_probe=vector_to_distro(Q0[idx])
        for i in range(0, 2**len(qubits)):
            val += abs(current_probe.get(i,0) - output_distribution.get(i,0))    
        vq0[idx]=val
    
    vq1=np.zeros(40)
    
    for idx in range(0, 40):
        val=0
        current_probe=vector_to_distro(Q1[idx])
        for i in range(0, 2**len(qubits)):
            val += abs(current_probe.get(i,0) - output_distribution.get(i,0))    
        vq1[idx]=val
    
    if verbose>=2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(vq0, 'b.')
        ax.plot(vq1, 'r.')
        
        plt.show()
    
    # prediction
    
    if verbose>=1:
        print()

    _threshold_one = 0.0
    _threshold_two = 0.0

    if _class==0 and _case==0:
        _threshold_one = 0.2
        _threshold_two = 0.2

    if _class==0 and _case==1:
        _threshold_one = 0.6
        _threshold_two = 0.6


    if _class==0 and _case==2:
        _threshold_one = 0.3
        _threshold_two = 0.15

    if _class==0 and _case==3:
        _threshold_one = 0.45
        _threshold_two = 0.45

    if _class==0 and _case==4:
        _threshold_one = 0.5
        _threshold_two = 0.5

    if _class==0 and _case==5:
        _threshold_one = 0.5
        _threshold_two = 0.5

    if _class==0 and _case==6:
        _threshold_one = 0.5
        _threshold_two = 0.5


    if _class==1 and _case==0:
        _threshold_one = 0.2
        _threshold_two = 0.4

    if _class==1 and _case==1:
        _threshold_one = 0.15
        _threshold_two = 0.3

    if _class==1 and _case==2:
        _threshold_one = 0.2
        _threshold_two = 0.4

    if _class==1 and _case==3:
        _threshold_one = 0.05
        _threshold_two = 0.4

    if _class==1 and _case==4:
        _threshold_one = 0.05
        _threshold_two = 0.2

    if _class==1 and _case==5:
        _threshold_one = 0.05
        _threshold_two = 0.5

    if _class==1 and _case==6:
        _threshold_one = 0.05
        _threshold_two = 0.2

          
    vq0_predict=np.zeros(40)
    vq1_predict=np.zeros(40)
    
    for idx in range(0, 40):
        if vq0[idx]<_threshold_one:
            vq0_predict[idx]=1
    
    vq0_predict_idx = np.where( vq0_predict == 1 )


    for idx in range(0, 40):
        if vq1[idx]<_threshold_two:
            vq1_predict[idx]=1
    
    vq1_predict_idx = np.where( vq1_predict == 1 )

    if  _class==0:
        _ngoodprobe=sum(vq0_predict)

    if  _class==1:
        _ngoodprobe=sum(vq1_predict)
   
    if verbose>=1 and _class==0:
        print("Accuracy in class %d for case %d: %0.1f%% (%d/%d)" % ( _class, _case, (_ngoodprobe / _n_case)*100, sum(vq0_predict),_n_case)) 

    if verbose>=1 and _class==1:
        print("Accuracy in class %d for case %d: %0.1f%% (%d/%d)" % ( _class, _case, (_ngoodprobe / _n_case)*100, sum(vq1_predict),_n_case)) 
        
    if verbose>=1:
        print("predicted indexes:")
        if _class==0:
            print("vq0_predict_idx=",vq0_predict_idx)

        if _class==1:
            print("vq1_predict_idx=",vq1_predict_idx)
    
    
    
        if _class==0 and _case==0:
            print("Q0_cluster0_idx=",Q0_cluster0_idx)
    
        if _class==0 and _case==1:
            print("Q0_cluster1_idx=",Q0_cluster1_idx)
    
        if _class==0 and _case==2:
            print("Q0_cluster2_idx=",Q0_cluster2_idx)
    
        if _class==0 and _case==3:
            print("Q0_cluster3_idx=",Q0_cluster3_idx)
    
        if _class==0 and _case==4:
            print("Q0_cluster4_idx=",Q0_cluster4_idx)
    
        if _class==0 and _case==5:
            print("Q0_cluster5_idx=",Q0_cluster5_idx)
            
        if _class==0 and _case==6:
            print("Q0_cluster6_idx=",Q0_cluster6_idx)
    
    
    
        if _class==1 and _case==0:
            print("Q1_cluster0_idx=",Q1_cluster0_idx)
    
        if _class==1 and _case==1:
            print("Q1_cluster1_idx=",Q1_cluster1_idx)
    
        if _class==1 and _case==2:
            print("Q1_cluster2_idx=",Q1_cluster2_idx)
    
        if _class==1 and _case==3:
            print("Q1_cluster3_idx=",Q1_cluster3_idx)

        if _class==1 and _case==4:
            print("Q1_cluster4_idx=",Q1_cluster4_idx)

        if _class==1 and _case==5:
            print("Q1_cluster5_idx=",Q1_cluster5_idx)

        if _class==1 and _case==6:
            print("Q1_cluster6_idx=",Q1_cluster6_idx)

        print("")

        if  _class==0:
            _nfalseprobe=sum(vq1_predict)

        if  _class==1:
            _nfalseprobe=sum(vq0_predict)


        if verbose>=1 and _class==0:    
            print("False probes in all class 1: %0.1f%% (%d/40)" % ((sum(vq1_predict) / 40)*100, sum(vq1_predict))) 
    
        if verbose>=1 and _class==1:    
            print("False probes in all class 0: %0.1f%% (%d/40)" % ((sum(vq0_predict) / 40)*100, sum(vq0_predict))) 
    
        if _class==1:
            print("predicted indexes for all class 0")
            print("vq1_predict_idx=",vq0_predict_idx)
    
        if _class==0:
            print("predicted indexes for all class 1")
            print("vq1_predict_idx=",vq1_predict_idx)
    
    return _ngoodprobe, _nfalseprobe


def simple_print_state( state, fullregdisp=False ):
    i=0
    print( "d bbbb Prob.Amplitude")
    print( "  012  " )
    for a in state:
        if a != 0 and fullregdisp==False:
            print( i, "{0:04b}".format(i), a)
        i=i+1

def entanglement_detection_in_data(Q0, Q1):
    nqubits = 3
          
    for idx in range(0, 40):
        part = ed.detection_entanglement_by_paritition_division(Q0[idx], nqubits)
        if len(part)==0:
            print("Q0: part is empty for:", Q0[idx], "idx=",idx)
        else:
            print("Q0 part:", part)

    for idx in range(0, 40):
        part = ed.detection_entanglement_by_paritition_division(Q1[idx], nqubits)
        if len(part)==0:
            print("Q1: part is empty for:", Q1[idx], "idx=",idx)
        else:
            print("Q1 part:", part)


print("read data")
df, Q, labels_for_Q, Q0, Q1 = read_data()

# Pearson Correlation Coefficient  (PCC)
pcc_rho = np.corrcoef( Q[:,0:7].transpose() )

print("Pearson Correlation Coefficients")
print(pcc_rho)

#
# alternate calculations of PCC
#
#pom = pd.DataFrame(Q, columns = ['x1','x2','x3','x4','x5','x6','x7','x8'])
#print(pom.corr())


pca = decomposition.PCA(n_components=2)
Q0_r = pca.fit(Q0).transform(Q0)
Q1_r = pca.fit(Q1).transform(Q1)

random_state = 170
common_params = {
    "n_init": "auto",
    "random_state": random_state,
}

cluster_for_Q0_r = KMeans(n_clusters=7, tol=1e-7, **common_params).fit(Q0_r)
cluster_for_Q1_r = KMeans(n_clusters=7, tol=1e-7, **common_params).fit(Q1_r)

cluster_for_Q0 = KMeans(n_clusters=7, **common_params).fit(Q0)
cluster_for_Q1 = KMeans(n_clusters=7, **common_params).fit(Q1)

Q0_cluster0 = get_vectors_for_label(0, cluster_for_Q0.labels_, Q0, 40)
Q0_cluster1 = get_vectors_for_label(1, cluster_for_Q0.labels_, Q0, 40)
Q0_cluster2 = get_vectors_for_label(2, cluster_for_Q0.labels_, Q0, 40)
Q0_cluster3 = get_vectors_for_label(3, cluster_for_Q0.labels_, Q0, 40)
Q0_cluster4 = get_vectors_for_label(4, cluster_for_Q0.labels_, Q0, 40)
Q0_cluster5 = get_vectors_for_label(5, cluster_for_Q0.labels_, Q0, 40)
Q0_cluster6 = get_vectors_for_label(6, cluster_for_Q0.labels_, Q0, 40) 

Q0_cluster0_idx =  get_vector_of_idx_for_label(0, cluster_for_Q0.labels_, Q0, 40)
Q0_cluster1_idx =  get_vector_of_idx_for_label(1, cluster_for_Q0.labels_, Q0, 40)
Q0_cluster2_idx =  get_vector_of_idx_for_label(2, cluster_for_Q0.labels_, Q0, 40)
Q0_cluster3_idx =  get_vector_of_idx_for_label(3, cluster_for_Q0.labels_, Q0, 40)
Q0_cluster4_idx =  get_vector_of_idx_for_label(4, cluster_for_Q0.labels_, Q0, 40)
Q0_cluster5_idx =  get_vector_of_idx_for_label(5, cluster_for_Q0.labels_, Q0, 40)
Q0_cluster6_idx =  get_vector_of_idx_for_label(6, cluster_for_Q0.labels_, Q0, 40)

Q0_cluster0_mean = np.mean(Q0_cluster0, axis=0)
Q0_cluster1_mean = np.mean(Q0_cluster1, axis=0)
Q0_cluster2_mean = np.mean(Q0_cluster2, axis=0)
Q0_cluster3_mean = np.mean(Q0_cluster3, axis=0)
Q0_cluster4_mean = np.mean(Q0_cluster4, axis=0)
Q0_cluster5_mean = np.mean(Q0_cluster5, axis=0)
Q0_cluster6_mean = np.mean(Q0_cluster6, axis=0)

n_Q0_cluster0 = Q0_cluster0.shape[0]
n_Q0_cluster1 = Q0_cluster1.shape[0]
n_Q0_cluster2 = Q0_cluster2.shape[0]
n_Q0_cluster3 = Q0_cluster3.shape[0]
n_Q0_cluster4 = Q0_cluster4.shape[0]
n_Q0_cluster5 = Q0_cluster5.shape[0]
n_Q0_cluster6 = Q0_cluster6.shape[0]

print( "Mean Q0_cluster0_mean =", Q0_cluster0_mean )
print( "Norm =", np.linalg.norm(Q0_cluster0_mean))
print( "Mean Q0_cluster1_mean =", Q0_cluster1_mean )
print( "Norm =", np.linalg.norm(Q0_cluster1_mean))
print( "Mean Q0_cluster2_mean =", Q0_cluster2_mean )
print( "Norm =", np.linalg.norm(Q0_cluster2_mean))
print( "Mean Q0_cluster3_mean =", Q0_cluster3_mean )
print( "Norm =", np.linalg.norm(Q0_cluster3_mean))
print( "Mean Q0_cluster4_mean =", Q0_cluster4_mean )
print( "Norm =", np.linalg.norm(Q0_cluster4_mean))
print( "Mean Q0_cluster5_mean =", Q0_cluster5_mean )
print( "Norm =", np.linalg.norm(Q0_cluster5_mean))
print( "Mean Q0_cluster6_mean =", Q0_cluster6_mean )
print( "Norm =", np.linalg.norm(Q0_cluster6_mean))

Q1_cluster0 = get_vectors_for_label(0, cluster_for_Q1.labels_, Q1, 40)
Q1_cluster1 = get_vectors_for_label(1, cluster_for_Q1.labels_, Q1, 40)
Q1_cluster2 = get_vectors_for_label(2, cluster_for_Q1.labels_, Q1, 40)
Q1_cluster3 = get_vectors_for_label(3, cluster_for_Q1.labels_, Q1, 40)
Q1_cluster4 = get_vectors_for_label(4, cluster_for_Q1.labels_, Q1, 40)
Q1_cluster5 = get_vectors_for_label(5, cluster_for_Q1.labels_, Q1, 40)
Q1_cluster6 = get_vectors_for_label(6, cluster_for_Q1.labels_, Q1, 40)

Q1_cluster0_idx =  get_vector_of_idx_for_label(0, cluster_for_Q1.labels_, Q1, 40)
Q1_cluster1_idx =  get_vector_of_idx_for_label(1, cluster_for_Q1.labels_, Q1, 40)
Q1_cluster2_idx =  get_vector_of_idx_for_label(2, cluster_for_Q1.labels_, Q1, 40)
Q1_cluster3_idx =  get_vector_of_idx_for_label(3, cluster_for_Q1.labels_, Q1, 40)
Q1_cluster4_idx =  get_vector_of_idx_for_label(4, cluster_for_Q1.labels_, Q1, 40)
Q1_cluster5_idx =  get_vector_of_idx_for_label(5, cluster_for_Q1.labels_, Q1, 40)
Q1_cluster6_idx =  get_vector_of_idx_for_label(6, cluster_for_Q1.labels_, Q1, 40)

Q1_cluster0_mean = np.mean(Q1_cluster0, axis=0)
Q1_cluster1_mean = np.mean(Q1_cluster1, axis=0)
Q1_cluster2_mean = np.mean(Q1_cluster2, axis=0)
Q1_cluster3_mean = np.mean(Q1_cluster3, axis=0)
Q1_cluster4_mean = np.mean(Q1_cluster4, axis=0)
Q1_cluster5_mean = np.mean(Q1_cluster5, axis=0)
Q1_cluster6_mean = np.mean(Q1_cluster6, axis=0)

n_Q1_cluster0 = Q1_cluster0.shape[0]
n_Q1_cluster1 = Q1_cluster1.shape[0]
n_Q1_cluster2 = Q1_cluster2.shape[0]
n_Q1_cluster3 = Q1_cluster3.shape[0]
n_Q1_cluster4 = Q1_cluster4.shape[0]
n_Q1_cluster5 = Q1_cluster5.shape[0]
n_Q1_cluster6 = Q1_cluster6.shape[0]


print( "Mean Q1_cluster0_mean =", Q1_cluster0_mean )
print( "Norm =", np.linalg.norm(Q1_cluster0_mean))
print( "Mean Q1_cluster1_mean =", Q1_cluster1_mean )
print( "Norm =", np.linalg.norm(Q1_cluster1_mean))
print( "Mean Q1_cluster2_mean =", Q1_cluster2_mean )
print( "Norm =", np.linalg.norm(Q1_cluster2_mean))
print( "Mean Q1_cluster3_mean =", Q1_cluster3_mean )
print( "Norm =", np.linalg.norm(Q1_cluster3_mean))
print( "Mean Q1_cluster4_mean =", Q1_cluster4_mean )
print( "Norm =", np.linalg.norm(Q1_cluster4_mean))
print( "Mean Q1_cluster5_mean =", Q1_cluster5_mean )
print( "Norm =", np.linalg.norm(Q1_cluster5_mean))
print( "Mean Q1_cluster6_mean =", Q1_cluster6_mean )
print( "Norm =", np.linalg.norm(Q1_cluster6_mean))

#
#
#
# plt.figure()
# fig, ax = plt.subplots()
# ax.scatter(Q0_r[:, 0], 
#             Q0_r[:, 1], 
#             s=14, marker="o", color="blue")
# ax.scatter(cluster_for_Q0_r.cluster_centers_  [:, 0], 
#             cluster_for_Q0_r.cluster_centers_  [:, 1], 
#             s=50, marker="o", alpha=0.5, color="blue")
# # ax.scatter(Q0_r[Q0_cluster2_idx, 0], 
# #            Q0_r[Q0_cluster2_idx, 1], 
# #            s=14, marker="o", color="blue")
# # ax.scatter(cluster_for_Q0_r.cluster_centers_  [:, 0], 
# #           cluster_for_Q0_r.cluster_centers_  [:, 1], 
# #           s=50, marker="o", alpha=0.5, color="blue")
# #for idx in range(0,40):
# #    ax.annotate( str(idx), (Q0_r[idx, 0], Q0_r[idx, 1]) )
# ax.scatter(Q1_r[:, 0], 
#            Q1_r[:, 1], 
#            s=10, marker="^", color="red")
# ax.scatter(cluster_for_Q1_r.cluster_centers_  [:, 0], 
#             cluster_for_Q1_r.cluster_centers_  [:, 1], 
#             s=50, marker="^", alpha=0.5, color="red")
# #for idx in range(0,40):
# #    ax.annotate( str(idx), (Q1_r[idx, 0], Q1_r[idx, 1]) )
# plt.title("Principal component analysis for data encoded as quantum states ")
# plt.xlabel("Values of first feature")
# plt.ylabel("Values of second feature")
# #plt.savefig("pca-figure.png")
# plt.savefig("pca-figure.eps")
# fig.show()

       

circuit_type = 0
layers=2
qubits = [0, 1, 2]

if circuit_type == 0:
    params = [1] * (9*layers) 

if circuit_type == 1:
        params = [1] * (6*layers) 

if circuit_type == 2:
        params = [1] * (6*layers) # for type 2
 
if circuit_type == 3:
        params = [1] * (9*layers) # for type 3

if circuit_type == 4:
        params = [1] * ((3 + (6*layers) + 3)) # for type 4

if circuit_type == 5:
        params = [1] * ((3 + (3*layers) + 3)) # for type 5
#    case _:
#        params = np.random.rand(18)

np.random.seed(999999)
#np.random.seed(111111)

target_distr = {0: 0,
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 0,
                6: 0,
                7: 0}

output_distribution = {0: 0,
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 0,
                6: 0,
                7: 0}
target_params = params

backend = Aer.get_backend("aer_simulator")

# start = time.time()
# train_data_and_save_angles_to_file()
# elapsed = time.time() - start

print("")

print("n_Q0_cluster0=", n_Q0_cluster0)
print("n_Q0_cluster1=", n_Q0_cluster1)
print("n_Q0_cluster2=", n_Q0_cluster2)
print("n_Q0_cluster3=", n_Q0_cluster3)
print("n_Q0_cluster4=", n_Q0_cluster4)
print("n_Q0_cluster5=", n_Q0_cluster5)
print("n_Q0_cluster6=", n_Q0_cluster6)

print("")

print("n_Q1_cluster0=", n_Q1_cluster0)
print("n_Q1_cluster1=", n_Q1_cluster1)
print("n_Q1_cluster2=", n_Q1_cluster2)
print("n_Q1_cluster3=", n_Q1_cluster3)
print("n_Q1_cluster4=", n_Q1_cluster4)
print("n_Q1_cluster5=", n_Q1_cluster5)
print("n_Q1_cluster6=", n_Q1_cluster6)


print("")
    
print("-------- CLASS 0 -------- ")
[ngoodprobe00, nfalse00]=test_data_tuned(0, 0, n_Q0_cluster0, 1) ; print("")
[ngoodprobe01, nfalse01]=test_data_tuned(0, 1, n_Q0_cluster1, 1) ; print("")
[ngoodprobe02, nfalse02]=test_data_tuned(0, 2, n_Q0_cluster2, 1) ; print("")
[ngoodprobe03, nfalse03]=test_data_tuned(0, 3, n_Q0_cluster3, 1) ; print("")
[ngoodprobe04, nfalse04]=test_data_tuned(0, 4, n_Q0_cluster4, 1) ; print("")
[ngoodprobe05, nfalse05]=test_data_tuned(0, 5, n_Q0_cluster5, 1) ; print("")
[ngoodprobe06, nfalse06]=test_data_tuned(0, 6, n_Q0_cluster6, 1) ; print("")

# print("-------- CLASS 1 -------- ")
[ngoodprobe10, nfalse10]=test_data_tuned(1, 0, n_Q1_cluster0, 1) ; print("")
[ngoodprobe11, nfalse11]=test_data_tuned(1, 1, n_Q1_cluster1, 1) ; print("")
[ngoodprobe12, nfalse12]=test_data_tuned(1, 2, n_Q1_cluster2, 1) ; print("")
[ngoodprobe13, nfalse13]=test_data_tuned(1, 3, n_Q1_cluster3, 1) ; print("")
[ngoodprobe14, nfalse14]=test_data_tuned(1, 4, n_Q1_cluster4, 1) ; print("")
[ngoodprobe15, nfalse15]=test_data_tuned(1, 5, n_Q1_cluster5, 1) ; print("")
[ngoodprobe16, nfalse16]=test_data_tuned(1, 6, n_Q1_cluster6, 1) ; print("")

# print("-------- CLASS 0 -------- ")
# [ngoodprobe00, nfalse00]=test_data(0, 0, n_Q0_cluster0, 1) ; print("")
# [ngoodprobe01, nfalse01]=test_data(0, 1, n_Q0_cluster1, 1) ; print("")
# [ngoodprobe02, nfalse02]=test_data(0, 2, n_Q0_cluster2, 1) ; print("")
# [ngoodprobe03, nfalse03]=test_data(0, 3, n_Q0_cluster3, 1) ; print("")
# [ngoodprobe04, nfalse04]=test_data(0, 4, n_Q0_cluster4, 1) ; print("")
# [ngoodprobe05, nfalse05]=test_data(0, 5, n_Q0_cluster5, 1) ; print("")
# [ngoodprobe06, nfalse06]=test_data(0, 6, n_Q0_cluster6, 1) ; print("")
    

# print("-------- CLASS 1 -------- ")
# [ngoodprobe10, nfalse10]=test_data(1, 0, n_Q1_cluster0, 1) ; print("")
# [ngoodprobe11, nfalse11]=test_data(1, 1, n_Q1_cluster1, 1) ; print("")
# [ngoodprobe12, nfalse12]=test_data(1, 2, n_Q1_cluster2, 1) ; print("")
# [ngoodprobe13, nfalse13]=test_data(1, 3, n_Q1_cluster3, 1) ; print("")
# [ngoodprobe14, nfalse14]=test_data(1, 4, n_Q1_cluster4, 1) ; print("")
# [ngoodprobe15, nfalse15]=test_data(1, 5, n_Q1_cluster5, 1) ; print("")
# [ngoodprobe16, nfalse16]=test_data(1, 6, n_Q1_cluster6, 1) ; print("")


ngoodprobe0 = ngoodprobe00+ngoodprobe01+ngoodprobe02+ngoodprobe03+ngoodprobe04+ngoodprobe05+ngoodprobe06
ngoodprobe1 = ngoodprobe10+ngoodprobe11+ngoodprobe12+ngoodprobe13+ngoodprobe14+ngoodprobe15+ngoodprobe16

nfalse0=nfalse00+nfalse01+nfalse02+nfalse03+nfalse04+nfalse05+nfalse06
nfalse1=nfalse10+nfalse11+nfalse12+nfalse13+nfalse14+nfalse15+nfalse16

print("good for 0 class",(ngoodprobe0/40)*100,"%")
print("good for 1 class",(ngoodprobe1/40)*100,"%")
print("good for both class",((ngoodprobe0+ngoodprobe1)/80)*100,"%")

print("false for 0 class",(nfalse0/40)*100,"%")
print("false for 1 class",(nfalse1/40)*100,"%")
print("false for both class",((nfalse0+nfalse1)/80)*100,"%")

#print(f"Training time: {round(elapsed)} seconds")
