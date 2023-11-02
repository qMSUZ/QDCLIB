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

import pandas as pd
import numpy as np

iris_dataset = None
original_data  = None
original_labels = None


d = None
dprime = None
Y = None

org_d0 = None
org_d1 = None
org_d2 = None

d0 = None
d1 = None
d2 = None

nrm_d0 = None
nrm_d1 = None
nrm_d2 = None


def _read_iris_data( fname ):
    
    df = pd.read_csv( fname )
    
    #classical normalization - 4 variables
    j=1
    K=np.ndarray(shape=(150,4))
    Kraw=np.ndarray(shape=(150,4))
    while(j<5):
        x_pom=df["X"+str(j)]
        min1=x_pom[0]
        max1=x_pom[0]
        for i in range(80):
            if x_pom[i] < min1:
                min1=x_pom[i]
            if x_pom[i] > max1:
                max1=x_pom[i]
        interval=max1-min1
        #normalized data saved in a numpy array K
        for i in range(150):
            K[i,(j-1)]=(x_pom[i]-min1)/interval
            Kraw[i,(j-1)]=x_pom[i]
        j+=1
    #print(K)
 
    #quantum normalization - final data saved in a numpy array Q
    Q=np.ndarray(shape=(150,4))
    QPrime=np.ndarray(shape=(150,4))
    Q0=np.ndarray(shape=(50,4))
    Q1=np.ndarray(shape=(50,4))
    Q2=np.ndarray(shape=(50,4))
    orgC0=np.ndarray(shape=(50,4))
    orgC1=np.ndarray(shape=(50,4))
    orgC2=np.ndarray(shape=(50,4))
    nrmC0=np.ndarray(shape=(50,4))
    nrmC1=np.ndarray(shape=(50,4))
    nrmC2=np.ndarray(shape=(50,4))
    
    for i in range(150):
        QPrime[i]=Kraw[i]/np.linalg.norm(Kraw[i])
        sum_all=0
        for j in range(4):
            sum_all+=K[i,j]
        for j in range(4):
            # IRIS data contains only real data
            Q[i,j]=np.sqrt(K[i,j]/sum_all)
    
    
    Y=np.ndarray(shape=(150,1))
    idx0=0
    idx1=0
    idx2=0
    for i in range(150):
        if df['class'][i] == 'Iris-setosa':
            Y[i]=0
            Q0[idx0]=Q[i]
            orgC0[idx0] = Kraw[i]
            nrmC0[idx0] = QPrime[i]
            idx0 = idx0 + 1
        if df['class'][i] == 'Iris-versicolor':
            Y[i]=1
            Q1[idx1]=Q[i]
            orgC1[idx1] = Kraw[i]
            nrmC1[idx1] = QPrime[i]
            idx1 = idx1 + 1
        if df['class'][i] == 'Iris-virginica':
            Y[i]=2
            Q2[idx2]=Q[i]
            orgC2[idx2] = Kraw[i]
            nrmC1[idx2] = QPrime[i]
            idx2 = idx2 + 1
    return df.values, Q, QPrime, Y,  Q0, Q1, Q2, orgC0, orgC1, orgC2, nrmC0, nrmC1, nrmC2


def info():
    pass

def load_data():
    global iris_dataset
    global original_data
    global original_labels
    global d
    global dprime
    global Y
    global d0
    global d1
    global d2
    global org_d0
    global org_d1
    global org_d2
    global nrm_d0
    global nrm_d1
    global nrm_d2


    iris_dataset, d,  dprime, Y, d0, d1, d2, org_d0, org_d1, org_d2, nrm_d0, nrm_d1, nrm_d2 = _read_iris_data( 'datasets/iris_data.txt')
    
    original_data = iris_dataset
    original_labels = Y

def get_original_data():
    return iris_dataset

def get_original_labels():
    return original_labels

def get_original_data_for_class( _idx ):
    
    if _idx==0:
        return org_d0
    
    if _idx==1:
        return org_d1

    if _idx==2:
        return org_d2
    
    return None

def get_normalised_data_for_class( _idx ):
    
    if _idx==0:
        return nrm_d0
    
    if _idx==1:
        return nrm_d1

    if _idx==2:
        return nrm_d2
    
    return None

def get_normalised_and_standarised_data_for_class( _idx ):
        if _idx==0:
            return d0
        
        if _idx==1:
            return d1
    
        if _idx==2:
            return d2

def get_quantum_data_for_class( _idx, _variant=0 ):
    
    if _variant==0: # only normalisation
        if _idx==0:
            return nrm_d0
        
        if _idx==1:
            return nrm_d1
    
        if _idx==2:
            return nrm_d2
    
    if _variant==1: # normalisation and standarisation
        if _idx==0:
            return d0
        
        if _idx==1:
            return d1
    
        if _idx==2:
            return d2
    
    return None

