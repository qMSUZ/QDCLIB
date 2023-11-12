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
import qdclib as qdcl

crabs_dataset = None
original_data  = None
original_class_labels = None


nrm_and_std_d = None
nrm_d = None

org_d0 = None
org_d1 = None

d0 = None
d1 = None

nrm_d0 = None
nrm_d1 = None

def _read_crabs_data():
    df = pd.read_excel(r'datasets/crabs_dataset.xlsx')
    col_no=6
    Q=qdcl.convert_data_to_vector_states_double_norm(df, col_no)
    QPrime=qdcl.convert_data_to_vector_states(df, col_no)
    _r,_c=QPrime.shape
    Kraw=np.zeros(shape=(_r,_c))
    for _i in range(_r):
        for _j in range(col_no):
            Kraw[_i,_j]=df.iloc[_i,_j]
    
    Q0=np.ndarray(shape=(100,8))
    Q1=np.ndarray(shape=(100,8))
    orgC0=np.ndarray(shape=(100,8))
    orgC1=np.ndarray(shape=(100,8))
    nrmC0=np.ndarray(shape=(100,8))
    nrmC1=np.ndarray(shape=(100,8))
    
    Y=np.ndarray(shape=(200,1))
    idx0=0
    idx1=0
    for i in range(200):
        if df['sp'][i] == 0:
            Y[i]=0
            Q0[idx0]=Q[i]
            orgC0[idx0] = Kraw[i]
            nrmC0[idx0] = QPrime[i]
            idx0 = idx0 + 1
        if df['sp'][i] == 1:
            Y[i]=1
            Q1[idx1]=Q[i]
            orgC1[idx1] = Kraw[i]
            nrmC1[idx1] = QPrime[i]
            idx1 = idx1 + 1

    return df.values, Q, QPrime, Y, Q0, Q1, orgC0, orgC1, nrmC0, nrmC1

def info():
    pass

def load_data():
    global crabs_dataset
    global original_data
    global original_class_labels
    global nrm_and_std_d
    global nrm_d
    global d0
    global d1
    global org_d0
    global org_d1
    global nrm_d0
    global nrm_d1

    crabs_dataset, nrm_and_std_d, nrm_d, Y, d0, d1, org_d0, org_d1, nrm_d0, nrm_d1 = _read_crabs_data()
    
    original_data = crabs_dataset
    original_class_labels = Y

def get_original_data():
    return crabs_dataset

def get_original_class_labels():
    return original_class_labels

def get_normalized_data():
    return nrm_d

def get_normalized_and_standarized_data():
    return nrm_and_std_d

def get_original_data_for_class( _idx ):
    
    if _idx==0:
        return org_d0
    
    if _idx==1:
        return org_d1
    
    return None

def get_normalized_data_for_class( _idx ):
    
    if _idx==0:
        return nrm_d0
    
    if _idx==1:
        return nrm_d1
    
    return None

def get_normalized_and_standarized_data_for_class( _idx ):
        if _idx==0:
            return d0
        
        if _idx==1:
            return d1

def get_quantum_data_for_class( _idx, _variant=0 ):
    
    if _variant==0: # only normalization
        if _idx==0:
            return nrm_d0
        
        if _idx==1:
            return nrm_d1
    
    if _variant==1: # normalization and standarization
        if _idx==0:
            return d0
        
        if _idx==1:
            return d1
    
    return None