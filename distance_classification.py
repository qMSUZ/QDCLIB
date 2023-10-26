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

import qdclib as qdcl
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from scipy import stats

from sklearn import decomposition

banana_dataset = np.loadtxt('data/banana_data.txt')

banana_dataset_CM1 = banana_dataset[banana_dataset[:,2]==-1][:,0:2]
banana_dataset_CP1 = banana_dataset[banana_dataset[:,2]== 1][:,0:2]

banana_dataset_CM1_q = banana_dataset_CM1
banana_dataset_CP1_q = banana_dataset_CP1

idx=0
for r in banana_dataset_CM1:
    banana_dataset_CM1_q[idx] = r / np.linalg.norm( r ) 
    idx=idx+1

idx=0
for r in banana_dataset_CP1:
    banana_dataset_CP1_q[idx] = r / np.linalg.norm( r ) 
    idx=idx+1
    
dm_for_CM1 = qdcl.create_quantum_centroid( banana_dataset_CM1_q )
dm_for_CP1 = qdcl.create_quantum_centroid( banana_dataset_CP1_q )


cnt_cm1=0
cnt_cp1=0
print("Banana for class Minus One")
for r in banana_dataset_CM1:
    dm_t1 = qdcl.vector_state_to_density_matrix( r )
    d1 = qdcl.trace_distance_density_matrix(dm_for_CM1, dm_t1)
    d2 = qdcl.trace_distance_density_matrix(dm_for_CP1, dm_t1)
    if d1 < d2:
        cnt_cm1 += 1 
    else:
        cnt_cp1 += 1
        
print("\ttrue for class cm1:", cnt_cm1)
print("\tfalse for class cp1:", cnt_cp1)
print("\ttrue positive", (cnt_cm1)/(cnt_cm1+cnt_cp1) * 100,'%')


cnt_cm1=0
cnt_cp1=0
print("Banana for class Plus One")
for r in banana_dataset_CP1:
    dm_t1 = qdcl.vector_state_to_density_matrix( r )
    d1 = qdcl.trace_distance_density_matrix(dm_for_CP1, dm_t1)
    d2 = qdcl.trace_distance_density_matrix(dm_for_CM1, dm_t1)
    if d1 < d2:
        cnt_cp1 += 1 
    else:
        cnt_cm1 += 1
        
print("\ttrue for class cp1:", cnt_cm1)
print("\tfalse for class cm1:", cnt_cp1)
print("\ttrue positive", (cnt_cp1)/(cnt_cm1+cnt_cp1) * 100,'%')

        