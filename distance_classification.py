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

# _ratio = 0.30
# idx_for_cutoff = int( banana_dataset.shape[0] * _ratio )
    

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

# qdcl.vector_state_to_density_matrix( )
        