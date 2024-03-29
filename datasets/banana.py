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

import numpy as np
import pandas as pd

banana_dataset = None
original_data  = None

banana_dataset_CM1 = None
banana_dataset_CP1 = None

banana_dataset_CM1_q = None
banana_dataset_CP1_q = None


def info():
    pass

def load_data():
    
    global banana_dataset
    global original_data
    
    global banana_dataset_CM1
    global banana_dataset_CP1
    
    global banana_dataset_CM1_q
    global banana_dataset_CP1_q

    bd_from_pd = pd.read_excel(r'datasets/banana_data.xlsx')
    banana_dataset = bd_from_pd.values
    original_data  = banana_dataset

    banana_dataset_CM1 = banana_dataset[banana_dataset[:,2]==-1][:,0:2]
    banana_dataset_CP1 = banana_dataset[banana_dataset[:,2]== 1][:,0:2]
    
    banana_dataset_CM1_q = banana_dataset_CM1.copy()
    banana_dataset_CP1_q = banana_dataset_CP1.copy()

    idx=0
    for r in banana_dataset_CM1:
        banana_dataset_CM1_q[idx] = r / np.linalg.norm( r ) 
        idx=idx+1
    
    idx=0
    for r in banana_dataset_CP1:
        banana_dataset_CP1_q[idx] = r / np.linalg.norm( r ) 
        idx=idx+1

def get_original_data_for_class( _idx ):
    
    if _idx==0:
        return banana_dataset_CM1
    
    if _idx==1:
        return banana_dataset_CP1
    
    return None

def get_quantum_data_for_class( _idx ):
    
    if _idx==0:
        return banana_dataset_CM1_q
    
    if _idx==1:
        return banana_dataset_CP1_q
    
    return None




