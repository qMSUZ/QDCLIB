#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#/***************************************************************************
# *   Copyright (C) 2022 -- 2026 by Marek Sawerwain                         *
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

import qdclib as qdcl
import entdetector as ed



# _n_samples = 4

# qAp = ed.create_qubit_bell_state( minus = 0 )
# qAm = ed.create_qubit_bell_state( minus = 1 )

# qB = ed.create_two_qubit_bell_state_non_maximal( 0.5 )
# qC = ed.create_random_2qubit_pure_state()

# # ed.is_entangled_vector_2q_state(qAp)
# # ed.is_entangled_vector_2q_state(qAm)

# # ed.is_entangled_vector_2q_state(qB)
# # ed.is_entangled_vector_2q_state(qC)


# qA = ed.create_qubit_bell_state( minus = 0 )
# qA_dm = qdcl.vector_state_to_density_matrix(qA)
# s_qA = qA_dm.reshape(16,1)

# qB = ed.create_qubit_bell_state( minus = 1 )
# qB_dm = qdcl.vector_state_to_density_matrix(qB)
# s_qB = qB_dm.reshape(16,1)

# np.linalg.norm(s_qA)
# np.linalg.norm(s_qB)

# qdcl.trace_distance_density_matrix(qA_dm, qB_dm)

# probes = np.empty(shape=(0,4), dtype=complex)

# probes = np.append( probes, [ qAp ], axis=0 )
# probes = np.append( probes, [ qAm ], axis=0 )
# probes = np.append( probes, [ qB  ], axis=0 )
# probes = np.append( probes, [ qC  ], axis=0 )

# print("the first set of probes")

# for p in probes:
#     print("entangled: ",ed.is_entangled_vector_2q_state(p) )

# # create _n_samples probes with create_random_2qubit_pure_state
# print("\n")

print("the second set of probes")


_n_samples = 100
probes_ent = 4
probes_not_ent = 0

train_d = np.empty( shape = (0, 4), dtype = complex )

train_labels = np.zeros( shape = (0, ) )


ENT_LBL = -1
NOENT_LBL = 1

# creation four Bell states

# 00 + 11
q=ed.create_qubit_bell_state()
train_d = np.append( train_d, [ q ], axis=0 )
train_labels = np.append( train_labels, [ ENT_LBL ], axis=0 )

# 00 - 11
q=ed.create_qubit_bell_state( minus=1)
train_d = np.append( train_d, [ q ], axis=0 )
train_labels = np.append( train_labels, [ ENT_LBL ], axis=0 )

# 01 + 10
q=ed.create_qubit_bell_state( form=1 )
train_d = np.append( train_d, [ q ], axis=0 )
train_labels = np.append( train_labels, [ ENT_LBL ], axis=0 )

# 01 - 10
q=ed.create_qubit_bell_state( minus=1, form=1)
train_d = np.append( train_d, [ q ], axis=0 )
train_labels = np.append( train_labels, [ ENT_LBL ], axis=0 )


for _ in range( _n_samples-4 ):
    
    q = ed.create_random_2qubit_separable_pure_state()
    
    isEnt = ed.is_entangled_vector_2q_state( q )
    
    if isEnt == True:
        train_labels = np.append( train_labels, [ ENT_LBL ], axis=0 )
        probes_ent = probes_ent + 1
    else:
        train_labels = np.append( train_labels, [  NOENT_LBL ], axis=0 )
        probes_not_ent = probes_not_ent + 1
    
    train_d = np.append( train_d, [ q ], axis=0 )

    
# objsvm=qdcl.QuantumSVM()
# objsvm.set_data(train_d, train_labels)
# objsvm.classic_fit()
# labels_predict = objsvm.classic_predict( train_d )

# print("Classic SVM for quantum states")
# print("Labels: labels_predict - test_labels")
# print(labels_predict - train_labels)

print( "Number of entangled states: ", probes_ent, "non-entangled: ", probes_not_ent )

print( "\n" )

# QuantumSVM
objsvm=qdcl.QuantumSVM()
objsvm.set_data(train_d, train_labels, _is_it_quantum=True)
objsvm.prepare_quantum_objects()
objsvm.quantum_fit()
labels_predict = objsvm.quantum_predict( train_d )

# QuantumSVM but the classical fit and Gaussian kernel will be used
objsvm=qdcl.QuantumSVM()
objsvm.set_data(train_d, train_labels, _is_it_quantum=True)
objsvm.prepare_quantum_objects()
objsvm.quantum_fit()
labels_predict = objsvm.quantum_predict( train_d )



print("Labels: labels_predict - train_labels")
print(labels_predict - train_labels)

print("Test data")

correct=0
ent_correct=0
ent_incorrect=0
noent_correct=0
noent_incorrect=0

for idx in range( _n_samples ):
    
    if  labels_predict[idx] == train_labels[idx]:
        correct = correct + 1
        
    if  labels_predict[idx] == ENT_LBL and  train_labels[idx] == ENT_LBL:
        ent_correct = ent_correct + 1

    if  labels_predict[idx] == NOENT_LBL and  train_labels[idx] == NOENT_LBL:
        noent_correct = noent_correct + 1

    if  labels_predict[idx] == ENT_LBL and  train_labels[idx] == NOENT_LBL:
        ent_incorrect = ent_incorrect + 1

    if  labels_predict[idx] == NOENT_LBL and  train_labels[idx] == ENT_LBL:
        noent_incorrect = noent_incorrect + 1

        
    #print("idx=",idx," Label = ", labels_predict[idx], "org label", train_labels[idx])
    #print("P < 1/2 we classify probe as +1, otherwise −1")
        
print("_n_samples", _n_samples, "correct", correct, "ratio: ", correct/_n_samples)
print("ent_correct", ent_correct, "ent_incorrect", ent_incorrect)
print("noent_correct", noent_correct, "noent_incorrect", noent_incorrect)
print("sum:", ent_correct+ent_incorrect+noent_correct+noent_incorrect  )

