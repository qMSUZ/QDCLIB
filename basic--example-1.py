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

from qdclib import *

v=np.array([1,0])
u=np.array([1,0])
print("A value of the Fidelity measure for the same states:")
print(fidelity_measure(u, v))

v=np.array([1/np.sqrt(2),1/np.sqrt(2)])
u=np.array([1/np.sqrt(2),-1/np.sqrt(2)])
print("A value of the Fidelity measure for the orthogonal states:")
print("without chop: ", fidelity_measure(u, v) ) 
print("   with chop: ", chop(fidelity_measure(u, v)) ) 

v=np.array([1/np.sqrt(2),1/np.sqrt(2)])
u=np.array([1/np.sqrt(2),0 + 1j/np.sqrt(2)])
print("A value of the Fidelity measure for two examplary states:")
print(fidelity_measure(u, v))
