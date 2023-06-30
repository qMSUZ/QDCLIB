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
import qdclib as qdcl


b = qdcl.BlochVisualization()
ptns = np.empty((0,3))
ptns = np.append(ptns, [[ 1, 0, 0]], axis=0) # positive x
ptns = np.append(ptns, [[-1, 0, 0]], axis=0) # negative x
ptns = np.append(ptns, [[ 0, 1, 0]], axis=0) # +y
ptns = np.append(ptns, [[ 0,-1, 0]], axis=0) # -y
ptns = np.append(ptns, [[ 0, 0, 1]], axis=0) # +z 
ptns = np.append(ptns, [[ 0, 0,-1]], axis=0) # -z

b.set_points( ptns )

f=b.make_figure()
f.show()