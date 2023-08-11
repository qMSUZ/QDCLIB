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

n_clusters = 4
d = qdcl.create_focused_circle_probes_with_uniform_placed_centers(30, n_clusters, _width_of_cluster=0.15)
labels, centers = qdcl.kmeans_quantum_states( d, n_clusters, _func_distance=qdcl.COSINE_DISTANCE )
#labels, centers = qdcl.kmedoids_quantum_states( d, n_clusters, _func_distance=qdcl.COSINE_DISTANCE )

#labels, centers = qdcl.kmeans_quantum_states( d, n_clusters, _func_distance=qdcl.DOT_DISTANCE )
#labels, centers = qdcl.kmeans_quantum_states( d, n_clusters, _func_distance=qdcl.FIDELITY_DISTANCE )
#labels, centers = qdcl.kmeans_quantum_states( d, n_clusters, _func_distance=qdcl.TRACE_DISTANCE )


f=qdcl.create_circle_plot_with_centers_for_2d_data( d, n_clusters, centers, labels )
f.show()

cm = qdcl.create_covariance_matrix(d)
print("Covariance matrix")
print(cm)

for idx_class in range( 0, qdcl.get_max_label_class( labels )+1 ):
    print("Data for class ", idx_class)
    f=qdcl.create_scatter_plot_for_2d_data( qdcl.get_data_for_class(d,labels, idx_class) )
