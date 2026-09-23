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
   
def convert_data_to_vector_states_double_norm(inputDF, cols=0):
    """
        Create quantum states - input data comes from a Pandas Data Frame, 
        each variable is normalized to avoid domination of some variables 
        (approach known from the classical machine learning), finally, 
        each observation is normalized (to generate correct quantum state). 
        If the first normalization results with an observation with all zero 
        values, then this observation will be deteled.

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
            Numpy array of normalized quantum states.

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
    intervals=np.zeros(shape=(b))
    for i in range(b):
        intervals[i]=maxs[i]-mins[i]
    KNtab=np.ndarray(shape=(a,b))
    for j in range(b):
        for i in range(a):
            KNtab[i,j]=(Ktab[i,j]-mins[j])/intervals[j]
    #detection of an observation with all zero values
    h=0
    i=0
    aa=a
    while (h==0 and i<a):
        sum_all=0
        for j in range(b):
            sum_all+=KNtab[i,j]
        if sum_all==0:
            h=1
            aa-=1
        i+=1
    #quantum normalization
    if math.ceil(math.log2(b)) != math.floor(math.log2(b)):
        c=math.ceil(math.log2(b))
        c=2**c
        Qtab=np.zeros(shape=(aa,c))
    else:
        Qtab=np.ndarray(shape=(aa,b))
    k=0
    for i in range(a):
        sum_all=0
        for j in range(b):
            sum_all+=KNtab[i,j]
        if sum_all==0:
            print('Warning: an observation with all zero values occured and it will be deleted!')
        else:
            for j in range(b):
                Qtab[k,j]=sympy.sqrt(KNtab[i,j]/sum_all)
            k+=1
    return Qtab

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
            Numpy array of normalized quantum states.

        Examples
        --------
        From file 'name.xlsx', four columns were fetched to produce 2-qubit states.
        >>> df = pd.read_excel(r'name.xlsx')
        >>> print(convert_data_to_vector_states(df,4))
        [[0.32438643 0.94034392 0.00251242 0.10256923]
         [0.38518862 0.91692333 0.00243558 0.10428524]
         [0.39649659 0.91048255 0.00235002 0.11750089]
         [0.40813284 0.91291981 0.00223538 0.        ]...
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
    Ktab=pd.DataFrame(inputDF).to_numpy()
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
            sum_all+=Ktab[i,j]
        for j in range(b):
            Qtab[i,j]=sympy.sqrt(Ktab[i,j]/sum_all)
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
        Qvec[i]=sympy.sqrt(Qvec[i]/sum_all)
    return Qvec

def create_learning_and_test_set(inputDF, var_no, learn_set_size=0.8, norm_type=0):
    """
        Create learning and test set of quantum states - input data from 
        Pandas Data Frame

        Parameters
        ----------
        inputDF : pandas.DataFrame
            File of input data.
        var_no : interger
            The number of variables to fetch from file. The last column 
            (variable) is always treated as a target variable.
        learn_set_size : float in range (0,1)
            The percent of observations to include in the learning set (the rest
            forms the test set). Default partition is 80% observations in 
            the learning set and 20% in the test set.
        norm_type : Boolean
            If single normalization is to be utilized then norm_type==0 (realized 
            by the function convert_data_to_vector_states). If double normalization 
            has to be performed norm_type==1 (realized by the function 
            convert_data_to_vector_states_double_norm).

        Returns
        -------
        Tuple
            Tuple of two numpy arrays containing learning and test set.

        Examples
        --------
        From file 'name.xlsx', three columns were fetched to produce 2-qubit states.
        >>> df = pd.read_excel(r'name.xlsx')
        >>> tp=create_learning_and_test_set(df,3)
        The number of variables is: 3
        The number of classes is: 2
        >>> print(tp[0])
        [[0.51199211 0.46564845 0.72182796 0.         0.        ]
         [0.50431663 0.47174501 0.72327131 0.         0.        ]
         [0.50552503 0.46547467 0.72648316 0.         0.        ]
         [0.50529115 0.45837363 0.731146   0.         0.        ]
         [0.50716649 0.45822893 0.72993726 0.         0.        ]...
        >>> print(tp[1])
        [[0.50597801 0.47623957 0.71915376 0.         0.        ]
         [0.4979296  0.47759729 0.72385561 0.         0.        ]
         [0.49186938 0.47857315 0.72734604 0.         0.        ]
         [0.50082988 0.47003216 0.72680066 0.         0.        ]
         [0.49958661 0.47237749 0.72613547 0.         0.        ]...
    """
    if learn_set_size <=0 or learn_set_size>=1:
        raise ValueError("Incorrect proportion between learning and testing set!")
        return None
    else:
        if norm_type==0:
            set_big=convert_data_to_vector_states(inputDF, var_no)
        elif norm_type==1:
            set_big=convert_data_to_vector_states_double_norm(inputDF, var_no)
        else:
            raise ValueError("Incorrect value of parameter norm_type!")
            return None
        target=pd.DataFrame(inputDF.iloc[: , -1]).to_numpy()
        Tab_class, cl_counts = np.unique(target, return_counts=True)
        class_no=len(Tab_class)
        print('The number of classes is:',class_no)
        set_big = np.append(set_big, target, axis=1)
        Tab_data = np.ndarray(shape=(2,class_no))
        for i in range(class_no):
            Tab_data[0,i]=round(cl_counts[i]*learn_set_size)
            Tab_data[1,i]=cl_counts[i]-Tab_data[0,i]
        a,b=set_big.shape
        lset = np.ndarray(shape=(1,b))
        tset = np.ndarray(shape=(1,b))  #int(y)
        counter=np.zeros(shape=class_no)
        for j in range(a): 
            cl=set_big[j,b-1]
            z=0
            i=0
            while z==0:
                if Tab_class[i]==cl:
                    z=1
                else:
                    i+=1
            if counter[i]<Tab_data[0,i]:
                lset=np.append(lset, [set_big[j,:]], axis=0)
                counter[i]+=1
            else:
                tset=np.append(tset, [set_big[j,:]], axis=0)
        lset = np.delete(lset, 0, 0)
        tset = np.delete(tset, 0, 0)
        return lset,tset




#
# TO DESC
#
def version():
    pass

