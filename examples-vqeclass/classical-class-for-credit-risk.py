#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#/***************************************************************************
# *   Copyright (C) 2020 -- 2024 by Marek Sawerwain                         *
# *                                  <M.Sawerwain@gmail.com>                *
# *                                  <M.Sawerwain@issi.uz.zgora.pl>         *
# *                                                                         *
# *                              by Joanna Wiśniewska                       *
# *                                  <Joanna.Wisniewska@wat.edu.pl>         *
# *                                                                         *
# *                                                                         *
# *   Part of the of QDCL the credit sales classification example --        *
# *      classical approach                                                 *
# *                                                                         *
# *         https://github.com/qMSUZ/QDCLIB                                 *
# *                                                                         *
# *   This program is free software; you can redistribute it and/or modify  *
# *   it under the terms of the GNU General Public License as published by  *
# *   the Free Software Foundation; either version 3 of the License, or     *
# *   (at your option) any later version.                                   *
# *                                                                         *
# *   This program is distributed in the hope that it will be useful,       *
# *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
# *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
# *   GNU General Public License for more details.                          *
# *                                                                         *
# *   You should have received a copy of the GNU General Public License     *
# *   along with this program; if not, write to the                         *
# *   Free Software Foundation, Inc.,                                       *
# *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
# ***************************************************************************/


import numpy as np
import seaborn as sns
import pandas as pd

from sympy import sqrt
#from sympy import I


import matplotlib.pyplot as plt


from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn import decomposition
from sklearn.preprocessing import MinMaxScaler

# we hide RuntimeWarning for cleanout text output
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def read_data():
    # data read-in
    # 7 decision variables - 4 integers (1-4) + 3 Boolean (5-7)
    # 2-valued classification task
    df = pd.read_excel (r'train-data-v0.xlsx')
    
    # classical normalization
    # variables x1..x4
    j=1
    K=np.ndarray(shape=(80,8))
    while(j<5):
        x_pom=df["X"+str(j)]
        min1=x_pom[0]
        max1=x_pom[0]
        for i in range(80):
            if x_pom[i] < min1:
                min1=x_pom[i]
            if x_pom[i] > max1:
                max1=x_pom[i]
        zakres=max1-min1
        #normalized data will be saved in "K" numpy array
        for i in range(80):
            K[i,(j-1)]=(x_pom[i]-min1)/zakres
        j+=1
    #altering array "K" with columns x5..x8 (Boolean variables)
    x5_pom=df["X5"]
    x6_pom=df["X6"]
    x7_pom=df["X7"]
    x8_pom=df["X8"]
    for i in range(80):
        K[i,4]=x5_pom[i]
        K[i,5]=x6_pom[i]
        K[i,6]=x7_pom[i]
        K[i,7]=x8_pom[i]
    
    #quantum normalizetion - data saved in "Q" numpy array
    Q=np.ndarray(shape=(80,8))
    Q0=np.ndarray(shape=(1,8))
    Q1=np.ndarray(shape=(1,8))
    for i in range(80):
        sum_all=0
        for j in range(8):
            sum_all+=K[i,j]
        for j in range(8):
            Q[i,j]=sqrt(K[i,j]/sum_all)
    for i in range(80):
        suma=0
        for j in range(8):
            suma+=Q[i,j]*Q[i,j]
    for i in range(80):  
        if df.Y[i] == 0:
            Q0=np.vstack((Q0, Q[i]));
        if df.Y[i] == 1:
            Q1=np.vstack((Q1, Q[i]));
    
    Q0 = np.delete(Q0, (0), axis=0)
    Q1 = np.delete(Q1, (0), axis=0)
     
    return df, Q, Q0, Q1


def perform_SVC(t_kernel):
    C = 10
    #n_features = 8
    classifier = SVC(kernel=t_kernel, C=C, probability=True, random_state=0)
    classifier.fit(Q, df.Y)
    y_pred = classifier.predict(Q)
    accuracy = accuracy_score(df.Y, y_pred)
    print("SVC kernel=", t_kernel)
    print("Accuracy (train) for %s: %0.1f%% " % ("SVC", accuracy * 100))

def perform_KNeighborsClassifier(n):
    classifier = KNeighborsClassifier(n)
    classifier.fit(Q, df.Y)
    y_pred = classifier.predict(Q)
    accuracy = accuracy_score(df.Y, y_pred)
    print("KNeighborsClassifier   n Neighbors=", n)
    print("Accuracy (train) for %s: %0.1f%% " % ("KNeighborsClassifier", accuracy * 100))


def perform_GaussianProcessClassifier():
    classifier = GaussianProcessClassifier(1.0 * RBF(1.0))
    classifier.fit(Q, df.Y)
    y_pred = classifier.predict(Q)
    accuracy = accuracy_score(df.Y, y_pred)
    print("GaussianProcessClassifier")
    print("Accuracy (train) for %s: %0.1f%% " % ("GaussianProcessClassifier", accuracy * 100))

def perform_DecisionTreeClassifier(n_max_depth):
    classifier = DecisionTreeClassifier(max_depth=n_max_depth)
    classifier.fit(Q, df.Y)
    y_pred = classifier.predict(Q)
    accuracy = accuracy_score(df.Y, y_pred)
    print("DecisionTreeClassifier")
    print("Accuracy (train) for %s: %0.1f%% " % ("DecisionTreeClassifier", accuracy * 100))

def perform_RandomForestClassifier(n_max_depth, n_num_estimators, n_max_features):
    classifier = RandomForestClassifier(max_depth=n_max_depth, n_estimators=n_num_estimators, max_features=n_max_features)
    classifier.fit(Q, df.Y)
    y_pred = classifier.predict(Q)
    accuracy = accuracy_score(df.Y, y_pred)
    print("RandomForestClassifier")
    print("Accuracy (train) for %s: %0.1f%% " % ("RandomForestClassifier", accuracy * 100))

def perform_MLPClassifier(n_alpha, n_max_iter):
    classifier = MLPClassifier(alpha=n_alpha, max_iter=n_max_iter)
    classifier.fit(Q, df.Y)
    y_pred = classifier.predict(Q)
    accuracy = accuracy_score(df.Y, y_pred)
    print("MLPClassifier")
    print("Accuracy (train) for %s: %0.1f%% " % ("MLPClassifier", accuracy * 100))

def perform_AdaBoostClassifier():
    classifier = AdaBoostClassifier()
    classifier.fit(Q, df.Y)
    y_pred = classifier.predict(Q)
    accuracy = accuracy_score(df.Y, y_pred)
    print("AdaBoostClassifier")
    print("Accuracy (train) for %s: %0.1f%% " % ("AdaBoostClassifier", accuracy * 100))

def perform_GaussianNB():
    classifier = GaussianNB()
    classifier.fit(Q, df.Y)
    classifier.fit(Q, df.Y)
    y_pred = classifier.predict(Q)
    accuracy = accuracy_score(df.Y, y_pred)
    print("GaussianNB")
    print("Accuracy (train) for %s: %0.1f%% " % ("GaussianNB", accuracy * 100))

def perform_QuadraticDiscriminantAnalysis():
    classifier = QuadraticDiscriminantAnalysis()
    classifier.fit(Q, df.Y)
    y_pred = classifier.predict(Q)
    accuracy = accuracy_score(df.Y, y_pred)
    print("QuadraticDiscriminantAnalysis")
    print("Accuracy (train) for %s: %0.1f%% " % ("QuadraticDiscriminantAnalysis", accuracy * 100))


def pca_figure():
    pca = decomposition.PCA(n_components=2)
    Q0_r = pca.fit(Q0).transform(Q0)
    Q1_r = pca.fit(Q1).transform(Q1)
    
    plt.figure()
    plt.scatter(Q0_r[:, 0], Q0_r[:, 1], color="blue")
    plt.scatter(Q1_r[:, 0], Q1_r[:, 1], color="red")
    plt.title("PCA")
    plt.savefig("pca-figure.png")
    plt.savefig("pca-figure.eps") 


def data_analysis_figure():
    features = Q
    features = MinMaxScaler().fit_transform(features)
    
    df2 = pd.DataFrame(Q, columns=df.columns[0:8])
    df2["class"] = pd.Series(df.Y)
    
    plt2 = sns.pairplot(df2, hue="class", palette="tab10")
    plt2.savefig("data-analysis.png")
    plt2.savefig("data-analysis.eps") 


print("read data")
df, Q, Q0, Q1 = read_data()

# Pearson Correlation Coefficient  (PCC)
pcc_rho = np.corrcoef( Q[:,0:7].transpose() )

print("Pearson Correlation Coefficients")
print(pcc_rho)

pca_figure()
data_analysis_figure()


perform_SVC("sigmoid")
print("")
perform_SVC("linear")
print("")
perform_SVC("rbf")
print("")
perform_SVC("poly")
print("")
perform_KNeighborsClassifier(3)
print("")
perform_GaussianProcessClassifier()
print("")
perform_DecisionTreeClassifier(5)
print("")
perform_RandomForestClassifier(5, 10, 8)
print("")
perform_MLPClassifier(1,1000)
print("")
perform_AdaBoostClassifier()
print("")
perform_GaussianNB()
print("")
perform_QuadraticDiscriminantAnalysis()
print("")
