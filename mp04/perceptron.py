# perceptron.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020


import numpy as np



def trainPerceptron(train_set, train_labels, max_iter):
    #Write code for Mp4
    learning_rate = 1;
    weights = np.zeros(len(train_set[0])+1)
    for e in range(max_iter):
        for features, label in zip(train_set, train_labels):
            prediction = 1 if (np.dot(features, weights[1:]) + weights[0]) > 0 else 0
            weights[1:] += learning_rate * (label - prediction) * features
            weights[0] += learning_rate * (label - prediction) * 1
    W = weights[1:]
    b = weights[0]
    return W, b

def classifyPerceptron(train_set, train_labels, dev_set, max_iter):
    #Write code for Mp4
    trained_weight, trained_bias = trainPerceptron(train_set, train_labels, max_iter)
    dev_label = []
    for img in dev_set:
        pred_res = 1 if (np.dot(img, trained_weight) + trained_bias) > 0 else 0
        dev_label.append(pred_res)
    return dev_label



