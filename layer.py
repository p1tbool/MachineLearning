#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:22:09 2018

@author: nico
"""
import numpy as np
import matplotlib.pyplot as plt


    
def sigmoid(x,derivative=False):
    if (derivative):
        return x * (1-x)
    return 1 /(1+ np.exp(-x))

def tanh(x, derivative=False):
    if (derivative):
        return 1.0 - np.tanh(x)**2
    return np.tanh(x)

############################3
class Layer:
    def __init__(self, neurons, inputs, activationFunction):
        self.inputWeight = np.random.random((inputs,neurons))
        self.neurons = neurons
        self.inputs  = inputs
        self.activationFunction = activationFunction
        self.sum = None
        self.res= None
        self.next = None
        self.processedInput = None
        self.delta_layer = None
    
    def feedForward(self,inputs):
        self.processedInput = inputs
        self.sum = np.dot(inputs, self.inputWeight)
        self.res = self.activationFunction(self.sum)
    
    def backFeed(self, deltaError):
        ## need to be launched from the last layer
        deltaSum = self.activationFunction(self.res, True)
        self.delta_layer = np.dot(self.processedInput.T,deltaError*deltaSum)

    def setNextLayer(self, layer):
        self.next = layer
        
    def process(self, inputs): ## groing through each layer
        self.feedForward(inputs)
        
        if self.next == None:
            return self.res
        
        return self.process(self.res)
    
    def computeError(self, expectedResults):
        delta_error = self.res-expectedResults
        return ((1/2)*(np.power((delta_error),2))) #??



inputs = np.array([
    [1,1,1],
    [1,0,1],
    [0,1,1],
    [0,0,1]])


expectedResults = np.array([[0],[1],[1],[0]])

graph_index = 0

        
layers = []

for i in range(0,2):
    layers.append(Layer(3, len(inputs[0]),sigmoid))

res = layers[0].process(inputs)
print(res)