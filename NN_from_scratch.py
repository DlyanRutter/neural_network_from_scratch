#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 14:56:50 2020

@author: dylanrutter
"""

from math import exp
from random import seed, random
 
# Initialize a neural network
def network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network
 
# Calculate weights * feature inputs
def multiply(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation
 
# Apply sigmoid activation
def sigmoid(activation):
	return 1.0 / (1.0 + exp(-activation))
 
# Forward propagate 
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			matrix = multiply(neuron['weights'], inputs)
			neuron['output'] = sigmoid(matrix)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs
 
# Calculate the derivative of neuron output
def derivative(output):
	return output * (1.0 - output)
 
# Backpropagate error 
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * derivative(neuron['output'])
 
# Update weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']
 
# Train for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
 
# Test training backprop algorithm
seed(1)
features = [[3.1, 2.5,0],
	[1.3,2.,0],
	[3.39,4.4,0],
	[1.38,1.85,0],
	[3.06,3.00,0],
	[7.62,2.7,1],
	[5.33,2.08,1],
	[6.9,1.77,1],
	[8.61,-0.24,1],
	[7.6,3.1,1]]

n_inputs = len(features[0]) - 1
n_outputs = len(set([row[-1] for row in features]))
network = network(n_inputs, 2, n_outputs)
train_network(network, features, 0.5, 20, n_outputs)
for layer in network:
	print(layer)


