# _neural_network_perceptron.py
'''
Created on Thu Oct 16 2016 by Linchang.

Implement nuerual network - Perceptron learning.

Assumptions: 
1. single-layer network: one input layer and one output layer.
2. activation function is sign.
'''

import random


class NeuralNetworkPN:
    '''Neural network to be trained using Perceptron.
    
    Weights of ouput layers get updated after training iteration if sign of output
    does not match target.
    
    Attributes:
        num_inputs: number of neurons on the input layer.
        output_layer: neurons of the output layer.
    '''
    
    def __init__(self, num_inputs, num_outputs,
                 output_layer_weights = None, output_layer_bias = None):
        self.num_inputs = num_inputs # number of neurons on the input layer.
        
        # initiate output layer neurons
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias)
        self.init_weights_from_input_layer_to_output_layer(output_layer_weights)

    def init_weights_from_input_layer_to_output_layer(self, output_layer_weights):
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(self.num_inputs):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(random.random())
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                    # sequentially append corresponding weight to each neuron on output layer
                weight_num += 1

    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')
 
    def feed_forward(self, inputs):
        '''feed forward the NN, from input layer to output layer.
        
        Args: 
            inputs: inputs from input layer.
            
        Returns:
            outputs from the outputlayer.
        '''
        
        return self.output_layer.feed_forward(inputs)
 
    def train(self, training_inputs, training_outputs):
        '''train the NN using Perceptron.
        
        Args:
        training_inputs: initial training input.
        training_outputs: training target.
        '''
        self.feed_forward(training_inputs)
 
        # update output layer weights
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):
                self.output_layer.neurons[o].weights[w_ho] += (training_outputs[o] - self.output_layer.get_outputs()[o]) * training_inputs[w_ho]
 
    def calculate_total_error(self, training_sets):
        '''Calculate total error by feeding into the training_sets.
        
        Args:
            training_sets: [training_inputs, training_outputs].
        '''
        total_error = 0
        for t in range(len(training_sets)):
             training_inputs, training_outputs = training_sets[t]
             self.feed_forward(training_inputs)
             for o in range(len(training_outputs)):
                 total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
        return float(total_error)/len(training_sets)

class NeuronLayer:
    '''Define a layer of neurons.
    
    Attributes:
        bias: assume nodes on the same layer share one bias b.
        neurons: all neurons (instances of Neuron) of the layer.
    '''
    
    def __init__(self, num_neurons, bias):
        self.bias = bias if bias else random.random()
        self.neurons = []
        
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))
 
    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                 print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)
 
    def feed_forward(self, inputs):
        '''feed forward the neural network from current layer to the next.
        
        Args:
            inputs: input vector on input layer.
        
        Returns:
            outputs: list of output of all neurons.
        '''
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs
 
    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs


class Neuron:
    '''Define a neuron.
    
    Attributes:
        bias: bias (constant input) for the neuron.
        weights: weights of this neuron relative to nodes from previous layer.
    '''
    
    def __init__(self, bias):
        self.bias = bias       
        self.weights = [] # when hte neuron instance is initiated, weight is an 
                          # empty list. It gets assigned when the NN is created.
 
    def calculate_total_net_input(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias
    
    def squash(self, total_net_input):
        return -1 if total_net_input <= 0 else 1  # sign: (-1, 1)

    def calculate_output(self, inputs):
        '''Calculate output at this neuron.
        
        Args:
            inputs: input vector from last layer, needed in calculat_output.
            
        Returns:
            output: output of this neuron (after applying activation function).
        '''
        
        self.inputs = inputs
        self.output = self.squash(self.calculate_total_net_input())
        return self.output
        
    # error for each neuron is calculated as the Squared Error of output against target  
    def calculate_error(self, target_output):
        '''Calculate error based on target output.
        
        Args:        
        target_output: target output for this neuron.
        '''
        return abs(target_output - self.output)




