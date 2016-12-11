# neural_network_bp
"""
Created on Thu Oct 13 2016 by Linchang.

Train nuerual network using the Backpropagation (BP) algorithm.

Assumptions: 
1. a two-layer network: one input, one hidden and one output layer.
2. activation functio is the sigmoid function: S(t) = 1/(1+exp(-ts)).
3. hidden nodes: 4.
"""

import random 
# import math
import numpy as np

# Parameters:
# pd_: prefix for partial derivative
# d_: prefix for derivative
# w_ho: weight from hidden layer to output layer
# w_ih: weight from input layer to hidden layer
# g: activation function 

# i: input
# y: total net input on hiddent layer: y = sum(w_ih * i)
# h: output at hidden layer: h = g(y), which is also input at output layer
# z: total net input on output layer: z = sum(w_ho * h)
# o: output at output layer: o = g(z)

# E = 1/2 * sum(target_i - o_i)^2
# dE/dw_ho = dE/do * do/dz * dz/dw_ho
# dE/dw_ih = dE/dh * dh/dy * dy/dw_hi
# dE/dh = sum(dE_i/dh)
# dE_i/dh = dE_i/z * dz/dh

class NeuralNetworkBP:
    '''Neural network to be trained using the BP algorithm.
    
    Weights of both hidden and ouput layers get updated after each training iteration.
    
    Attributes:
        num_inputs: number of neurons on the input layer.
        hidden_layer: neurons of the hidden layer.
        output_layer: neurons of the output layer.
    '''
    
    LEARNING_RATE = 0.5
    def __init__(self, num_inputs, num_hidden, num_outputs, 
                 hidden_layer_weights = None, hidden_layer_bias = None, 
                 output_layer_weights = None, output_layer_bias = None):
        self.num_inputs = num_inputs # number of neurons on the input layer.
        
        # initial hidden and output layer neurons
        self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_bias)
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias)

        # append corresponding weight to each neuron on the hidden and output layer
        # if weight not available for a certain neuron, generate a random # b/w 0 and 1
        self.init_weights_from_inputs_to_hidden_layer(hidden_layer_weights)
        self.init_weights_from_hidden_layer_to_output_layer(output_layer_weights)

    def init_weights_from_inputs_to_hidden_layer(self, hidden_layer_weights):
        weight_num = 0
        for h in range(len(self.hidden_layer.neurons)):
            for i in range(self.num_inputs):
                if not hidden_layer_weights:
                    self.hidden_layer.neurons[h].weights.append(random.random())
                else:
                    self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
                    # sequentially append corresponding weight to each neuron on hidden layer
                weight_num += 1

    def init_weights_from_hidden_layer_to_output_layer(self, output_layer_weights):
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer.neurons)):
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
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')
 
    def feed_forward(self, inputs):
        '''feed forward the NN, from input layer to hidden layer, and to output layer.
        
        Args: 
            inputs: inputs from input layer.
            
        Returns:
            outputs from the outputlayer.
        '''
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_layer_outputs)
 
    def train(self, training_inputs, training_outputs):
        '''train the NN using the BP algorithm.
        
        Args:
        training_inputs: initial training input.
        training_outputs: training target.
        '''
        self.feed_forward(training_inputs)
 
        # 1. output layer neuron derivatives (error wrt total net input on output layer, dE/dz)
        pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):
            # dE/dz for each output layer neuron
            pd_errors_wrt_output_neuron_total_net_input[o] = (self.output_layer.
            neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o]))
 
        # 2. hiddent layer neuron derivatives (error wrt total net input on hidden layer, dE/dy)
        # dE/dy = dE/dh* dh/dy
        pd_errors_wrt_hidden_neuron_total_net_input = [0] * len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):
            # dE/dh = sum(dE/dz * dz/dh) = sum(dE/dz * w_ho_i)
            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_output += (pd_errors_wrt_output_neuron_total_net_input[o]*
                self.output_layer.neurons[o].weights[h])

            # dE/dy = dE/dh * dh/dy
            pd_errors_wrt_hidden_neuron_total_net_input[h] = (d_error_wrt_hidden_neuron_output*
            self.hidden_layer.neurons[h].calculate_pd_output_wrt_total_net_input())
 
        # 3. update output layer weights
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):
                # dE/dw_ho_i = dE/dz * dz/dw_ho_i
                pd_error_wrt_weight = (pd_errors_wrt_output_neuron_total_net_input[o]*
                self.output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_ho))
                
                # delta(w_ho) = alpha * dE/dw_ho_i
                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight
 
        # 4. update hidden layer weights
        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):
                # dE/dw_ih_i = dE/dy * dy/dw_ih_i
                pd_error_wrt_weight = (pd_errors_wrt_hidden_neuron_total_net_input[h]*
                self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_weight(w_ih))

                # delta(w_ih) = alpha * dE/dw_ih_i
                self.hidden_layer.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight

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
    '''Define a neuron, calcualte error and chained derivatives when applicable.
    
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
        # return 1 / (1 + math.exp(-total_net_input))  # sigmoid: [0, 1]
        # return (1 - np.exp(-2*total_net_input))/(1 + np.exp(-2*total_net_input))  # tanh [-1, 1]
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
        return 0.5 * (target_output - self.output) ** 2
    
    # calculate below derivatives using the delta rule (chain rule)
    
    # calculate derivative of error wrt output (dE/do)
    def calculate_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)
    
    # calculate derivative of output wrt total net input (do/dz)
    def calculate_pd_output_wrt_total_net_input(self):
        return self.output * (1 - self.output) 
    
    # calculate derivative of total net income wrt weights (dz/dw_ho)
    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]

    # calculate derivative of error wrt total net input (dE/dz = dE/do * do/dz)
    def calculate_pd_error_wrt_total_net_input(self, target_output):
        return (self.calculate_pd_error_wrt_output(target_output)*
                self.calculate_pd_output_wrt_total_net_input())
                