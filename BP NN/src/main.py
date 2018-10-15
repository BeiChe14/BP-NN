# -*- coding: utf-8 -*-

import numpy as np;
from matplotlib import pyplot as plt;
from scipy import special as s;

# neural network class definitions
class neuralNetwork:
    
    # initialise the neural network
    def __init__(self,input_nodes,hidden_nodes,output_nodes,learning_rate):
        # set number of nodes in each input,hidden,output layer
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        
        # learning rate
        self.lr = learning_rate
        
        self.weight_matrix()
        
        # activation function is the sigmoid function
        self.activation_function = lambda x:s.expit(x)

        pass
    
    def weight_matrix(self):
        # link weight matrices,wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        # 正态分布 均值 标准差 矩阵大小
        self.wih = np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = np.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        pass

    # train the neural network
    def train(self,inputs_list,targets_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list,ndmin=2).T
        targets = np.array(targets_list,ndmin=2).T
        # calculate the signals into hidden layer
        hidden_inputs = np.dot(self.wih,inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate the signals into final layer
        final_inputs = np.dot(self.who,hidden_outputs)
        # calculate the signals emerging from final layer
        final_outputs = self.activation_function(final_inputs) 
        
        # error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights,recombined at hidden nodes
        hidden_errors = np.dot(self.who.T,output_errors)
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors*final_outputs*(1.0-final_outputs)),np.transpose(hidden_outputs))
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)),np.transpose(inputs))
        pass
    
    # query the neural network
    def query(self,inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list,ndmin=2).T
        
        # calculate the signals into hidden layer
        hidden_inputs = np.dot(self.wih,inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate the signals into final layer
        final_inputs = np.dot(self.who,hidden_outputs)
        # calculate the signals emerging from final layer
        final_outputs = self.activation_function(final_inputs) 
        
        return final_outputs
    
    
if __name__=="__main__":
    
    score_card = []
    
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10
    
    learning_rate = 0.1
    
    n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
    
    training_data_file = open("mnist_dataset/mnist_train.csv",'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()
    
    # epochs is the number of times the training data set is used for training
    epochs = 5
    for e in range(epochs):
        for record in training_data_list:
            all_values = record.split(',')
            inputs = (np.asfarray(all_values[1:])/255.0*0.99)+0.01
            targets = np.zeros(output_nodes)+0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)
        
    test_data_file = open("mnist_dataset/mnist_test.csv",'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    
    for test in test_data_list:
        all_values = test.split(',')
        r = n.query(np.asfarray(all_values[1:]))
        i = np.argmax(r)
        
        if int(all_values[0]) != i:
            score_card.append(0)
        else:
            score_card.append(1)
    score_array = np.asfarray(score_card)
    print("performance = ",score_array.sum()/score_array.size)
    # 95.8%
    
    