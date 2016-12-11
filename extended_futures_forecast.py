# extended_futures_forecast_v01.py 
"""
Created on 10/8 2016 by Linchang.

Replicating SP index futures strategy. 
Par 2 of SPY forecast model: extended futures forecast model.

Update in v02 (11/16): Standardized syntax based on Pylint feedback
"""

import pandas as pd
import neural_network_bp
import neural_network_pn


def train_bp(train_set, num_hidden=4, num_train=1000, least_false=0.5):
    """Train nn using BP.

    train_set
    num_hidden: number of hidden nodes, default set to 4.
    num_train: number of training times, default set to 1000
    least_false: Error rate threshold to stop training. Error rate is defined as
                 (total error/number of training samples). Can set the threshold
                 low enough to complete the specified training times.
                 Default is 0.5.
    """

    nn = neural_network_bp.NeuralNetworkBP(len(train_set[0][0]),
                                           num_hidden,
                                           len(train_set[0][1]))

    for i in range(num_train):
        train_inputs, train_outputs = neural_network_bp.random.choice(train_set)
        nn.train(train_inputs, train_outputs)
        print(i, nn.calculate_total_error(train_set),
              train_inputs, nn.output_layer.get_outputs())
        if nn.calculate_total_error(train_set) < least_false:
            break

    return nn


def train_pn(train_set, num_train=1000, least_false=0.2):
    """Train nn using PN.

    train_set
    num_train: number of training times, default set to 1000
    least_false: Error rate thresthold to stop training. Error rate is defined as
                 (total error/number of training samples). Can set the threshold
                 low enough to complete the specified training times.
                 Default is 0.2.
    """

    nn = neural_network_pn.NeuralNetworkPN(len(train_set[0][0]),
                                           len(train_set[0][1]))

    for i in range(num_train):
        train_inputs, train_outputs = neural_network_bp.random.choice(train_set)
        nn.train(train_inputs, train_outputs)
        print(i, nn.calculate_total_error(train_set),
              train_inputs, nn.output_layer.get_outputs())
        if nn.calculate_total_error(train_set) < least_false:
            break

    return nn


class ExtendedFuturesForecastModel:
    """Implements the extended future forecast model with input data.
    
    Steps:
        1. Extract only previous Obvious cases as training sample.
        2. Create specific training set for each of the four triggers.
        3. Train the neural network (Backpropagation or Perceptron).
        4. Use the neural network to generate recommendations for the NON_Obvious 
           forecast cases.
       
    Attributes:
        trained_case: Obvious cases.
        train: training sample corresponds to Obvious cases.
    """
    
    def __init__(self, train, trained_case, forecast):
        # use previous Obvious cases
        self.trained_case = trained_case[trained_case["case_result"].isin(["Obvious_LONG",
                                                                           "Obvious_SHORT",
                                                                           "Obvious_WAIT"])]
        self.train = train.loc[self.trained_case.index]  # use corresponding training sample
        self.forecast = forecast

        self.train_set_lu = []
        self.train_set_ld = []
        self.train_set_rsi1 = []
        self.train_set_rsi2 = []
        self.nn_lu = None
        self.nn_ld = None
        self.nn_rsi1 = None
        self.nn_rsi2 = None
        self.updated_forecast_case = None
    
    def create_train_set(self):
        target = pd.Series(([],)*self.train.shape[0], index=self.train.index)
        for i, index in enumerate(self.trained_case[self.trained_case.case_result == "Obvious_LONG"].index):
            target.loc[index] = [1, -1]
        for i, index in enumerate(self.trained_case[self.trained_case.case_result == "Obvious_SHORT"].index):
            target.loc[index] = [-1, 1]
        for i, index in enumerate(self.trained_case[self.trained_case.case_result == "Obvious_WAIT"].index):
            target.loc[index] = [-1, -1]

        for i in range(len(self.train[self.train.lu == 1])):
            self.train_set_lu.append([list(self.train[self.train.lu == 1].ix[i, ["sp", "sn", "ud", "aud",
                                                                                 "rsi3", "rsi4"]]),
                                     target[self.train.lu == 1].iloc[i]])
        for i in range(len(self.train[self.train.ld == 1])):
            self.train_set_ld.append([list(self.train[self.train.ld == 1].ix[i, ["sp", "sn", "ud", "aud",
                                                                                 "rsi3", "rsi4"]]),
                                      target[self.train.ld == 1].iloc[i]])
        for i in range(len(self.train[self.train.rsi1 == 1])):
            self.train_set_rsi1.append([list(self.train[self.train.rsi1 == 1].ix[i, ["sp", "sn", "ud", "aud",
                                                                                     "rsi3", "rsi4"]]),
                                       target[self.train.rsi1 == 1].iloc[i]])
        for i in range(len(self.train[self.train.rsi2 == 1])):
            self.train_set_rsi2.append([list(self.train[self.train.rsi2 == 1].ix[i, ["sp", "sn", "ud", "aud",
                                                                                     "rsi3", "rsi4"]]),
                                       target[self.train.rsi2 == 1].iloc[i]])

    def train_lu(self, method="bp", num_hidden=4, num_train=1000, least_false=0.2):
        if method == "bp":
            print "\nTraining neural network for lu using Backpropagation\n"
            self.nn_lu = train_bp(self.train_set_lu, num_hidden, num_train, least_false)
        elif method == "pn":
            self.nn_lu = train_pn(self.train_set_lu, num_train, least_false)
            print "\nTraining neural network for lu using Perceptron\n"
        else:
            raise Exception("Incorrect neural network method!")
        return self.nn_lu

    def train_ld(self, method="bp", num_hidden=4, num_train=1000, least_false=0.2):
        if method == "bp":
            print "\nTraining neural network for ld using Backpropagation\n"
            self.nn_ld = train_bp(self.train_set_ld, num_hidden, num_train, least_false)
        elif method == "pn":
            self.nn_ld = train_pn(self.train_set_ld, num_train, least_false)
            print "\nTraining neural network for ld using Perceptron\n"
        else:
            raise Exception("Incorrect neural network method!")
        return self.nn_ld

    def train_rsi1(self, method="bp", num_hidden=4, num_train=1000, least_false=0.2):
        if method == "bp":
            print "\nTraining neural network for rsi1 using Backpropagation\n"
            self.nn_rsi1 = train_bp(self.train_set_rsi1, num_hidden, num_train, least_false)
        elif method == "pn":
            print "\nTraining neural network for rsi1 using Perceptron\n"
            self.nn_rsi1 = train_pn(self.train_set_rsi1, num_train, least_false)
        else:
            raise Exception("Incorrect neural network method!")
        return self.nn_rsi1

    def train_rsi2(self, method="bp", num_hidden=4, num_train=1000, least_false=0.2):
        if method == "bp":
            print "\nTraining neural network for rsi2 using Backpropagation\n"
            self.nn_rsi2 = train_bp(self.train_set_rsi2, num_hidden, num_train, least_false)
        elif method == "pn":
            print "\nTraining neural network for rsi2 using Perceptron\n"
            self.nn_rsi2 = train_pn(self.train_set_rsi2, num_train, least_false)
        else:
            raise Exception("Incorrect neural network method!")
        return self.nn_rsi2

    def run_nn(self):
        self.updated_forecast_case = pd.Series("", index=self.forecast.index, name="case_result")

        for i in range(len(self.updated_forecast_case)):
            L = 0
            S = 0
            nn_lu_output = []
            nn_ld_output = []
            nn_rsi1_output = []
            nn_rsi2_output = []

            if self.forecast.lu[i] == 1:
                nn_lu_output = self.nn_lu.feed_forward(list(self.forecast.ix[i, ["sp", "sn", "ud",
                                                                                 "aud", "rsi3", "rsi4"]]))
            if self.forecast.ld[i] == 1:
                nn_ld_output = self.nn_ld.feed_forward(list(self.forecast.ix[i, ["sp", "sn", "ud",
                                                                                 "aud", "rsi3", "rsi4"]]))
            if self.forecast.rsi1[i] == 1:
                nn_rsi1_output = self.nn_rsi1.feed_forward(list(self.forecast.ix[i, ["sp", "sn", "ud",
                                                                                     "aud", "rsi3", "rsi4"]]))
            if self.forecast.rsi2[i] == 1:
                nn_rsi2_output = self.nn_rsi2.feed_forward(list(self.forecast.ix[i, ["sp", "sn", "ud",
                                                                                     "aud", "rsi3", "rsi4"]]))

            if nn_lu_output == [1, -1]:
                L += 1
            elif nn_lu_output == [-1, 1]:
                S += 1
            if nn_ld_output == [1, -1]:
                L += 1
            elif nn_ld_output == [-1, 1]:
                S += 1
            if nn_rsi1_output == [1, -1]:
                L += 1
            elif nn_rsi1_output == [-1, 1]:
                S += 1
            if nn_rsi2_output == [1, -1]:
                L += 1
            elif nn_rsi2_output == [-1, 1]:
                S += 1

            if L > S:
                self.updated_forecast_case[i] = "LONG"
            elif L < S:
                self.updated_forecast_case[i] = "SHORT"
            elif L == S:
                self.updated_forecast_case[i] = "WAIT"

        self.updated_forecast_case = pd.DataFrame(self.updated_forecast_case)
        self.updated_forecast_case.columns = ["case_result"]

        return self.updated_forecast_case

    def evaluate_cases(self):
        print "EFFM trading recommendations by category: \n%s\n" % (self.updated_forecast_case.groupby("case_result").size())
        print "Most recent EFFM trading recommendations: \n%s\n" % (self.updated_forecast_case.tail())
