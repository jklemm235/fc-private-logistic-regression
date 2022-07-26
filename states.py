"""
This demo implementation works as follows:
1. The coordinator sends a message to a random participant
2. The receiver adds it's own ID to the message and sends it again to a random participant
3. When the message bounced n times (n = number of clients), it's sent to the coordinator with a 'DONE:' prefix
4. The coordinator stops the "computation"
"""




from FeatureCloud.app.engine.app import AppState, app_state, Role
import algo

import os
from time import sleep
import yaml
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random


@app_state("initial")
class InitialState(AppState):
    """
    Loads in data to the AppState as well as necessary hyperparameters from config.yaml
    """
    def register(self):
        self.register_transition("local_computation", role.BOTH)

    def run(self):
        # load in config self.internal["epsilon", "delta", ...]
        try:
            with open("config.yaml", 'r') as stream:
                config = yaml.safe_load(stream)
        except:
            self.log("Error loading config.yaml" , LogLevel = LogLevel.FATAL)

        self.store(key = "config", value = config)

        # load in data in self.internal["data"]
        train_data = pd.read_csv(os.path.join("data", "training_set.csv"), index_col=0)
        test_data = pd.read_csv(os.path.join("data", "test_set.csv"), index_col=0)

        X_train = np.array(train_data.drop(columns=['label']))
        X = np.array(X_train)
        y_train = np.array(train_data['label'])
        self.store(key = "X", value = X)
        self.store(key = "y_train", value = y_train)

        X_test = np.array(test_data.drop(columns=['label']))
        y_test = np.array(test_data['label'])
        self.store(key = "X_test", value = X_test)
        self.store(key = "y_test", value = y_test)

        # load in weights and other parameters
        n, d = X.shape
        self.store(key = "n", value = n)
        self.store(key = "d", value = d)
        self.store(key = "weights", value = None)
        if self.is_coordinator:
            self.store(key = "cur_communication_round", value = 0)

@app_state("local_computation")
class localComputationState(AppState):

    def register(self):
        self.register_transition("aggregate_data", roles.COORDINATOR)

    def run(self):
        random.seed(10) #TODO: remove this
        # Set parameter
        X = self.load("X")
        y = self.load("y_train")
        n = self.load("n")
        d = self.load("d")

        max_iter = self.load("config")["dpsgd"]["max_iter"]
        alpha = self.load("config")["dpsgd"]["alpha"]
        lambda_ = self.load("config")["dpsgd"]["lambda_"]
        L = self.load("config")["dpsgd"]["L"]
        C = self.load("config")["dpsgd"]["C"]
        sigma = self.load("config")["dpsgd"]["sigma"]
        delta = self.load("config")["dpsgd"]["delta"]

        weights = self.load("weights")
        if not weights:
            weights = np.ones(d)
        else:
            weights = self.await_data(n = 1, unwrap=True, is_json=False)

        "Train Logistic regression with SGD"
        #TODO: manage if called from aggregate data
        weights, cost =  algo.SGD(X, y, weights, alpha, max_iter, lambda_)
        if not weights:
            self.log("Error in Gradient descent, no data returned" , LogLevel = LogLevel.FATAL)
        self.store(key = "weights", value = weights)
        if self.is_coordinator:
            #TODO: add dp noise here possibly
            self.send_data_to_coordinator(weights, send_to_self = True, use_smpc = False)
        else:
            self.send_data_to_coordinator(weights, send_to_self = False, use_smpc = False)
        return "aggregate_data"



@app_state("aggregate_data")
class aggregateDataState(AppState):

    def register(self):
        self.register_transition("local_computation", roles.COORDINATOR)
        self.register_transition("terminal", roles.COORDINATOR)

    def run(self):
        # TODO: how to manage coordinator adding noise, best add func in template
        weights_updated = self.aggregate_data(use_smpc=False)
        cur_comm = self.load("cur_communication_round") + 1
        self.store(key = "cur_communication_round",
                        value = cur_comm)
        if cur_comm >= self.load("config")["dpsgd"]["communication_rounds"]:
            # finnished
            #TODO: safe result in file?
            print("Done:")  #TODO: remove these lines
            print(weights)
            print(analyse_and_plot())
            return "terminal"
        else:
            # send data to clients
                broadcast_data(weights_updated, send_to_self = False)


