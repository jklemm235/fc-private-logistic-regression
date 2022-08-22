"""
This demo implementation works as follows:
1. The coordinator sends a message to a random participant
2. The receiver adds it's own ID to the message and sends it again to a random participant
3. When the message bounced n times (n = number of clients), it's sent to the coordinator with a 'DONE:' prefix
4. The coordinator stops the "computation"
"""




from FeatureCloud.app.engine.app import AppState, app_state, Role, LogLevel
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
        self.register_transition("local_computation", Role.BOTH)

    def run(self):
        # load in config self.internal["epsilon", "delta", ...]
        #TODO: remove this
        self.log("Init fc-private-logistic-regression start", level = LogLevel.DEBUG)
        print("Inital state start")
        try:
            #print(os.listdir(os.getcwd()))
            #print('Home contains:')
            #print(os.listdir(
            #    os.path.join(os.getcwd(), "home")))
            #print('mnt/input contains:')
            #print(os.listdir(
            #    os.path.join(os.getcwd(), "mnt", "input")))
            #TODO: remove prints
            with open(os.path.join(os.getcwd(), "mnt", "input", "config.yaml"), 'r') as stream:
                config = yaml.safe_load(stream)
                print(config) #TODO: rmv
                #TODO: fix open to work without the hack of copying everything to
                # data
        except Exception as err:
            self.log(f"Error loading config.yaml: f{err}" , level = LogLevel.FATAL)

        self.store(key = "config", value = config)

        # load in data in self.internal["data"]
        #TODO: fix so it always loads data correctly when it is known how docker works
        train_data = pd.read_csv(os.path.join(os.getcwd(), "mnt", "input","training_set.csv"), index_col=0)
        test_data = pd.read_csv(os.path.join(os.getcwd(), "mnt", "input","test_set.csv"), index_col=0)

        X_train = np.array(train_data.drop(columns=['label']))
        X = np.array(X_train)
        y_train = np.array(train_data['label'])
        print("Shape of loaded data is:") #TODO: rmv
        print(X.shape) #TODO: rmv
        print(y_train.shape) #TODO: rmv
        # not put in dict yet, they will be modified first by the LogisticRegression_DPSGD class

        X_test = np.array(test_data.drop(columns=['label']))
        y_test = np.array(test_data['label'])
        self.store(key = "X_test", value = X_test)
        self.store(key = "y_test", value = y_test)

        # load in weights and other parameters
        n, d = X.shape
        self.store(key = "n", value = n)
        self.store(key = "d", value = d) #TODO: works if just one row in predictions
                # are there models that predict n values and therefore n cols?
        if self.is_coordinator:
            self.store(key = "cur_communication_round", value = 0)

        # DP information
        #TODO: also other DP modes
        if "dpSgd" in config["dpMode"]:
            withDPSGD = True
        else:
            withDPSGD = False

        # SGD Class creation
        DPSGD_class = algo.LogisticRegression_DPSGD()
        try:
            DPSGD_class.alpha = config["sgdOptions"]["alpha"]
            DPSGD_class.max_iter = config["sgdOptions"]["max_iter"]
            DPSGD_class.lambda_ = config["sgdOptions"]["lambda_"]
            DPSGD_class.tolerance = config["sgdOptions"]["tolerance"]
            DPSGD_class.L = config["sgdOptions"]["L"]
            DPSGD_class.C = config["dpOptions"]["C"]
            DPSGD_class.epsilon = config["dpOptions"]["epsilon"]
            DPSGD_class.delta = config["dpOptions"]["epsilon"]
            #TODO: theta should be read in here
        except Exception as err:
            self.log(f"Config file seems to miss fields: {err}")

        # TODO fix theta, should be in config file
        print("Shape of X and y_train and theta before storing") #TODO; rmv
        X, y_train = DPSGD_class.init_theta(X, y_train)
        print(X.shape) #TODO rmbv
        print(y_train.shape) #TODO rmbv
        print(DPSGD_class.theta.shape) #TODO: rmv
        self.store(key = "X", value = X)
        self.store(key = "y_train", value = y_train)

        #TODO: change sigma to epsilon + delta and calc sigma, also change config.yaml accordingly
        print(vars(DPSGD_class)) #TODO: rmv
        self.store(key="DPSGD_class", value = DPSGD_class)
        self.log("Init fc-private-logistic-regression end", level = LogLevel.DEBUG) #TODO: rmv

        self.store(key = "cur_computation_round", value = 0)
        return "local_computation"

@app_state("obtain_weights")
class obtainWeights(AppState):

    def register(self):
        self.register_transition("local_computation", Role.BOTH)

    def run(self):
        # update from broadcast_data
        DPSGD_class = self.load("DPSGD_class")
        DPSGD_class.theta = self.await_data(n = 1, unwrap=True, is_json=False)
        self.store(key="DPSGD_class", value = DPSGD_class)
        print(vars(DPSGD_class))
        print("obtained weights: {}".format(DPSGD_class.theta))
        return "local_computation"

@app_state("local_computation")
class localComputationState(AppState):

    def register(self):
        self.register_transition("aggregate_data", Role.COORDINATOR)
        self.register_transition("obtain_weights", Role.PARTICIPANT)

    def run(self):
        random.seed(10) #TODO: remove this
        # Set parameter
        X = self.load("X")
        y = self.load("y_train")
        n = self.load("n")
        d = self.load("d")

        "Train Logistic regression with SGD"
        DPSGD_class = self.load("DPSGD_class")
        print("Training with the following shaped data:") #TODO rmv
        print(X.shape)
        print(y.shape)
        DPSGD_class.train(X, y)
        self.store(key="DPSGD_class", value = DPSGD_class)
        print("Local Training finished, updated class is:") #TODO rmv
        print(vars(DPSGD_class)) #TODO rmv
        print(DPSGD_class.theta.shape) #TODO: rmv

        # save theta of each client
        cur_computation_round = self.load("cur_computation_round") + 1
        self.store(key = "cur_computation_round", value = cur_computation_round)
        fp = open(os.path.join("mnt", "output", "model_{}_{}.pyc".format(
            self.id, cur_computation_round)), 'wb')
        np.save(fp, DPSGD_class.theta)

        # local update
        if self.is_coordinator:
            #TODO: add dp noise here possibly
            self.send_data_to_coordinator(DPSGD_class.theta, send_to_self = True, use_smpc = False)
            return "aggregate_data"
        else:
            self.send_data_to_coordinator(DPSGD_class.theta, send_to_self = False, use_smpc = False)
            #TODO: possibly an infinite loop here? Or does the coordinator
            # finnishing also terminate every client?
            return "obtain_weights"




@app_state("aggregate_data")
class aggregateDataState(AppState):

    def register(self):
        self.register_transition("obtain_weights", Role.COORDINATOR)
        self.register_transition("terminal", Role.COORDINATOR)

    def run(self):
        # TODO: how to manage coordinator adding noise, best add func in template
        weights_updated = self.aggregate_data(use_smpc=False)
        weights_updated = weights_updated / len(self._app.clients)
        print(weights_updated.shape) #TODO rmv
        print("aggregated weights:") #TODO: remove prints
        print(weights_updated) #TODO rmv
        cur_comm = self.load("cur_communication_round") + 1
        self.store(key = "cur_communication_round",
                        value = cur_comm)
        fp = open(os.path.join("mnt", "output", "aggmodel_{}.pyc".format(cur_comm)), "wb")
        np.save(fp, weights_updated)
        print("cur_comm is {}".format(cur_comm))
        print("max_comm is {}".format(self.load("config")["communication_rounds"]))
        if cur_comm >= self.load("config")["communication_rounds"]:
            # finnished
            #TODO: safe result in file?
            print("Done:")  #TODO: remove these lines
            print(weights_updated) #TODO rmv
            print(weights_updated.shape)
            print("Shape of test data to be evaluated:") #TODO: rmv
            print(self.load("X_test").shape) #TODO: rmv
            print(self.load("y_test").shape) #TODO: rmv
            #TODO send data to itself for adding dp necessary
            DPSGD_class = self.load("DPSGD_class")
            DPSGD_class.theta = weights_updated
            DPSGD_class.evaluate(X = self.load("X_test"), y = self.load("y_test"))
            # save config (with numClients)
            config = self.load("config")
            config["numClients"] = len(self.clients)
            fp = open(os.path.join("mnt", "output", "config.yaml"), "w")
            yaml.dump(config, fp)
            return "terminal"
        else:
            # send data to clients
            self.broadcast_data(weights_updated, send_to_self = True)
            #TODO: careful, send_to_self just means the instance adds the weights
            # to its own incoming vector
            return "obtain_weights"


