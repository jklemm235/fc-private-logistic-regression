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
        ### load in config self.internal["epsilon", "delta", ...]
        if os.path.exists(os.path.join(os.getcwd(), "mnt", "input",
                                       "config.yaml")):
            configFileName = "config.yaml"
        elif os.path.exists(os.path.join(os.getcwd(), "mnt", "input",
                                         "config.yml")):
            configFileName = "config.yml"
        else:
            self.log("Error, no config.yaml file found, aborting",
                     level = LogLevel.FATAL)

        try:
            with open(os.path.join(os.getcwd(), "mnt", "input",
                                   configFileName), 'r') as stream:
                config = yaml.safe_load(stream)
        except Exception as err:
            self.log(f"Error loading config.yaml: f{err}",
                     level = LogLevel.FATAL)

        # save config (with numClients) #TODO: rmv in Master
        if self.is_coordinator:
            config["numClients"] = len(self.clients)
            fp = open(os.path.join("mnt", "output", "config.yaml"), "w")
            yaml.dump(config, fp)


        self.store(key = "config", value = config)

        if self.is_coordinator:
            self.store(key = "cur_communication_round", value = 0)

        ### SGD Class creation
        DPSGD_class = algo.LogisticRegression_DPSGD()
        # DP information
        if "DPSGD" in [x.upper() for x in config["dpMode"]]:
            DPSGD_class.DP = True
        else:
            DPSGD_class.DP = False

        try:
            DPSGD_class.alpha = config["sgdOptions"]["alpha"]
            DPSGD_class.max_iter = config["sgdOptions"]["max_iter"]
            DPSGD_class.lambda_ = config["sgdOptions"]["lambda_"]
            DPSGD_class.tolerance = config["sgdOptions"]["tolerance"]
            DPSGD_class.L = config["sgdOptions"]["L"]
            DPSGD_class.C = config["dpOptions"]["C"]
            DPSGD_class.epsilon = config["dpOptions"]["epsilon"]
            DPSGD_class.delta = config["dpOptions"]["delta"]
            labelCol = config["labelColumn"]
        except Exception as err:
            self.log(f"Config file seems to miss fields: {err}",
                     level = LogLevel.FATAL)

        #TODO add outputDP here as well
        if DPSGD_class.DP:
            if DPSGD_class.delta < 0 or DPSGD_class.delta >= 1:
                self.log("delta must be [0,1)",
                         level = LogLevel.FATAL)
            elif DPSGD_class.delta != 0 and DPSGD_class.epsilon >= 1:
                self.log("When delta is >= 0, gauss noise is used. " +\
                    "For gauss noise, epsilon must be between 0 and 1",
                    level = LogLevel.FATAL)
        print(vars(DPSGD_class))
        ### Load in Data

        # check if data files exist
        if not os.path.exists(os.path.join(os.getcwd(), "mnt",
                                           "input","training_set.csv")):
            self.log("Could not find training_set.csv containing training data",
                        level = LogLevel.FATAL)
        if not os.path.exists(os.path.join(os.getcwd(), "mnt",
                                           "input","test_set.csv")):
            self.log("Could not find training_set.csv containing test data",
                        level = LogLevel.FATAL)

        # load in data
        train_data = pd.read_csv(os.path.join(os.getcwd(), "mnt", "input","training_set.csv"))
        test_data = pd.read_csv(os.path.join(os.getcwd(), "mnt", "input","test_set.csv"))

        # preprocess data
        X_train = np.array(train_data.drop(columns=[labelCol]))
        X = np.array(X_train)
        y_train = np.array(train_data[labelCol])

        X_test = np.array(test_data.drop(columns=[labelCol]))
        y_test = np.array(test_data[labelCol])
        self.store(key = "X_test", value = X_test)
        self.store(key = "y_test", value = y_test)

        # load in weights and other parameters
        n, d = X.shape
        self.store(key = "n", value = n)
        self.store(key = "d", value = d)

        if not DPSGD_class.L:
            # use all data if L = nil was given
            DPSGD_class.L = n
        elif DPSGD_class.L < 1:
            # change L to the correct value if L is a percentage
            DPSGD_class.L = int(DPSGD_class.L * n)
        # else keep L as it is

        # modify data depending on which prediction function is used
        # (binary vs multiple classes)
        X, y_train = DPSGD_class.init_theta(X, y_train)
        self.store(key = "X", value = X)
        self.store(key = "y_train", value = y_train)

        self.store(key="DPSGD_class", value = DPSGD_class)

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
        return "local_computation"

@app_state("local_computation")
class localComputationState(AppState):

    def register(self):
        self.register_transition("aggregate_data", Role.COORDINATOR)
        self.register_transition("obtain_weights", Role.PARTICIPANT)

    def run(self):
        # Set parameter
        X = self.load("X")
        y = self.load("y_train")
        n = self.load("n")
        d = self.load("d")

        # Training
        DPSGD_class = self.load("DPSGD_class")
        DPSGD_class.train(X, y)
        self.store(key="DPSGD_class", value = DPSGD_class)


        # save theta of each client
        cur_computation_round = self.load("cur_computation_round") + 1
        self.store(key = "cur_computation_round", value = cur_computation_round)
        fp = open(os.path.join("mnt", "output", "model_{}_{}.pyc".format(
            self.id, cur_computation_round)), 'wb')
        np.save(fp, DPSGD_class.theta)

        # local update
        if self.is_coordinator:
            self.send_data_to_coordinator(DPSGD_class.theta,
                                          send_to_self = True, use_smpc = False)
            return "aggregate_data"
        else:
            self.send_data_to_coordinator(DPSGD_class.theta,
                                        send_to_self = False, use_smpc = False)
            return "obtain_weights"




@app_state("aggregate_data")
class aggregateDataState(AppState):

    def register(self):
        self.register_transition("obtain_weights", Role.COORDINATOR)
        self.register_transition("terminal", Role.COORDINATOR)

    def run(self):
        # aggregate the weights
        weights_updated = self.aggregate_data(use_smpc=False)
        weights_updated = weights_updated / len(self._app.clients)

        # update communication_rounds
        cur_comm = self.load("cur_communication_round") + 1
        self.store(key = "cur_communication_round",
                        value = cur_comm)

        if cur_comm >= self.load("config")["communication_rounds"]:
            # finnished
            fp = open(os.path.join("mnt", "output", "trained_model.pyc".format(cur_comm)), "wb")
            np.save(fp, weights_updated)

            DPSGD_class = self.load("DPSGD_class")
            DPSGD_class.theta = weights_updated
            acc, confMat = DPSGD_class.evaluate(X = self.load("X_test"),
                                                y = self.load("y_test"))


            # write evaluation results in output #TODO remove this from master
            config = self.load("config")
            config["accuracy"] = acc.item()
            config["conf_matrix"] = confMat.tolist()


            # overwrite config file output
            fp = open(os.path.join("mnt", "output", "config.yaml"), "w")
            yaml.dump(config, fp)

            return "terminal"
        else:
            # save agg_model
            fp = open(os.path.join("mnt", "output", "aggmodel_{}.pyc".format(cur_comm)), "wb")
            np.save(fp, weights_updated)

            # send data to clients
            self.broadcast_data(weights_updated, send_to_self = True)
            return "obtain_weights"


