#!/usr/bin/python3
# TODO: description
import os
import argparse
import sys
import shutil
import pandas as pd
import numpy as np
import yaml
import time



##### Starting point of any test done #####
#
# labelColumn is set via commandline
# delta is set in main as 1 / (databasesize*10)
config_base = {"sgdOptions":
                {"alpha":  0.01,
                 "max_iter": 100, #TODO: change back to 1000
                 "lambda_": 0.5,
                 "tolerance": 0.000005,
                 "L": 0.5},
                "labelColumn": None,
                "communication_rounds": 2, #TODO change back to 10?
                "dpMode": [],
                "dpOptions":
                  {"epsilon":  0.1,
                   "delta":  None,
                   "C": 1}
                }

numClientsDefault = 2 # default number of clients
numFoldsCrossValidation = 5 # default number of folds for numFoldsCrossValidation
                            # 1 fold will be used for testing, one for privacy
                            # attacks and the rest for training, so at least
                            # 3 folds are necessary


def TESTING(dfTotal, locationfolder, port):
  """
  TESTING, change accordingly if other tests are wanted
  """
  curFold = 0
  for dfTest, dfPrivacytest, dfTrain, foldInfoDict in fold_generator(dfTotal):
    curFold += 1
    config_base.update(foldInfoDict)

    print(f"Starting Fold {curFold}")
    # TEST number clients
    print("Running test number clients:")
    # Warning, for iris, don't use more than 12 clients, the dataset is too
    # small
    testNumClients = [1] #[1, 2, 4, 8, 12, 16, 24, 32]
    print("numClients checked = {}".format(testNumClients))
    config_running = config_base.copy()
    for numClients in testNumClients:
        run_test(config_running, dfTrain, dfTest, locationfolder, port,
                numClients = numClients, dataDistribution = None,
                resetClientDirs = True)
    #TODO: rmv this
    exit()
    print("______________________________________________________")
    # TEST number aggregation rounds (communication_rounds)
    print("Running test number communication_rounds:")
    testComRounds = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    print("com_rounds checked = {}".format(testComRounds))
    for com_rounds in testComRounds:
      config_running = config_base.copy()
      config_running["communication_rounds"] = com_rounds
      run_test(config_running, dfTrain, dfTest, locationfolder, port,
              numClients = numClientsDefault, dataDistribution = None)
    print("______________________________________________________")


    # TEST sample size (L)
    print("Running test samplingRatio:")
    testSamplingRatio = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, None]
    print("samplingRatio checked = {}".format(testSamplingRatio))
    for samplingRatio in testSamplingRatio:
      config_running = config_base.copy()
      config_running["sgdOptions"]["L"] = samplingRatio
      run_test(config_running, dfTrain, dfTest, locationfolder, port,
              numClients = numClientsDefault, dataDistribution = None)
    print("______________________________________________________")


    # TEST epsilon, using gauss noise
    print("Running test epsilon")
    testEpsilon = [0.001, 0.01, 0.1, 0.2, 0.4, 0.8]
    print("epsilon checked = {}".format(testEpsilon))
    for epsilon in testEpsilon:
      config_running = config_base.copy()
      config_running["dpOptions"]["epsilon"] = epsilon
      run_test(config_running, dfTrain, dfTest, locationfolder, port,
              numClients = numClientsDefault, dataDistribution = None)
    print("______________________________________________________")


#####                                                                     #####
##### TO CHANGE TESTS, CHANGE TESTING FUNCTION / config_base / numClientsDefault #####
##### EVERYTHING ELSE DOES NOT HAVE TO BE CHANGED                         #####
#####                                                                     #####

def fold_generator(df):
  chunk_size = int(df.shape[0] / numFoldsCrossValidation)
  curFold = 1
  for start in range(0, df.shape[0] - chunk_size, chunk_size):
    yield df.iloc[start:start + chunk_size], \
          df.iloc[start + chunk_size:start + chunk_size * 2], \
          df.drop(df.iloc[start:start + chunk_size * 2].index, inplace = False), \
          {"foldTest": [start, start + chunk_size - 1],
           "foldPrivacytest": [start + chunk_size, start + chunk_size * 2 - 1]
          }
    # yield dfTest, dfPrivacytest, dfTrain



##### Helper, actually runs the tests #####
def run_test(configDict, dfTrain, dfTest, locationfolder, port, numClients,
             dataDistribution = None, resetClientDirs = False):
  if not resetClientDirs:
    # check if dirs are there
    clientCount = 0
    for folder in os.scandir(locationfolder):
      if os.path.basename(folder)[0:6] == "client":
        clientCount += 1
        # overwrite config file
        with open(os.path.join(locationfolder, folder,
                               "config.yaml"), 'w') as fp:
          yaml.dump(configDict, fp)
    if clientCount != numClients:
      # incorrect client structure currently in data folder, reset it
      resetClientDirs = True

  if resetClientDirs:
    # remove all client dirs
    for folder in os.scandir(locationfolder):
      if os.path.basename(folder)[0:6] == "client":
        shutil.rmtree(folder)
    # create client dirs
    for numb in range(numClients):
      os.mkdir(os.path.join(locationfolder, "client" + str(numb)))

    # fill directories with training_data and config file
    # read data
    # create splits
    chunk_size = int(dfTrain.shape[0] / numClients)
    curClient = 0
    for start in range(0, dfTrain.shape[0], chunk_size):
      if curClient == numClients-1:
        # use all remaining data
        df_subset = dfTrain.iloc[start:]
      else:
        df_subset = dfTrain.iloc[start:start + chunk_size]
      # write split in corresponding folder
      df_subset.to_csv(os.path.join(locationfolder, "client" + str(curClient), "training_set.csv"), index=False)
      # write config
      with open(os.path.join(locationfolder, "client" + str(curClient),   "config.yaml"), 'w') as fp:
        yaml.dump(configDict, fp)
      curClient += 1
      if curClient == numClients:
        # all data iterated
        break

    # fill directories with test_data
    # create splits
    chunk_size = int(dfTest.shape[0] / numClients)
    curClient = 0
    for start in range(0, dfTest.shape[0], chunk_size):
      if curClient == numClients-1:
        # use all remaining data
        df_subset = dfTest.iloc[start:]
      else:
        df_subset = dfTest.iloc[start:start + chunk_size]
      # write split in corresponding folder
      df_subset.to_csv(os.path.join(locationfolder, "client" + str(curClient), "test_set.csv"), index=False)
      curClient += 1
      if curClient == numClients:
        # all data iterated
        break


  # Start test
  os.popen("featurecloud test start " +\
      "--controller-host=http://localhost:{} ".format(port) +\
      "--client-dirs={} ".format(','.join(["./client" + str(x) for x in \
        range(numClients)])) +\
      "--generic-dir=./ --app-image=fc-private-logistic-regression " +\
      "--channel=local --query-interval=2 --download-results=./output")
  # check if test is done
  time.sleep(10) # wait before checking if test is still running to let
  while True:
    listoutput = os.popen("featurecloud test list").read()
    if not "running" in listoutput and not "init" in listoutput:
      break
    time.sleep(5)




##### Read in Information and do tests #####
if __name__ == "__main__":
  ##### Parser #####
  parser = argparse.ArgumentParser(
    description="testing script for logistic regression, " + \
                "controller must be running")
  parser.add_argument("-cd", "--controller-data-folder", dest="location",
                      nargs = 1,
                      help = "location of the data folder, automatically " + \
                      "created when starting the controller, contains input " +\
                      "and output.", required = True)
  parser.add_argument("-l", "--label-column", dest="label", nargs = 1,
                      help = "Name of the column to be predicted, " +\
                      "(y column), use target for iris", required = True)
  parser.add_argument("-dp", "--dp-mode", choices = ["dpSGD", "OutputDP", "noDP"],
                      nargs = 1, dest="dp", default = ["dpSGD"],
                      help = "Where to add DP, either during " + \
                      "stochastic gradient descent, during aggregation or no " \
                      "added DP.")
  parser.add_argument("-d", "--data", nargs = 1,
                      default = ["iris"], dest = "data",
                      help = "Either \"iris\" to use the iris dataset or " +\
                      "the folder containing data.csv. Default uses " +\
                      "iris dataset,")
  parser.add_argument("-p", "--port-controller", dest="port", nargs = 1,
                      default=[8000], help="Port of the fc-controller, " +\
                        "usually 8000 or 8002")
  args = parser.parse_args()

  ##### change config_base ####
  # load in dpMode
  config_base["dpMode"] = args.dp

  # read in data
  datafolder = args.data[0]
  if datafolder == "iris":
    from sklearn import datasets
    dfTotal = datasets.load_iris(as_frame = True).frame
    config_base["labelColumn"] = "target"
  else:
    dfTotal = pd.read_csv(os.path.join(datafolder, "data.csv"))
    config_base["labelColumn"] = args.label[0]

  # change delta accordingly to 1/ databasesize * 10
  databasesize = dfTotal.shape[0]

  config_base["dpOptions"]["delta"] = 1 / (databasesize * 10)

  # read out args

  locationfolder = args.location[0]
  port = args.port[0]
  print("Starting tests with the following baseline:")
  print(config_base)
  print("numClients: {}".format(numClientsDefault))
  print("______________________________________________________")
  buildoutput = os.popen('sh build.sh')
  print(buildoutput.read())
  print("______________________________________________________")

  # shuffle data once to remove any ordering
  dfTotal = dfTotal.sample(frac = 1.0, random_state = 42)

  # make dir for output
  outputDir = os.path.join(os.getcwd(), "outputTestScript")
  if not os.path.isdir(outputDir):
    os.mkdir(outputDir)

  # save shuffled data as output
  dfTotal.to_csv(os.path.join(outputDir, "data_shuffled.csv"), index = False)

  TESTING(dfTotal, locationfolder, port)

  # check if any errors and report
  listoutput = os.popen("featurecloud test list").read()
  if "error" in listoutput:
    print("WARNING: The controller session seems to contain at least one " +\
          "ERROR, check the logs and run featurecloud test list to see more")













