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
import math
import re
import zipfile
import pandas as pd
from FeatureCloud.api.imp.test import commands as fc

##### Starting point of any test done #####
# NOTE: DO NOT USE sgdOptions.L = 1 if you want to consider ALL data,
# all values under 1 get seen as percentage, but 1 gets seen as just use one
# sample!!
#
# labelColumn is set via commandline
# delta is set in main as 1 / (databasesize*10)
config_base = {"sgdOptions":
                {"alpha":  0.1,
                 "max_iter": 500,
                 "lambda_": 0.01,
                 "tolerance": 1e-5,
                 "L": 1.0},
                "labelColumn": None,
                "communication_rounds": 5,
                "dpMode": [],
                "dpOptions":
                  {"epsilon":  0.1,
                   "delta":  None,
                   "C": 1}
                }

numClientsDefault = 3 # default number of clients
numFoldsCrossValidation = 5 # default number of folds for numFoldsCrossValidation
                            # 1 fold will be used for testing, one for privacy
                            # attacks and the rest for training, so at least
                            # 3 folds are necessary


def TESTING(dfTotal, locationfolder, port, controllerfolder):
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
    testNumClients = [1,3,5,7]
    testComRounds = [1, 5, 10, 15]
    testEpsilon = [0.01, 0.1, 0.4, 0.8, 3.0]
    config_running = config_base.copy()
    config_running["dpOptions"]["delta"] = 0 # to use laplace noise
    for numClients in testNumClients:
      print("Num Clients: {}".format(numClients))
      for com_rounds in testComRounds:
        print("Com_rounds: {}".format(com_rounds))
        config_running["communication_rounds"] = com_rounds
        for epsilon in testEpsilon:
          print("Epsilon: {}".format(epsilon))
          config_running["dpOptions"]["epsilon"] = epsilon
          run_test(config_running, dfTrain, dfTest,
                              locationfolder, port, controllerfolder,
                              numClients = numClients,
                              dataDistribution = None, resetClientDirs = True)
    print("______________________________________________________")

  return None


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
def run_test(configDict, dfTrain, dfTest, locationfolder, port,
             controllerfolder, numClients,
             dataDistribution = None, resetClientDirs = True,
             num_redos = 5):
  print("Running a test")
  #Save and print noise used for dp client
  if "DPCLIENT" in [x.upper() for x in configDict["dpMode"]]:
    print("noise scale will be for dpClient:")
    eps = configDict["dpOptions"]["epsilon"] / configDict["communication_rounds"]
    delt = configDict["dpOptions"]["delta"] / configDict["communication_rounds"]
    lam = configDict["sgdOptions"]["lambda_"]
    if delt != 0:
      configDict["noiseScale"] = \
        (2 * math.log(1.25 / delt) * math.pow(lam * 2, 2)) / math.pow(eps, 2)
    else:
      configDict["noiseScale"] = (lam * 2) / eps
    configDict["noiseScale"] = configDict["noiseScale"] * \
                              configDict["sgdOptions"]["L"]
  else:
    configDict["noiseScale"] = None

  if not resetClientDirs:
    # check if dirs are there
    clientCount = 0
    for folder in os.scandir(locationfolder):
      if os.path.basename(folder)[0:6] == "client":
        clientCount += 1
        # overwrite config file
        with open(os.path.join(folder, "config.yaml"), 'w') as fp:
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
      with open(os.path.join(locationfolder, "client" + str(curClient), "config.yaml"), 'w') as fp:
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
  for _ in range(num_redos):
    testStartTime = time.time()
    try:
      startID = fc.start(controller_host = "http://localhost:{}".format(port),
                       client_dirs = ','.join(["./client" + str(x) for x in \
                          range(numClients)]),
                       generic_dir = "./",
                       app_image = "fc-private-logistic-regression",
                       channel = 'local',
                       query_interval = 2,
                       download_results = "./output")
    except Exception as err:
      print("ERROR occured: {}".format(str(err)))
      print("restarting controller")
      # restart controller
      _ = os.popen(os.path.join(controllerfolder, "stop_controller.sh"))
      time.sleep(10)
      if int(port) == 8002:
        _ = os.popen(str(os.path.join(controllerfolder, "start_controller_dev.sh")))
      else:
        _ = os.popen(str(os.path.join(controllerfolder, "start_controller.sh")))
      # wait
      time.sleep(15)
      continue
  
    # check if test is done
    time.sleep(10) # wait before checking if test is still running to let
    while True:
      try:
        dfListTests = fc.list(controller_host = "http://localhost:{}".format(port))
      except Exception as err:
        print("ERROR occured: {}".format(str(err)))
        print("restarting controller")
        # restart controller
        _ = os.popen(os.path.join(controllerfolder, "stop_controller.sh"))
        time.sleep(10)
        if int(port) == 8002:
          _ = os.popen(str(os.path.join(controllerfolder, "start_controller_dev.sh")))
        else:
          _ = os.popen(str(os.path.join(controllerfolder, "start_controller.sh")))
        # wait
        time.sleep(15)
        break

      status = dfListTests.loc[int(startID)]["status"].strip()
      if status == "finished":
        break
      elif status == "error" or status == "stopped":
        print("ERROR: Test returned error, Id = {}".format(startID))
        break
      elif time.time() - testStartTime > 900:
        print("ERROR: Test is running for more than 15 mins, Id = {}".format(
          startID))
        try:
          fc.stop(controller_host = "http://localhost:{}".format(port),
                test_id = startID)
        except:
          print("test {} could not be stopped".format(startID))
        break
      time.sleep(5)
  return None



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
  parser.add_argument("-dp", "--dp-mode", choices = ["dpSGD", "dpClient", "noDP"],
                      nargs = 1, dest="dp", default = ["dpSGD"],
                      help = "Where to add DP, either during " + \
                      "stochastic gradient descent, during aggregation or no " \
                      "added DP.", required = True)
  parser.add_argument("-d", "--data", nargs = 1,
                      default = ["iris"], dest = "data",
                      help = "Either \"iris\" to use the iris dataset or " +\
                      "the folder containing data.csv. Default uses " +\
                      "iris dataset,")
  parser.add_argument("-p", "--port-controller", dest="port", nargs = 1,
                      default=[8000], help="Port of the fc-controller, " +\
                        "usually 8000 or 8002")
  parser.add_argument("-o", "--output", required = True, dest = "output",
                      nargs = 1, help = "Folder in which to write results")
  parser.add_argument('--analyse-all', help = 'skips testing and analyses ' +\
                      'everything found in the controllerdata/tests/output',
                     action = "store_true", default = False, dest = "analyse")
  parser.add_argument("-c", "--controller-dir", help = "Directory of the " +\
                      "controller used, needed for restarting of the controller" +\
                      "in case of an error", required = True, nargs = 1,
                      dest = "controller")

  args = parser.parse_args()

  ##### change config_base ####
  # load in dpMode
  config_base["dpMode"] = args.dp
  starttime = time.time()

  # read in data
  datafolder = args.data[0]
  if datafolder == "iris":
    from sklearn import datasets
    dfTotal = datasets.load_iris(as_frame = True).frame
    config_base["labelColumn"] = "target"
  else:
    dfTotal = pd.read_csv(os.path.join(datafolder, "data.csv"))
    config_base["labelColumn"] = args.label[0]

  # change delta accordingly to 1/ databasesize * 5
  databasesize = dfTotal.shape[0]

  config_base["dpOptions"]["delta"] = 1 / (databasesize * 5)

  # read out args

  locationfolder = args.location[0]
  controllerfolder = args.controller[0]
  if os.path.basename(locationfolder) != "data":
    print("ERROR: locationfolder must end with data")
    exit()
  port = args.port[0]

  if not args.analyse:
    print("Starting tests with the following baseline:")
    print(config_base)
    print("numClients: {}".format(numClientsDefault))
    print("______________________________________________________")
    buildoutput = os.popen('sh build.sh')
    print(buildoutput.read())
    print("______________________________________________________")

  # make dir for output
  outputDir = os.path.join(os.getcwd(), args.output[0])
  if not os.path.isdir(outputDir):
    os.mkdir(outputDir)

  if not args.analyse:
    print("Starting TESTING")
    TESTING(dfTotal, locationfolder, port, controllerfolder)
    time.sleep(30) # wait for results to be saved

  outList = list()
  analysisCSVPath = os.path.join(outputDir, "analysis.csv")
  zipResultFolder = os.path.join(locationfolder, "tests", "output")
  # get models into the output folder and read in config files
  for zipout in os.listdir(zipResultFolder):
    m = re.match("^results_test_(\d+)_client_(\d+)_fc_[a-zA-Z]+_(\d+)\.zip$", zipout)
    if m:
      zipout = os.path.join(zipResultFolder, zipout)
      testnum = m.group(1)
      clientnum = m.group(2)
      testID = m.group(3)
      testIdent = str(testnum) + "_" + str(testID)
      #ignore older results
      if not args.analyse and os.path.getmtime(zipout) <= starttime:
        continue
      curModelOutDir = os.path.join(outputDir, "test_{}".format(testIdent))
      if not os.path.isdir(curModelOutDir):
        os.mkdir(curModelOutDir)
      with zipfile.ZipFile(zipout, "r") as zippyfile:
        for outputFile in zippyfile.namelist():
            if outputFile[-4:] == ".pyc":
              zippyfile.extract(outputFile, path=curModelOutDir)
            elif outputFile == "config.yml" or \
                  outputFile == "config.yaml":
              config = yaml.safe_load(zippyfile.read(outputFile))
              config["testnum"] = testIdent
              outList.append(config)

  # write csv
  # remove unnecessary entries from the dicts in outList
  for i in range(len(outList)):
    outList[i].pop("conf_matrix", None)
    outList[i].pop("conf_matrix", None)

    outList[i]["dpMode"] = "|".join(outList[i]["dpMode"])
    outList[i]["foldPrivacytest"] = \
        "|".join([str(x) for x in outList[i]["foldPrivacytest"]])
    outList[i]["foldTest"] = \
        "|".join([str(x) for x in outList[i]["foldTest"]])

    for key, val in outList[i]["dpOptions"].items():
      outList[i][key] = val
    outList[i].pop("dpOptions", None)

    for key, val in outList[i]["sgdOptions"].items():
      outList[i][key] = val
    outList[i].pop("sgdOptions", None)

  # get headers, eliminate errorous runs
  err_ind = list()
  headers = outList[0].keys()

  if os.path.exists(analysisCSVPath):
    with open(analysisCSVPath, "r") as csvOut:
      if csvOut.readline().strip().split(",") != list(headers):
        raise Exception("analysis.csv file contains other header then newly " +\
          "created tests")

  for idx, nextHeader in enumerate(outList[1:]):
    if "status" not in nextHeader or nextHeader["status"] != "finished":
      err_ind.append(idx + 1)
      continue
    if headers != nextHeader.keys():
      print("ERROR: in test: {}".format(nextHeader))
      raise Exception(
            "ERROR, config output files were different over " +\
            "different test runs, redo tests or a csv output cannot be " +\
            "generated")

  if len(err_ind) > 0:
    print("The following test runs did not run correctly:")
    for idx in err_ind:
        print(outList[idx])

  # write header if necessary
  if not os.path.exists(analysisCSVPath):
    with open(analysisCSVPath, "w") as csvOut:
      csvOut.write(','.join(headers) + '\n')

  # write lines
  with open(analysisCSVPath, "a") as csvOut:
    for idx, outDict in enumerate(outList):
      if idx not in err_ind:
        csvOut.write(','.join([str(x) for x in outDict.values()]) + '\n')

