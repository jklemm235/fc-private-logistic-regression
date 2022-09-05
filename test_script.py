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


##### Starting point of any test done #####
# NOTE: DO NOT USE sgdOptions.L = 1 if you want to consider ALL data,
# all values under 1 get seen as percentage, but 1 gets seen as just use one
# sample!!
#
# labelColumn is set via commandline
# delta is set in main as 1 / (databasesize*10)
config_base = {"sgdOptions":
                {"alpha":  0.001,
                 "max_iter": 100,
                 "lambda_": 10e-4,
                 "tolerance": 10e-6,
                 "L": 0.2},
                "labelColumn": None,
                "communication_rounds": 5,
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
  testnumsList = list() # needed in analysis, fill with results of run_test
  for dfTest, dfPrivacytest, dfTrain, foldInfoDict in fold_generator(dfTotal):
    #TODO: remove testnumlist
    curFold += 1
    config_base.update(foldInfoDict)

    print(f"Starting Fold {curFold}")
    # TEST number clients
    print("Running test number clients:")
    # Warning, for iris, don't use more than 12 clients, the dataset is too
    # small
    testNumClients = [1, 2, 4, 6, 8, 10]
    print("numClients checked = {}".format(testNumClients))
    config_running = config_base.copy()
    for numClients in testNumClients:
        testnumsList = testnumsList + run_test(config_running, dfTrain, dfTest,
                              locationfolder, port, numClients = numClients,
                              dataDistribution = None, resetClientDirs = True)
        print(testnumsList)
    continue # first batch of tests

    print("______________________________________________________")
    # TEST number aggregation rounds (communication_rounds)
    print("Running test number communication_rounds:")
    testComRounds = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    print("com_rounds checked = {}".format(testComRounds))
    for com_rounds in testComRounds:
      config_running = config_base.copy()
      config_running["communication_rounds"] = com_rounds
      testnumsList = testnumsList + run_test(config_running, dfTrain, dfTest,
                    locationfolder, port, numClients = numClientsDefault,
                    dataDistribution = None)
    print("______________________________________________________")


    # TEST sample size (L)
    print("Running test samplingRatio:")
    testSamplingRatio = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5,
                         0.6, 0.7, 0.8, dfTrain.shape[0]]
    print("samplingRatio checked = {}".format(testSamplingRatio))
    for samplingRatio in testSamplingRatio:
      config_running = config_base.copy()
      config_running["sgdOptions"]["L"] = samplingRatio
      testnumsList = testnumsList + run_test(config_running, dfTrain, dfTest,
              locationfolder, port, numClients = numClientsDefault,
              dataDistribution = None)
    print("______________________________________________________")


    # TEST epsilon, using gauss noise
    if not "NODP" in [x.upper() for x in config_base["dpMode"]]:
      # Just run tests with any dp mode activated, as without it, it doesnt
      # make any difference to change epsilon
      print("Running test epsilon")
      testEpsilon = [0.001, 0.01, 0.1, 0.2, 0.4, 0.8]
      print("epsilon checked = {}".format(testEpsilon))
      for epsilon in testEpsilon:
        config_running = config_base.copy()
        config_running["dpOptions"]["epsilon"] = epsilon
        testnumsList = testnumsList + run_test(config_running, dfTrain, dfTest,
                locationfolder, port, numClients = numClientsDefault,
                dataDistribution = None)
      print("______________________________________________________")

  return testnumsList


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
             dataDistribution = None, resetClientDirs = False,
             num_redos = 5):
  #TODO: remove testnumList
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
  testnumsList = list()
  for _ in range(num_redos):
    startoutput = os.popen("featurecloud test start " +\
        "--controller-host=http://localhost:{} ".format(port) +\
        "--client-dirs={} ".format(','.join(["./client" + str(x) for x in \
          range(numClients)])) +\
        "--generic-dir=./ --app-image=fc-private-logistic-regression " +\
        "--channel=local --query-interval=2 --download-results=./output")
    m = re.match("^Test id=(\d+) started$", startoutput.read())
    if not m:
      raise Exception("A test could not be started, ERROR")
    testnumsList.append(int(m.group(1)))
    # check if test is done
    time.sleep(5) # wait before checking if test is still running to let
    while True:
      listoutput = os.popen("featurecloud test list " +\
        "--controller-host=http://localhost:{} ".format(port)).read()
      if not "running" in listoutput and not "init" in listoutput:
        break
      time.sleep(5)
  print(testnumsList)
  return testnumsList



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

  # change delta accordingly to 1/ databasesize * 5
  databasesize = dfTotal.shape[0]

  config_base["dpOptions"]["delta"] = 1 / (databasesize * 5)

  # read out args

  locationfolder = args.location[0]
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
    testnumsList = TESTING(dfTotal, locationfolder, port)
    print(testnumsList) #TODO rmv this print
    # check if any errors and report
    listoutput = os.popen("featurecloud test list " +\
        "--controller-host=http://localhost:{} ".format(port)).read()
    if "error" in listoutput:
      print("WARNING: The controller session seems to contain at least one " +\
            "ERROR, check the logs and run featurecloud test list to see more")

    time.sleep(30) # wait for results to be saved

  # get models into the output folder and read in config files
  for zipout in os.listdir(locationfolder):
    m = re.match("^results_test_(\d+)_client_(\d+)_[a-zA-Z]+_(\d+)\.zip$", zipout)
    if m:
      zipout = os.path.join(locationfolder, zipout)
      testnum = m.group(1)
      clientnum = m.group(2)
      testID = m.group(3)
      #TODO implement getting of only test specific data via
      # os.path.getmtime(path)
      # and if not current, continue
      if not os.path.isdir(curModelOutDir):
        os.mkdir(curModelOutDir)
      clientnum = m.group(2)
      with zipfile.ZipFile(zipout, "r") as zippyfile:
        for outputFile in zippyfile.namelist():
            if outputFile[-4:] == ".pyc":
              zippyfile.extract(outputFile, path=curModelOutDir)
            elif outputFile == "config.yml" or \
                  outputFile == "config.yaml":
              config = yaml.safe_load(zippyfile.read(outputFile))
              config["testnum"] = testnum
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
  #TODO check if headers same as cur analysis file
  if dfAnalysis:
    print(dfAnalysis.columns)
    if dfAnalysis.columns != headers:
      exit() #TODO: change this to raise

  for idx, nextHeader in enumerate(outList[1:]):
    if "status" not in nextHeader or nextHeader["status"] != "finished":
      err_ind.append(idx + 1)
      continue
    nextHeader = nextHeader.keys()
    if headers != nextHeader:
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

