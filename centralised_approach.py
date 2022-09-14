import pandas as pd
import os
import algo
import statistics
import numpy as np

numFoldsCrossValidation = 5
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


dfTotal = pd.read_csv(os.path.join("dataIris", "data.csv"))
train_data = dfTotal[60:]
test_data = dfTotal[0:30]

#train_data = pd.read_csv("../fc-controller-go/data/client0/training_set.csv")
#test_data =  pd.read_csv("../fc-controller-go/data/client0/test_set.csv")


config = {"sgdOptions":
                {"alpha":  0.1,
                 "max_iter": 500,
                 "lambda_": 0.01,
                 "tolerance": 0.00001,
                 "L": 1.0},
                "labelColumn": "target",
                "communication_rounds": 1,
                "dpMode": ["noDP"],
                "dpOptions":
                  {"epsilon":  0.1,
                   "delta":  0.01,
                   "C": 1}
                }
print(config)
labelCol = config["labelColumn"]
X_train = np.array(train_data.drop(columns=[labelCol]))
X = np.array(X_train)
y_train = np.array(train_data[labelCol])
X_test = np.array(test_data.drop(columns=[labelCol]))
y_test = np.array(test_data[labelCol])



#"""
results = dict()
for test_data, dfPrivacytest, train_data, _ in fold_generator(dfTotal):
  X_train = np.array(train_data.drop(columns=[labelCol]))
  X = np.array(X_train)
  y_train = np.array(train_data[labelCol])
  X_test = np.array(test_data.drop(columns=[labelCol]))
  y_test = np.array(test_data[labelCol])
  for x in range(25):
    DPSGD_class = algo.LogisticRegression_DPSGD()
    DPSGD_class.DP = False
    DPSGD_class.alpha = config["sgdOptions"]["alpha"]
    DPSGD_class.max_iter = config["sgdOptions"]["max_iter"]
    DPSGD_class.lambda_ = config["sgdOptions"]["lambda_"]
    DPSGD_class.tolerance = config["sgdOptions"]["tolerance"]
    DPSGD_class.L = int(config["sgdOptions"]["L"] * X_train.shape[0])
    DPSGD_class.C = config["dpOptions"]["C"]
    DPSGD_class.epsilon = config["dpOptions"]["epsilon"]
    DPSGD_class.delta = config["dpOptions"]["delta"]
    X_cur, y_train_cur = DPSGD_class.init_theta(X, y_train)
    for _ in range(config["communication_rounds"]):
      DPSGD_class.train(X_cur, y_train_cur)
    acc, confMat = DPSGD_class.evaluate(X = X_test, y = y_test)
    if "bestCase" in results:
      results["bestCase"].append(acc)
    else:
      results["bestCase"] = [acc]
results_processed = dict()
for key, valuelist in results.items():
  results_processed[key] = (sum(valuelist) / len(valuelist), statistics.stdev(valuelist))
print(results_processed)
print(max(results_processed, key=results_processed.get))
print("ACCURACY RESULT:")
print(vars(DPSGD_class))
#"""


#ALPHA
#'''
results = dict()
for test_data, dfPrivacytest, train_data, _ in fold_generator(dfTotal):
  X_train = np.array(train_data.drop(columns=[labelCol]))
  X = np.array(X_train)
  y_train = np.array(train_data[labelCol])
  X_test = np.array(test_data.drop(columns=[labelCol]))
  y_test = np.array(test_data[labelCol])
  for x in range(25):
    for alpha in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
      #print(f"L = {curL}")
      DPSGD_class = algo.LogisticRegression_DPSGD()
      DPSGD_class.DP = False
      DPSGD_class.alpha = alpha
      DPSGD_class.max_iter = config["sgdOptions"]["max_iter"]
      DPSGD_class.lambda_ = config["sgdOptions"]["lambda_"]
      DPSGD_class.tolerance = config["sgdOptions"]["tolerance"]
      DPSGD_class.L = int(config["sgdOptions"]["L"] * X_train.shape[0])
      DPSGD_class.C = config["dpOptions"]["C"]
      DPSGD_class.epsilon = config["dpOptions"]["epsilon"]
      DPSGD_class.delta = config["dpOptions"]["delta"]
      X_cur, y_train_cur = DPSGD_class.init_theta(X, y_train)
      for _ in range(config["communication_rounds"]):
        DPSGD_class.train(X_cur, y_train_cur)
      acc, confMat = DPSGD_class.evaluate(X = X_test, y = y_test)
      if alpha in results:
        results[alpha].append(acc)
      else:
        results[alpha] = [acc]
results_processed = dict()
for key, valuelist in results.items():
  results_processed[key] = (sum(valuelist) / len(valuelist), statistics.stdev(valuelist))
print("SEARCH BEST ALPHA:")
print(results_processed)
#'''
exit()
#MAXIT
#'''
results = dict()
for x in range(100):
  for maxIt in [10, 30, 60, 70, 80, 90, 100, 120, 140, 200, 250, 300, 350, 400, 500, 600, 700]:
    #print(f"L = {curL}")
    DPSGD_class = algo.LogisticRegression_DPSGD()
    DPSGD_class.DP = False
    DPSGD_class.alpha = config["sgdOptions"]["alpha"]
    DPSGD_class.max_iter = maxIt
    DPSGD_class.lambda_ = config["sgdOptions"]["lambda_"]
    DPSGD_class.tolerance = config["sgdOptions"]["tolerance"]
    DPSGD_class.L = int(config["sgdOptions"]["L"] * X_train.shape[0])
    DPSGD_class.C = config["dpOptions"]["C"]
    DPSGD_class.epsilon = config["dpOptions"]["epsilon"]
    DPSGD_class.delta = config["dpOptions"]["delta"]
    X_cur, y_train_cur = DPSGD_class.init_theta(X, y_train)
    for _ in range(config["communication_rounds"]):
      DPSGD_class.train(X_cur, y_train_cur)
    acc, confMat = DPSGD_class.evaluate(X = X_test, y = y_test)
    if maxIt in results:
      results[maxIt].append(acc)
    else:
      results[maxIt] = [acc]
results_processed = dict()
for key, valuelist in results.items():
  results_processed[key] = (sum(valuelist) / len(valuelist), statistics.stdev(valuelist))
print("SEARCH BEST MAXITERATIONS:")
print(results_processed)
print(max(results_processed, key=results_processed.get[0]))
#'''



#L
#'''
results = dict()
for x in range(50):
  for curL in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    DPSGD_class = algo.LogisticRegression_DPSGD()
    DPSGD_class.DP = False
    DPSGD_class.alpha = config["sgdOptions"]["alpha"]
    DPSGD_class.max_iter = config["sgdOptions"]["max_iter"]
    DPSGD_class.lambda_ = config["sgdOptions"]["lambda_"]
    DPSGD_class.tolerance = config["sgdOptions"]["tolerance"]
    DPSGD_class.L = int(curL * X_train.shape[0])
    DPSGD_class.C = config["dpOptions"]["C"]
    DPSGD_class.epsilon = config["dpOptions"]["epsilon"]
    DPSGD_class.delta = config["dpOptions"]["delta"]
    X_cur, y_train_cur = DPSGD_class.init_theta(X, y_train)
    for _ in range(config["communication_rounds"]):
      DPSGD_class.train(X_cur, y_train_cur)
    acc, confMat = DPSGD_class.evaluate(X = X_test, y = y_test)
    if curL in results:
      results[curL].append(acc)
    else:
      results[curL] = [acc]
results_processed = dict()
for key, valuelist in results.items():
  results_processed[key] = sum(valuelist) / len(valuelist)
print("SEARCH BEST L:")
print(results_processed)
print(max(results_processed, key=results_processed.get))
#'''

#LAMBDA_
results = dict()
for x in range(50):
  for lambda_ in [10e-3, 10e-2, 10e-1]:
    #print(f"L = {curL}")
    DPSGD_class = algo.LogisticRegression_DPSGD()
    DPSGD_class.DP = False
    DPSGD_class.alpha = config["sgdOptions"]["alpha"]
    DPSGD_class.max_iter = config["sgdOptions"]["max_iter"]
    DPSGD_class.lambda_ = lambda_ #config["sgdOptions"]["lambda_"]
    DPSGD_class.tolerance = config["sgdOptions"]["tolerance"]
    DPSGD_class.L = int(config["sgdOptions"]["L"] * X_train.shape[0])
    DPSGD_class.C = config["dpOptions"]["C"]
    DPSGD_class.epsilon = config["dpOptions"]["epsilon"]
    DPSGD_class.delta = config["dpOptions"]["delta"]
    X_cur, y_train_cur = DPSGD_class.init_theta(X, y_train)
    for _ in range(config["communication_rounds"]):
      DPSGD_class.train(X_cur, y_train_cur)
    acc, confMat = DPSGD_class.evaluate(X = X_test, y = y_test)
    if lambda_ in results:
      results[lambda_].append(acc)
    else:
      results[lambda_] = [acc]
results_processed = dict()
for key, valuelist in results.items():
  results_processed[key] = sum(valuelist) / len(valuelist)
print("SEARCH BEST LAMBDA:")
print(results_processed)
print(max(results_processed, key=results_processed.get))
# {1e-05: 0.9880000000000004, 0.0001: 0.9900000000000001, 0.001: 0.9883333333333334, 0.01: 0.9780000000000001, 0.1: 0.7356666666666672, 1.0: 0.6228333333333346}
# 0.0001


#Results:
'''
{'sgdOptions': {'alpha': 0.1, 'max_iter': 500, 'lambda_': 0.01, 'tolerance': 1e-05, 'L': 0.2}, 'labelColumn': 'target', 'communication_rounds': 1, 'dpMode': ['noDP'], 'dpOptions': {'epsilon': 0.1, 'delta': 0.01, 'C': 1}}
SEARCH BEST MAXITERATIONS:
{10: 0.6073333333333336, 30: 0.680333333333334, 60: 0.7143333333333337, 70: 0.7396666666666669, 80: 0.7496666666666674, 90: 0.7596666666666668, 100: 0.7703333333333339, 120: 0.7633333333333335, 140: 0.7993333333333337, 200: 0.8436666666666666, 250: 0.8579999999999999, 300: 0.8939999999999999, 350: 0.8899999999999997, 400: 0.9139999999999994, 500: 0.9223333333333325, 600: 0.9299999999999997, 700: 0.9343333333333331}
700

SEARCH BEST L:
{0.1: 0.8680000000000001, 0.2: 0.9273333333333337, 0.3: 0.958, 0.4: 0.9626666666666663, 0.5: 0.9653333333333335, 0.6: 0.9713333333333334, 0.7: 0.976, 0.8: 0.9953333333333333, 0.9: 0.9973333333333333, 1.0: 1.0}
1.0
SEARCH BEST LAMBDA:
{0.01: 0.9180000000000001, 0.1: 0.7446666666666668, 1.0: 0.5286666666666666}
0.01

Final Parameters should be:
{'sgdOptions': {'alpha': 0.1, 'max_iter': 500, 'lambda_': 0.01, 'tolerance': 1e-05, 'L': 1.0}, 'labelColumn': 'target', 'communication_rounds': 1, 'dpMode': ['noDP'], 'dpOptions': {'epsilon': 0.1, 'delta': 0.01, 'C': 1}}


{'sgdOptions': {'alpha': 0.1, 'max_iter': 500, 'lambda_': 0.01, 'tolerance': 1e-05, 'L': 1.0}, 'labelColumn': 'target', 'communication_rounds': 1, 'dpMode': ['noDP'], 'dpOptions': {'epsilon': 0.1, 'delta': 0.01, 'C': 1}}
{'bestCase': (0.9666666666666671, 0.02368896848395671)}
bestCase
ACCURACY RESULT:
{'alpha': 0.1, 'max_iter': 500, 'lambda_': 0.01, 'tolerance': 1e-05, 'DP': False, 'L': 90, 'C': 1, 'epsilon': 0.1, 'delta': 0.01, 'theta': array([[ 0.86151277,  0.91374272,  0.04388135],
       [ 1.1448114 ,  0.99449418, -0.32016875],
       [ 1.9433187 ,  0.33952976, -0.46371162],
       [-1.22302622,  0.7163912 ,  2.32577186],
       [-0.22670001, -0.03206092,  2.07789776]]), 'pred_func': <function softmax at 0x7f042410e560>}

SEARCH BEST ALPHA:
{0.001: (0.6916666666666664, 0.027777777777777773), 0.005: (0.808333333333333, 0.049549032500880295), 0.01: (0.9083333333333339, 0.027777777777777773), 0.05: (0.9666666666666671, 0.02368896848395671), 0.1: (0.9666666666666671, 0.02368896848395671), 0.5: (0.7383333333333333, 0.15658927811380327)}


'''
