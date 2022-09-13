import pandas as pd
import os
import algo
import numpy as np

dfTotal = pd.read_csv(os.path.join("dataIris", "data.csv"))
train_data = dfTotal[60:]
test_data = dfTotal[0:30]

#train_data = pd.read_csv("../fc-controller-go/data/client0/training_set.csv")
#test_data =  pd.read_csv("../fc-controller-go/data/client0/test_set.csv")


config = {"sgdOptions":
                {"alpha":  0.1,
                 "max_iter": 500,
                 "lambda_": 10e-2,
                 "tolerance": 10e-6,
                 "L": 0.4},
                "labelColumn": "target",
                "communication_rounds": 1, #TODO change back to 10?
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
for x in range(5):
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
  DPSGD_class.train(X_cur, y_train_cur)
  acc, confMat = DPSGD_class.evaluate(X = X_test, y = y_test)
  print(type(acc))
  print(acc)
  print(acc.item())
  if "bestCase" in results:
    results["bestCase"].append(acc)
  else:
    results["bestCase"] = [acc]
results_processed = dict()
for key, valuelist in results.items():
  results_processed[key] = sum(valuelist) / len(valuelist)
print("ACCURACY RESULT:")
print(vars(DPSGD_class))
print(results_processed)
exit()
#"""





#MAXIT
'''
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
    DPSGD_class.train(X_cur, y_train_cur)
    acc, confMat = DPSGD_class.evaluate(X = X_test, y = y_test)
    if maxIt in results:
      results[maxIt].append(acc)
    else:
      results[maxIt] = [acc]
results_processed = dict()
for key, valuelist in results.items():
  results_processed[key] = sum(valuelist) / len(valuelist)
print("SEARCH BEST MAXITERATIONS:")
print(results_processed)
print(max(results_processed, key=results_processed.get))
'''



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
