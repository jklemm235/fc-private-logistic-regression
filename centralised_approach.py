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
                 "max_iter": 100,
                 "lambda_": 10e-6,
                 "tolerance": 10e-6,
                 "L": 50},
                "labelColumn": "target",
                "communication_rounds": 1, #TODO change back to 10?
                "dpMode": ["noDP"],
                "dpOptions":
                  {"epsilon":  0.1,
                   "delta":  0.01,
                   "C": 1}
                }

labelCol = config["labelColumn"]
X_train = np.array(train_data.drop(columns=[labelCol]))
X = np.array(X_train)
y_train = np.array(train_data[labelCol])

X_test = np.array(test_data.drop(columns=[labelCol]))
y_test = np.array(test_data[labelCol])



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
    DPSGD_class.L = 15 # seems like the best option
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






results = dict()
for x in range(100):
  for curL in [1, 3, 8, 10, 12, 14, 15, 18, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90]:
    #print(f"L = {curL}")
    DPSGD_class = algo.LogisticRegression_DPSGD()
    DPSGD_class.DP = False
    DPSGD_class.alpha = config["sgdOptions"]["alpha"]
    DPSGD_class.max_iter = config["sgdOptions"]["max_iter"]
    DPSGD_class.lambda_ = config["sgdOptions"]["lambda_"]
    DPSGD_class.tolerance = config["sgdOptions"]["tolerance"]
    DPSGD_class.L = curL #config["sgdOptions"]["L"]
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
