import pandas as pd
import os
import numpy as np
import algo

"""
dfRaw =  pd.read_csv("analysis.csv")
df = dfRaw.drop(["foldPrivacytest", "noiseScale", "testnum"],
                axis = 1)
index_names = df[(df['alpha'] != 0.1) | (df["lambda_"] != 0.01) | (df["tolerance"] != 0.00001) | (df["L"] != 1.0) ].index
df.drop(index_names, inplace=True)
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 120):  # more options can be specified also
    print(df.groupby(["dpMode", "epsilon", "numClients", "max_iter", "L", 
        "communication_rounds", "tolerance"])\
                .agg({"accuracy":["mean", "std"], "tolerance": "count"}))
"""
#Input:
baseFolder = "output"
labelCol = "target"
df = pd.read_csv(os.path.join(baseFolder, "analysis.csv"))
dataDf = pd.read_csv(os.path.join("dataIris", "data.csv"))


# extend dataframe and redo accuracy with the correct data
# drop some old tests
index_names = df[(df['alpha'] != 0.1) | (df["tolerance"] != 0.00001) | (df["L"] != 1.0) ].index
df.drop(index_names, inplace=True)
DPSGD_class = algo.LogisticRegression_DPSGD()
dictList = df.to_dict(orient = "records")
dictListNewRows = list()
for idx, baseRow in enumerate(dictList):
    numComRounds = baseRow["communication_rounds"]
    testnum = baseRow["testnum"]
    # Add com Rounds
    model_folder = os.path.join(baseFolder, "test_{}".format(testnum))
    foldInfo = baseRow["foldTest"].split('|')
    test_data = dataDf.iloc[int(foldInfo[0]):int(foldInfo[1]) + 1]
    x_test = np.array(test_data.drop(columns=[labelCol]))
    y = np.array(test_data[labelCol])
    for curComRound in range(1, numComRounds):
        newRow = baseRow.copy()
        # get corresponding model for curComRound
        model = np.load(os.path.join(model_folder,
                                     "aggmodel_{}.pyc".format(curComRound)))
        DPSGD_class.theta = model
        newRow["accuracy"], _ = DPSGD_class.evaluate(x_test, y)
        newRow["communication_rounds"] = curComRound
        dictListNewRows.append(newRow)
    # correction of base original Row
    model = np.load(os.path.join(model_folder, "trained_model.pyc"))
    DPSGD_class.theta = model
    dictList[idx]["accuracy"], _ = DPSGD_class.evaluate(x_test, y)

# concat dictList and dictListNewRows
dictList.extend(dictListNewRows)
df = pd.DataFrame(dictList)

# save df
df.to_csv("processedAnalysis.csv")


# analyse part
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 120):  # more options can be specified also
    print(df.groupby(["dpMode", "epsilon", "numClients", "max_iter", "L",
        "communication_rounds", "tolerance"])\
                .agg({"accuracy":["mean", "std"], "tolerance": "count"}))
