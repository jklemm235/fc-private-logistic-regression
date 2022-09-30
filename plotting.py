import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("processedAnalysis.csv")

'''
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 120):  # more options can be specified also
    print(df.groupby(["dpMode", "max_iter", "epsilon", "L", "lambda_", "numClients",
        "communication_rounds", "tolerance"])\
                .agg({"accuracy":["mean", "std"], "tolerance": "count"}))
'''
#TODO: next part only temp

data = df[(df["lambda_"].isin([0.05, 0.1])) & (df["numClients"].isin([5])) \
        & (df["communication_rounds"] == 1) & (df["dpMode"].isin(["noDP",
            "dpClient"])) & (df["max_iter"] >= 500)]
data = data.drop(data[(data["dpMode"] == "dpClient") & (data["epsilon"] != 30)].index)

print(data.groupby(["max_iter", "L", "alpha",
                    "communication_rounds", "epsilon", "tolerance",
                    "lambda_", "dpMode", "numClients"])\
                .agg({"accuracy":["mean", "std"], "tolerance": "count"}))
exit()
#TODO: changing of labels, maybe of ticks, maybe add titles -> make everything prettier
### Show that behaviour is worse, lambda can improve the behaviour
### also shows numClients impact
### 1 communication round, 1,5,7,10 clients, 0.01 and 0.5 for lambda_, eps = 30
'''
data = df[(df["lambda_"].isin([0.01, 0.5])) & (df["numClients"].isin([1,3,5,7,10])) \
        & (df["communication_rounds"] == 1) & (df["dpMode"].isin(["noDP",
            "dpClient"])) & (df["max_iter"] == 500)]
data = data.drop(data[(data["dpMode"] == "dpClient") & (data["epsilon"] != 30)].index)

print(data.groupby(["max_iter", "L", "alpha",
                    "communication_rounds", "epsilon", "tolerance",
                    "lambda_", "dpMode", "numClients"])\
                .agg({"accuracy":["mean", "std"], "tolerance": "count"}))
facetgrid = sns.catplot(x="numClients", y="accuracy", hue="dpMode",
                        kind="box", col="lambda_", data=data)
fig = facetgrid.figure
fig.savefig("noDP_vs_dpClient_lowHigh_Lambda.png")
'''

### show where variance comes from
### 1 communication round, 5 clients, 0.5 for lambda_, eps = 30, all Folds
'''
data = df[(df["lambda_"].isin([0.5])) & (df["numClients"].isin([5])) \
        & (df["communication_rounds"] == 1) & (df["dpMode"].isin(["noDP",
            "dpClient"])) & (df["max_iter"].isin([500, 2500]))]
data = data.drop(data[(data["dpMode"] == "dpClient") & (data["epsilon"] != 30)].index)

print(data.groupby(["max_iter", "L", "alpha",
                    "communication_rounds", "epsilon", "tolerance",
                    "lambda_", "dpMode", "numClients", "foldTest"])\
                .agg({"accuracy":["mean", "std"], "tolerance": "count"}))

facetgrid = sns.catplot(x="foldTest", y="accuracy", hue="dpMode",
                        kind="box", col="max_iter", data=data,
                        palette=sns.color_palette(
                            [(1.0, 0.4980392156862745, 0.054901960784313725),
                             (0.12156862745098039, 0.4666666666666667, 0.7058823529411765) ]))

facetgrid.set_xlabels('Fold number')
facetgrid.set_xticklabels(["1", "2", "3", "4"])
facetgrid.savefig("foldImpact.png")
'''


### Show the impact of com rounds
#TODO: change to single colour
#'''
data = df[(df["lambda_"].isin([0.5])) & (df["numClients"].isin([5])) \
        & (df["dpMode"].isin(["dpClient"])) & (df["max_iter"].isin([500]))]
data = data.drop(data[(data["dpMode"] == "dpClient") & (data["epsilon"] != 30)].index)

print(data.groupby(["max_iter", "L", "alpha",
                    "communication_rounds", "epsilon", "tolerance",
                    "lambda_", "dpMode", "numClients"])\
                .agg({"accuracy":["mean", "std"], "tolerance": "count"}))

facetgrid = sns.catplot(x="communication_rounds", y="accuracy",
                        kind="box", data=data,
                        palette=sns.color_palette(
                            [(1.0, 0.4980392156862745, 0.054901960784313725)]))

facetgrid.savefig("comRoundImpact.png")
#'''

### Show the impact of epsilon TODO

