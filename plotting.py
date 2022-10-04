import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("processedAnalysis.csv")
df = df.drop(df[(df["C"].isin([1, "1"])) & (df["max_iter"] == 2500) & \
    (df["lambda_"] == 0.05) & (df["epsilon"] == 30) & \
    (df["numClients"] == 5) & (df["communication_rounds"] == "1_1")].index) #invalid tests
df = df.drop(df[df["alpha"] != 0.1].index) # old tests
df = df.drop(df[df["C"] == '1.2'].index) # used invalid sensitivity

### Difference when using noDP tuned parameters
### also shows numClients impact
### 1 communication round, 1,5,7,10 clients, 0.01 and 0.5 for lambda_, eps = 30
#'''
data = df[(df["lambda_"].isin([0.01])) & (df["numClients"].isin([3,5,7])) \
        & (df["communication_rounds"] == "1_1") & (df["dpMode"].isin(["noDP",
            "dpClient"])) & (df["max_iter"].isin([500]))]
data = data.drop(data[(data["dpMode"] == "dpClient") & (data["epsilon"] != 30)].index)

print(data.groupby(["max_iter", "L", "alpha",
                    "communication_rounds", "epsilon", "tolerance",
                    "lambda_", "dpMode", "numClients"])\
                .agg({"accuracy":["mean", "std"], "tolerance": "count"}))
facetgrid = sns.catplot(x="numClients", y="accuracy", hue="dpMode",
                        kind="box", data=data)
fig = facetgrid.figure
fig.savefig("noDP_vs_dpClient_noDPoptimised.png")
#'''

### Show the impact of com rounds, without clipping
#'''
data = df[(df["lambda_"].isin([0.01])) & (df["numClients"].isin([5])) \
        & (df["dpMode"].isin(["dpClient"])) & (df["max_iter"].isin([500])) \
        & (df["communication_rounds"].isin(["1_1", "20_20"]))]
data = data.drop(data[(data["dpMode"] == "dpClient") & (data["epsilon"] != 30)].index)

print(data.groupby(["max_iter", "L", "alpha",
                    "communication_rounds", "epsilon", "tolerance",
                    "lambda_", "dpMode", "numClients"])\
                .agg({"accuracy":["mean", "std"], "tolerance": "count"}))

facetgrid = sns.catplot(x="communication_rounds", y="accuracy",
                        kind="box", data=data,
                        palette=sns.color_palette(
                            [(1.0, 0.4980392156862745, 0.054901960784313725)]))

facetgrid.set_xticklabels(["20", "1"])
facetgrid.savefig("comRoundImpact.png")
#'''

### show that from comRound 1 to 20, no difference in accuracy
#'''
data = df[(df["lambda_"].isin([0.01])) & (df["numClients"].isin([5])) \
        & (df["dpMode"].isin(["dpClient"])) & (df["max_iter"].isin([500])) \
        & (df["communication_rounds"].isin(["{}_20".format(x) for x in range(1,21)]))]
data = data.drop(data[(data["dpMode"] == "dpClient") & (data["epsilon"] != 30)].index)
data["communication_rounds"] = data["communication_rounds"].map(lambda x: int(x.split("_")[0]))
print(data.groupby(["max_iter", "L", "alpha",
                    "communication_rounds", "epsilon", "tolerance",
                    "lambda_", "dpMode", "numClients"])\
                .agg({"accuracy":["mean", "std"], "tolerance": "count"}))
print(data.groupby(["max_iter", "L", "alpha",
                    "communication_rounds", "epsilon", "tolerance",
                    "lambda_", "dpMode", "numClients"])\
                .agg({"accuracy":["median", "std"], "tolerance": "count"}))

facetgrid = sns.catplot(x="communication_rounds", y="accuracy",
                        kind="box", data=data,
                        palette=sns.color_palette(
                            [(1.0, 0.4980392156862745, 0.054901960784313725)]))

#facetgrid.set_xticklabels(["20", "1"])
facetgrid.savefig("comRoundImpactAll20Rounds.png")

#'''

### show where variance comes from
### 1 communication round, 5 clients, 0.5 for lambda_, eps = 30, all Folds
#'''
data = df[(df["lambda_"].isin([0.01])) & (df["numClients"].isin([5])) \
        & (df["communication_rounds"] == "1_1") & (df["dpMode"].isin(["noDP",
            "dpClient"])) & (df["max_iter"].isin([500]))]
data = data.drop(data[(data["dpMode"] == "dpClient") & (data["epsilon"] != 30)].index)

print(data.groupby(["max_iter",
                    "communication_rounds", "epsilon", "tolerance",
                    "lambda_", "dpMode", "numClients", "foldTest"])\
                .agg({"accuracy":["mean", "std"], "tolerance": "count"}))

facetgrid = sns.catplot(x="foldTest", y="accuracy", hue="dpMode",
                        kind="box", data=data)

facetgrid.set_xlabels('Fold number')
facetgrid.set_xticklabels(["1", "2", "3", "4"])
facetgrid.savefig("foldImpact.png")
#'''



### Show results with clipping
#'''
data = df[(df["lambda_"].isin([0.01])) & (df["numClients"].isin([5])) \
        & (df["dpMode"].isin(["dpClient", "noDP"])) & (df["max_iter"].isin([2500])) \
        & (df["communication_rounds"] == "1_1")]

data = data.drop(data[(data["dpMode"] == "dpClient") & (data["epsilon"] != 30)].index)
data["C"] = data["C"].map(lambda x: float(x))
print(data.groupby(["max_iter", "alpha",
                    "communication_rounds", "epsilon", "tolerance",
                    "lambda_", "dpMode", "numClients", "C"])\
                .agg({"accuracy":["mean", "std"], "tolerance": "count"}))
print(data["C"].unique())
facetgrid = sns.catplot(x="C", y="accuracy",
                        kind="box", data=data)
facetgrid.savefig("clippingImpact.png")
#'''

### Show the impact of comRounds with clipping
#'''
data = df[(df["lambda_"].isin([0.01])) & (df["numClients"].isin([5])) \
        & (df["dpMode"].isin(["dpClient"])) & (df["max_iter"].isin([2500])) \
        & (df["C"] == 0.5)]
data = data.drop(data[(data["dpMode"] == "dpClient") & (data["epsilon"] != 30)].index)
data["communication_rounds"] = data["communication_rounds"].map(lambda x: int(x.split("_")[0]))
print(data.groupby(["max_iter", "L", "alpha",
                    "communication_rounds", "epsilon", "tolerance",
                    "lambda_", "dpMode", "numClients", "C"])\
                .agg({"accuracy":["mean", "std"], "tolerance": "count"}))
print(data.groupby(["max_iter", "L", "alpha",
                    "communication_rounds", "epsilon", "tolerance",
                    "lambda_", "dpMode", "numClients", "C"])\
                .agg({"accuracy":["median", "std"], "tolerance": "count"}))
facetgrid = sns.catplot(x="communication_rounds", y="accuracy",
                        kind="box", data=data,
                        palette=sns.color_palette(
                            [(1.0, 0.4980392156862745, 0.054901960784313725)]))

facetgrid.savefig("comRoundClippingImpact.png")
#'''

### show the impact of epsilon with clipping
#'''
data = df[(df["lambda_"].isin([0.01])) & (df["numClients"].isin([5])) \
        & (df["dpMode"].isin(["dpClient", "noDP"])) & (df["max_iter"].isin([2500])) \
        & (df["communication_rounds"] == "1_1") & (df["C"] == "0.5")]

print(data.groupby(["max_iter", "alpha",
                    "communication_rounds", "epsilon", "tolerance",
                    "lambda_", "dpMode", "numClients", "C"])\
                .agg({"accuracy":["mean", "std"], "tolerance": "count"}))
print(data["C"].unique())
facetgrid = sns.catplot(x="epsilon", y="accuracy",
                        kind="box", data=data)
facetgrid.savefig("clippingImpactNoise.png")
#'''
