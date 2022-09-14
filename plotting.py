import pandas as pd
dfRaw =  pd.read_csv("analysis.csv")
df = dfRaw.drop(["foldPrivacytest", "noiseScale", "testnum"],
                axis = 1)
print(df.groupby(["dpMode", "epsilon", "numClients", "max_iter", "L", "communication_rounds", "alpha", "lambda_", "tolerance"])\
    .agg({"accuracy":["mean", "std"], "tolerance": "count"}))
