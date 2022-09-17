import pandas as pd
dfRaw =  pd.read_csv("analysis.csv")
df = dfRaw.drop(["foldPrivacytest", "noiseScale", "testnum"],
                axis = 1)
index_names = df[(df['alpha'] != 0.1) | (df["lambda_"] != 0.01) | (df["tolerance"] != 0.00001) | (df["L"] != 1.0) ].index
df.drop(index_names, inplace=True)
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 120):  # more options can be specified also
    print(df.groupby(["dpMode", "epsilon", "numClients", "max_iter", "L", 
        "communication_rounds", "tolerance"])\
                .agg({"accuracy":["mean", "std"], "tolerance": "count"}))
