import pandas as pd
import os
import numpy as np

df = pd.read_csv("processedAnalysis.csv")
index_names = df[(df['alpha'] != 0.1) | (df["tolerance"] != 0.00001) | (df["L"] != 1.0) | (df["communication_rounds"] != 1)].index #TODO: remove restriction of com_rounds
df.drop(index_names, inplace=True)

with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 120):  # more options can be specified also
    print(df.groupby(["dpMode", "max_iter", "epsilon", "L", "lambda_", "numClients",
        "communication_rounds", "tolerance"])\
                .agg({"accuracy":["mean", "std"], "tolerance": "count"}))
