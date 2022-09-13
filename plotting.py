import pandas as pd
dfRaw =  pd.read_csv("analysis.csv")
df = dfRaw.drop(["foldPrivacytest", "noiseScale", "testnum"],
                axis = 1)

print(df.groupby(["numClients", "max_iter", "L", "communication_rounds"]).mean())
