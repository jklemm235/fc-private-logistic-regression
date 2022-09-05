#!/usr/bin/python3
import pandas as pd
import os


inputFolder = "outputTestScript"
dfAnalysis =  pd.read_csv(os.path.join(inputFolder, "analysis.csv"))
print(dfAnalysis)
