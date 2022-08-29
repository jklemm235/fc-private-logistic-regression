#!/usr/bin/python3
import zipfile
import re
import yaml
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="analyse script for logistic regression, " + \
                "analyses produced output from test_script")

    parser.add_argument("-i", "--input", dest="inputDir",
                    nargs = 1,
                    help = "The folder of output from the test_script to " +\
                    "be analyzed, should be ./data/tests/output",
                    required = True)
    parser.add_argument("-o", "--output", dest="outputDir",
                      nargs = 1,
                      help = "Location where the output of the analysis, " +\
                      "default is outputTestScript",
                      default = "outputTestScript")
    args = parser.parse_args()

    # inputdir
    inputDir = args.inputDir[0]
    if not os.path.isabs(inputDir):
        inputDir = os.path.join(os.getcwd(), args.inputDir[0])

    # outpudir and create csv
    outputDir = args.outputDir[0]
    if not os.path.isabs(outputDir):
        outputDir = os.path.join(os.getcwd(), args.outputDir[0])
    outList = list()

    # get models into the output folder and read in config files
    for zipout in os.listdir(inputDir):
        m = re.match("^results_test_(\d+)_client_(\d+)_.*\.zip$", zipout)
        if m:
            zipout = os.path.join(inputDir, zipout)
            testnum = m.group(1)
            curModelOutDir = os.path.join(outputDir,
                                          f"models_testnum_{testnum}")
            if not os.path.isdir(curModelOutDir):
                os.mkdir(curModelOutDir)
            clientnum = m.group(2)
            with zipfile.ZipFile(zipout, "r") as zippyfile:
                for outputFile in zippyfile.namelist():
                    if outputFile[-4:] == ".pyc":
                        zippyfile.extract(outputFile, path=curModelOutDir)
                    elif outputFile == "config.yml" or \
                         outputFile == "config.yaml":
                        config = yaml.safe_load(zippyfile.read(outputFile))
                        config["testnum"] = testnum
                        outList.append(config)

    # write csv

    # remove unnecessary entries from the dicts in outList
    for i in range(len(outList)):
        outList[i].pop("conf_matrix", None)
        outList[i].pop("conf_matrix", None)

        outList[i]["dpMode"] = "|".join(outList[i]["dpMode"])
        outList[i]["foldPrivacytest"] = \
            "|".join([str(x) for x in outList[i]["foldPrivacytest"]])
        outList[i]["foldTest"] = \
            "|".join([str(x) for x in outList[i]["foldTest"]])

        for key, val in outList[i]["dpOptions"].items():
            outList[i][key] = val
        outList[i].pop("dpOptions", None)

        for key, val in outList[i]["sgdOptions"].items():
            outList[i][key] = val
        outList[i].pop("sgdOptions", None)

    # get headers
    headers = outList[0].keys()
    for nextHeader in outList[1:]:
        nextHeader = nextHeader.keys()
        if headers != nextHeader:
            raise Exception("ERROR, config output files were different over " +\
                "different test runs, redo tests or a csv output cannot be " +\
                "generated")

    # write lines
    with open(os.path.join(outputDir, "analysis.csv"), "w") as csvOut:
        csvOut.write(','.join(headers) + '\n')
        for outDict in outList:
            csvOut.write(','.join([str(x) for x in outDict.values()]) + '\n')








