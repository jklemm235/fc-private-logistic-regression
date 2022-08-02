# WIP of DP and SMPC including logistic regression
Usage right now:
the folder data must include:
    config.yaml
    test_set.csv
    trainig_set.csv
Then
    1. featurecloud controller start must be called while within the fc-private-logistic-regression folder
    2. configuration of tests can be done in config.yaml
    3. bash app_build_start.sh must be called to run a test, here under --client-dirs a comma
       seperated list can be given if more than one client should be simulated, client folders
       should be put into data, e.g. data/client_1, data/client_2 and in the command:
       --client-dirs ./client_1,./client_2
    4. Results:
        for each client in data/tests/output/result_test_[testID]_client_[clientNumber][...].zip:
            model_[clienID]_[local_computation_run].pyc
        for the coordinator under data/tests/output/result_test_[testID]_client_0_[...].zip:
            config.yaml, additionally contains numClients value
            aggmodel_[cur_communication_round].pyc
            model_[clienID]_[local_computation_run].pyc (since the coordinator is a client as well)
