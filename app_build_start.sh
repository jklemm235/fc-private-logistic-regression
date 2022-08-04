#!/bin/bash
featurecloud app build
featurecloud test start --controller-host=http://localhost:8002 --client-dirs=./client_1,./client_2 --generic-dir=./ --app-image=fc-private-logistic-regression --channel=local --query-interval=2 --download-results=./output
