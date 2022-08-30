#!/bin/bash
featurecloud controller start
featurecloud app build
featurecloud test start --controller-host=http://localhost:8000 --client-dirs=./client1,./client2 --generic-dir=./ --app-image=fc-private-logistic-regression --channel=local --query-interval=2 --download-results=./output

