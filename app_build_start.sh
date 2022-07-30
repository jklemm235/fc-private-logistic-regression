#!/bin/bash
featurecloud app build
featurecloud test start --controller-host=http://localhost:8000 --client-dirs=./ --generic-dir=./ --app-image=fc-private-logistic-regression --channel=local --query-interval=2 --download-results=./output
