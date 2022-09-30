rm -rf ./output
python3.8 -u test_script.py -cd ../fc-controller-go/data -l target -dp dpClient -d dataIris -p 8002 -o output -c ../fc-controller-go --analyse-all
python3.8 process_results.py
