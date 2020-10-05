# Retraining the model or testing it 

This section allows you to retrain the model or run the test on the existing model. The following prerequisites must be met.

- If best, have a GPU available. If not, please change the `device` from `cuda` to `cpu` in the config file (ca be found at `data/config.json`)
- Having docker
- Created all the dataset files (for further information, see the dataset section)
- If only want to test the model, it should be available at `data/models/computer_science/CoMa.model`

In order to run the website, use the following commands: 
```
docker build -t wts-train:latest -f model/Dockerfile .
docker run wts-website:latest
```

## Running the Medline dataset

In order to use the second dataset, please go to the `model/Dockerfile` and change the following lines:
```
ENTRYPOINT python3 -u model/WTS_Train.py 0
```  
to 
```
ENTRYPOINT python3 -u model/WTS_Train.py 1
```