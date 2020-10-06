# 2. Train and Test the Model

Before running the model, please make sure that you [downloaded and preprocessed the dataset](../dataset/README.md).
Also, if you do not have a GPU, change the `device` from `cuda` to `cpu` in `data/config.json`.

To train the model on the Computer Science data, make sure to have the following line in `model/Dockerfile`: `ENTRYPOINT python3 -u model/WTS_Train.py 0`

To use the medicine dataset, change the line to `ENTRYPOINT python3 -u model/WTS_Train.py 1`.

Then, to train and test the model, run the following commands from the *main directory*:

```
docker build -t wts-train:latest -f model/Dockerfile .
docker run wts-website:latest
```

After a successful run, you should have a trained model in `data/models/DATASET/CoMa.model`, where `DATASET` is either `computer_science` or `medline`.

## Only Testing the Model

If you only want to test the model, run the same commands as above but have the model available at `data/models/DATASET/CoMa.model`, where `DATASET` is either `computer_science` or `medline`.
