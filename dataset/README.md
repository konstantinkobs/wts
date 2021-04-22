# 1. Create the Dataset

Inside the file ``dataset/generate_dataset.py``, set the variable ``DOWNLOAD_SEMANTIC_SCHOLAR`` to `True` if you want to download the Semantic Scholar dataset. Be aware that the size of the dataset is **108 GB**. If you set it to `False`, the dataset is expected to be present in `data/semantic_scholar/`.

Then, to prepare the dataset, run the following commands from the *main directory*:

```
docker build -t wts-dataset:latest -f dataset/Dockerfile .
docker run wts-dataset:latest
```

After a successful run, the following data should be available. `DATASET` represents either `computer_science` or `medline`:

- `data/data/DATASET/venue_dict`: Contains the list of conferences. Needed to train and run the model/website.
- `data/data/DATASET/word_embeddings.bin`: Contains the word embeddings for the respective dataset
- `data/data/DATASET/DATASET_reduces.json`: Final version of the respective dataset
- `data/data/DATASET/DATASET_train.json`: Training split for the respective dataset
- `data/data/DATASET/DATASET_test.json`: Test split for the respective dataset
- `data/data/DATASET/DATASET_val.json`: Validation split for the respective dataset

Next, you can [Train and Test the model](../model/README.md).
