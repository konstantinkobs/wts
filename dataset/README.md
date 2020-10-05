# Creating the Dataset

In order to create the dataset, the following prerequisites must be met.
 
- Having docker
- At best, loads of free space. Care, the semantic scholar dataset needs to be downloaded. It may take up to *100* GB of memory. 

## Run it 
In order to run the website, use the following commands: 
```
docker build -t wts-dataset:latest -f dataset/Dockerfile .
docker wts-dataset:latest
```

## Having downloaded the semantic scholar already 

If you have downloaded the semantic scholar dataset, please change the variable `DOWNLOAD_SEMANTIC_SCHOLAR` to `True`. 
Please also double check the path to the dataset. 

## Create data

When finishing successfully, the following data should be available. 
`dataset` represents either `computer_science` or `medline`:
- `data/data/dataset/venue_dict`: Contains the list of conferences. Needed to train and run the model/website. 
- `data/data/dataset/word_embeddings.bin`: Contains the word embeddings for the respective dataset 
- `data/data/dataset/dataset_reduces.json`: Final version of the respective dataset
- `data/data/dataset/dataset_train.json`: Training split for the respective dataset
- `data/data/dataset/dataset_test.json`: Test split for the respective dataset
- `data/data/dataset/dataset_val.json`: Validation split for the respective dataset 
