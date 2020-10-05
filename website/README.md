# Running the website

In order to run the website, the following prerequisites must be met.

- Having docker
- Created the files `data/data/computer_science/venue_dict` & `data/data/computer_science/word_embeddings.bin` 
(for further information, see the dataset section)
- Having a model at `data/models/computer_science/CoMa.model`

In order to run the website, use the following commands: 
```
docker build -t wts-website:latest -f website/Dockerfile .
docker run --publish 5000:5000 wts-website:latest
```
 If everything works, the website should be available at `localhost:5000`