# 3. Run the Website (optional)

If you want to replicate the website we provide on https://wheretosubmit.ml, please make sure you have the files `data/data/computer_science/venue_dict` and `data/data/computer_science/word_embeddings.bin`, which are created from the [dataset](../dataset/README.md).
Also you need to have the trained model at `data/models/computer_science/CoMa.model`.

In order to start the website, run the following commands from the *main directory*:

```
docker build -t wts-website:latest -f website/Dockerfile .
docker run --publish 5000:5000 wts-website:latest
```

If successful, the website is available at `localhost:5000`.
