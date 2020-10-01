# Conference Matcher

1. To start, create a virtual environment in `website`: `python3 -m venv venv`.
2. Activate it: `source venv/bin/activate`.
3. Install requirements: `pip3 install -r requirements.txt`.
4. You need the correct word embeddings and output vectors. They are located on vingilot:
    - `/scratch/regio/{computer_science/medline}/word_embeddings.bin`
    - `/scratch/regio/{computer_science/medline}/venue_dict`
5. Copy them to the corresponding folder: `../data/{computer_science/medline}/` 
6. Copy the Model weights. 
    - The Job ID is written in `../website_{medline/computer_science}_config.json`
    - `scp -r vingilot:/scratch/regio/Models/{Job_ID}/ ./data/Models/`
6. Then run `./run.sh`