import json
import re
import os
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from joblib import cpu_count

ITERATIONS = 20
NUM_CPUS = min(cpu_count(), 16)
MINIMUM_TOKEN_OCCURRENCES = 200
VECTOR_SIZE = 200
WINDOW = 5
SAMPLE = 1e-5


class StreamDataset(object):
    def __init__(self, data_path: str):
        self.file_reader = open(data_path, "r", encoding="utf-8")

    def __iter__(self):
        for line in self.file_reader:
            line = json.loads(line)
            tmp = []
            for val in ["abstract", "title", "keywords"]:
                if val in line:
                    if val == "keywords":
                        tmp.extend(re.split('[^\w-]+', " ".join(line[val]).lower()))
                    else:
                        tmp.extend(re.split('[^\w-]+', line[val].lower()))
            yield tmp


def generate(data_path: str, vector_file: str) -> None:
    """
    :param data_path:
    :param vector_file:
    :return:
    """
    print("Training the word2vec embeddings from", data_path)
    model = Word2Vec(sentences=StreamDataset(data_path), size=VECTOR_SIZE, window=WINDOW,
                     min_count=MINIMUM_TOKEN_OCCURRENCES, workers=NUM_CPUS, iter=ITERATIONS, sample=SAMPLE)
    model.wv.save_word2vec_format(vector_file, binary=False)
    print("done")


def generate_venue_dict(path_to_data: str, output_path: str) -> None:
    y_values = set()
    with open(path_to_data, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            y_values.update([line['venue'].lower()])
    ret_dict = Dictionary([list(y_values)])
    ret_dict.save(output_path)


if __name__ == '__main__':
    PATH = "./data/data/"
    for ds in ['medline', 'computer_science']:
        print("Running W2V and target class dictionary creation for", ds, ". ")
        generate(os.path.join(PATH, ds, ds + "_reduced.json"), os.path.join(PATH, ds, "word_embeddings.bin"))
        generate_venue_dict(os.path.join(PATH, ds, ds + "_reduced.json"), os.path.join(PATH, ds, "venue_dict"))
