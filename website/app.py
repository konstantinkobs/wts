import sys
import os
sys.path.append(os.getcwd())
import datetime
import json
import torch

from gensim.corpora import Dictionary
from torchtext.vocab import Vectors
from util.CoMa_Model import CoMaModel
from website.model_predict import predict
from flask import Flask, escape, request, send_from_directory, jsonify

from gevent.pywsgi import WSGIServer

app = Flask(__name__)

print("Loading config. ")
config_file = sys.argv[1]
with open(config_file, "r", encoding="utf-8") as f:
    args = json.loads(f.readlines()[0])
    # args = json.loads(f.read())
for k, v in args.items():
    print(k, ":", v)
print("Loading model and vectors. ")
device = torch.device(args['device'])
model = CoMaModel(args).to(device)
model.load_state_dict(torch.load(args['corpus_path'] + "Models/" + args['job_id'] + '/CoMa.model', map_location=device))
print("Model: ", device, datetime.datetime.now())
model.eval()
vectors = Vectors(args['corpus_path'] + args['dataset'] + "/word_embeddings.bin").stoi
print("Embeddings: ", datetime.datetime.now())
output_vectors = Dictionary.load(args['corpus_path'] + args['dataset'] + "/venue_dict")
print("Output vectors: ", datetime.datetime.now())
print("Website running now: ", datetime.datetime.now())


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)


@app.route('/match', methods=['POST'])
def match():
    title = request.json.get("title", "")
    abstract = request.json.get("abstract", "")
    keywords = request.json.get("keywords", "")

    # Process title, abstract, and keywords.
    # keywords = " ".join(re.split('[^\w-]+', keywords.lower()))
    # abstract = " ".join(re.split('[^\w-]+', abstract.lower()))
    # title = " ".join(re.split('[^\w-]+', title.lower()))

    print(f"Request started: {datetime.datetime.now()}")
    # Return JSON with the same structure as in example.json
    return predict(input_abstract=abstract, input_title=title, input_keywords=keywords, model=model, vectors=vectors,
                   output_vectors=output_vectors, device=device)


if __name__ == '__main__':
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
    print("Page running. ")
