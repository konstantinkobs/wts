import datetime
import torch

from torch import nn

from gensim.corpora import Dictionary
from util.utils import build_json
from torchtext.vocab import Vectors
from captum.attr import IntegratedGradients, configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
from util.CoMa_Model import interpret_sentence


def predict(input_abstract: str, input_title: str, input_keywords: str, model: nn.Module, vectors: Vectors,
            output_vectors: Dictionary, device: torch.device):
    """
    :param input_abstract:
    :param input_title:
    :param input_keywords:
    :param model:
    :param vectors:
    :param output_vectors:
    :param device:
    :return:
    """
    # Prepare for visualization
    vis_data_records_ig = []

    # Interpret sentence
    try:
        interpretable_embedding_abstracts = configure_interpretable_embedding_layer(model, 'embedding_abstracts')
        interpretable_embedding_titles = configure_interpretable_embedding_layer(model, 'embedding_titles')
        interpretable_embedding_keywords = configure_interpretable_embedding_layer(model, 'embedding_keywords')
        ig = IntegratedGradients(model)
    except:
        exit(1)

    print(f"Created Interpretable Layers: {datetime.datetime.now()}")

    interpret_sentence(model=model, input_abstract=input_abstract, input_title=input_title,
                       input_keywords=input_keywords, vectors=vectors,
                       interpretable_embedding_abstracts=interpretable_embedding_abstracts,
                       interpretable_embedding_titles=interpretable_embedding_titles,
                       interpretable_embedding_keywords=interpretable_embedding_keywords, ig=ig,
                       vis_data_records_ig=vis_data_records_ig, output_vectors=output_vectors, device=device)

    print(f"Interpreted: {datetime.datetime.now()}")

    # Show interpretations
    #print(build_html(vis_data_records_ig))
    json_data = build_json(vis_data_records_ig)

    print(f"Built JSON: {datetime.datetime.now()}")

    remove_interpretable_embedding_layer(model, interpretable_embedding_abstracts)
    remove_interpretable_embedding_layer(model, interpretable_embedding_titles)
    remove_interpretable_embedding_layer(model, interpretable_embedding_keywords)

    print(f"Removed Layers: {datetime.datetime.now()}")

    return json_data
