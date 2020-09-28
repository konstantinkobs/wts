import sys

import torch
import numpy
import json
import os
from typing import List
import matplotlib.pyplot as plt
from captum.attr import visualization
from captum.attr._utils.visualization import VisualizationDataRecord, format_classname, format_special_tokens, \
    _get_color
from gensim.corpora import Dictionary


def check_top(predictions, field_name, print_top_five=True, truth=None, count=5):
    in_top_five = 0
    counter = 0
    for prediction in predictions:
        top_idx = torch.argsort(prediction, descending=True)[0:count]
        top_values = [prediction[i] for i in top_idx]

        if truth is not None:
            if numpy.isin(truth[counter].cpu().item(), top_idx.cpu()):
                in_top_five += 1

        if print_top_five:
            print("Based on your " + field_name + " we think you should publish at following conferences: ")

            for i in range(0, count):
                print(str(i + 1) + ". " + str(get_venue(top_idx[i].item())) + " - {:.0f}%".format(top_values[i] * 100))

            if truth != None:
                print("It should be " + str(get_venue(truth[counter].item())))
            print("")

        counter += 1

    return in_top_five, top_idx, top_values


def print_loss_curves(model_name, train_loss, val_loss):
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
    plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')

    # find early stopping point
    min_poss = val_loss.index(min(val_loss)) + 1
    plt.axvline(min_poss, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 7)  # consistent scale
    plt.xlim(0, len(train_loss) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('./Documentation/loss_' + model_name + '.png', bbox_inches='tight')


def mean_reciprocal_rank_at_5(y_prob: numpy.array, class_indices: numpy.array, args: dict) -> float:
    ranks = (numpy.swapaxes(numpy.argsort(-y_prob), 0, 1) == class_indices).argmax(0) + 1
    with open(args['corpus_path'] + '/Models/' + args['job_id'] + '/ranks.txt', "w", encoding="utf-8") as f:
        for rank in ranks:
            f.write(str(rank) + "\n")
    print_distribution_curve(ranks, len(y_prob[0]), args)

    rr = 1/ranks
    rr[ranks > 5] = 0

    mrr_at_5 = numpy.mean(rr)

    return mrr_at_5


def print_distribution_curve(ranks, conf_n, args: dict):
    fig = plt.figure()
    plt.hist(ranks, density=True)
    plt.xlabel('Rank')
    plt.ylim(0, 1)  # consistent scale
    plt.xlim(1, conf_n)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    output_path = args['corpus_path'] + '/Models/' + args['job_id'] + '/distribution_curve.png'
    fig.savefig(output_path, bbox_inches='tight')


def add_attributions_to_visualizer(attributions, text, pred, pred_ind, delta, vis_data_records, vectors):
    attributions = attributions.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()
    pred_ind = pred_ind.item()

    # storing couple samples in an array for visualization purposes
    vis_data_records.append(visualization.VisualizationDataRecord(
        attributions,
        pred,
        vectors[pred_ind],
        vectors[pred_ind],
        vectors[1],
        attributions.sum(),
        text[:len(attributions)],
        delta.cpu()))


def build_html(datarecords: VisualizationDataRecord):
    dom = ["<table width: 100%>"]
    rows = [
        "<tr>"
        #      "<th>Predicted Label (1)</th>"
        "<th>Predicted Venue</th>"
        #       "<th>Attribution Label</th>"
        #       "<th>Attribution Score</th>"
        "<th>Word Importance</th>"
    ]
    for datarecord in datarecords:
        rows.append(
            "".join(
                [
                    "<tr>",
                    #                   format_classname(datarecord.target_class),
                    format_classname(
                        "{0} {1:.0f}%".format(
                            datarecord.pred_class, datarecord.pred_prob * 100
                        )
                    ),
                    #                   format_classname(datarecord.attr_class),
                    #                   format_classname("{0:.2f}".format(datarecord.attr_score)),
                    format_word_importances(
                        datarecord.raw_input, datarecord.word_attributions
                    ),
                    "<tr>",
                ]
            )
        )

    dom.append("".join(rows))
    dom.append("</table>")
    return dom


def remove_padding(tokenized_list):
    tokenized_list_new = []

    for token in tokenized_list:
        if token == "pad":
            break
        else:
            tokenized_list_new.append(token)

    return tokenized_list_new


def build_json(data_records: VisualizationDataRecord):
    json_builder = {}
    tokens = {}
    conferences = []
    lengths = {}
    current_conf = {}
    conf_list = get_conf_list()

    i = 0
    for data_record in data_records:
        type = i % 3
        fieldname = ""

        if type == 0:
            fieldname = "title"
            conference = {}
            importances = {}
            current_conf = get_conf_from_list(data_record.pred_class, conf_list)
        elif type == 1:
            fieldname = "abstract"
        elif type == 2:
            fieldname = "keywords"

        if fieldname not in tokens:
            tokens[fieldname] = remove_padding(data_record.raw_input)
            lengths[fieldname] = len(tokens[fieldname])

        importances[fieldname] = data_record.word_attributions.tolist()[0:lengths[fieldname]]

        if fieldname == "keywords":  # reset to start
            conference["importances"] = importances
            conference["info"] = current_conf["info"]
            conference["longname"] = current_conf["longname"]
            conference["info_source"] = current_conf["info_source"]
            conference["name"] = data_record.pred_class
            conferences.append(conference)

        i += 1

    json_builder["tokens"] = tokens
    json_builder["conferences"] = conferences

    return json.dumps(json_builder)


def format_word_importances(words, importances):
    if importances is None or len(importances) == 0:
        return "<td></td>"
    assert len(words) <= len(importances)
    tags = ["<td>"]
    for word, importance in zip(words, importances[: len(words)]):
        if word == "pad":
            continue

        word = format_special_tokens(word)
        color = _get_color(importance)
        unwrapped_tag = '<mark style="background-color: {color}; opacity:1.0; \
                    line-height:1.75"><font color="black"> {word}\
                    </font></mark>'.format(
            color=color, word=word
        )
        tags.append(unwrapped_tag)
    tags.append("</td>")
    return "".join(tags)


def get_venue(id: int):
    output_vectors = Dictionary.load(args['corpus_path'] + args['dataset'] + "/venue_dict").token2id

    return list(output_vectors)[id]


def get_conf_list():
    # todo somehow do not use static paths
    if os.path.exists("../website/confs_description.json"):
        path = "../website/confs_description.json"
    else:
        path = "./website/confs_description.json"
    with open(path, "r", encoding="utf-8") as f:
        confs = f.readlines()
    confs = [json.loads(line) for line in confs]

    return confs


def get_conf_from_list(conf_name: str, conf_list: list):
    for conf in conf_list:
        if conf["name"].lower() == conf_name.lower():
            return conf

    return {"name": "", "longname": "", "info": ""}