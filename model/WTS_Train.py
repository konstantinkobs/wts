import torch
import numpy
import time
import json
import os
import sys
import pandas as pd

sys.path.append('../')
from torch import nn
from torch import optim
from tqdm import tqdm
from model.util.EarlyStopping import EarlyStopping
from util.CoMa_Model import predict_batch
from util.CoMa_Model import CoMaModel
from gensim.corpora import Dictionary
from model.util.IterableConferenceDataset import IterableConferenceDataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, label_ranking_average_precision_score, accuracy_score
from sklearn.preprocessing import label_binarize
from torchtext.vocab import Vectors
from util.utils import mean_reciprocal_rank_at_5


torch.manual_seed(42)
numpy.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class TrainCoMaModel:
    def __init__(self, args: dict):
        """
        :param args:
        """
        self.args = args
        self.vectors = Vectors(args['corpus_path'] + args['dataset'] + "/word_embeddings.bin").stoi
        self.output_vectors = self.get_target_dictionary(path_to_dict=args['corpus_path'] + args['dataset'] +
                                                                      "/venue_dict")
        self.val_ds = IterableConferenceDataset(args['corpus_path'] + args['dataset'] + "/" + args['dataset'] +
                                                "_val.json", args['pad_length_titles'], args['pad_length_abstracts'],
                                                args['pad_length_keywords'], self.vectors, self.output_vectors.token2id)
        self.val_loader = torch.utils.data.DataLoader(self.val_ds, batch_size=args['batch_size'])
        self.start_time = time.time()
        self.early_stopping = EarlyStopping(args)
        self.train_loss_all = []
        self.val_loss_all = []
        args['output_classes'] = self.val_ds.get_amount_output_classes()
        self.device = torch.device(args['device'])
        self.model = CoMaModel(args).to(self.device)
        if "test_only" in args:
            self.only_test = True
        else:
            self.only_test = False
        if not self.only_test:
            self.train_ds = IterableConferenceDataset(
                args['corpus_path'] + args['dataset'] + "/" + args['dataset'] + "_train.json",
                args['pad_length_titles'],
                args['pad_length_abstracts'],
                args['pad_length_keywords'], self.vectors, self.output_vectors.token2id)
            self.train_loader = torch.utils.data.DataLoader(self.train_ds, batch_size=args['batch_size'])
        if args['optimizer'] == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=args['learning_rate'])
        elif args['optimizer'] == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=args['learning_rate'])
        self.loss_function = nn.CrossEntropyLoss()
        self.epoch = 0
        if not self.only_test:
            self.train_epochs()
        self.test_ds = IterableConferenceDataset(
            args['corpus_path'] + args['dataset'] + "/" + args['dataset'] + "_test.json",
            args['pad_length_titles'],
            args['pad_length_abstracts'],
            args['pad_length_keywords'], self.vectors, self.output_vectors.token2id)
        self.test_loader = torch.utils.data.DataLoader(self.test_ds, batch_size=args['batch_size'])
        f1, mrr, mrr5, acc, top_five_acc = self.calculate_scores()
        self.args["f1_score"] = f1
        self.args["mrr_score"] = mrr
        self.args["mrr_five_score"] = mrr5
        self.args["acc_score"] = acc
        self.args["top_five_accuracy"] = top_five_acc

    def get_result_dict(self) -> dict:
        return self.args

    @staticmethod
    def get_word_vectors(path_to_dict: str, path_to_train: str = None):
        """
        :param path_to_dict:
        :param path_to_train:
        :return:
        """
        if os.path.exists(path_to_dict):
            return Vectors(args['corpus_path'] + args['dataset'] + "/word_embeddings.bin").stoi
        else:
            print("Cry")
            exit(1)

    @staticmethod
    def get_target_dictionary(path_to_dict: str, path_to_train: str = None) -> dict:
        """
        :param path_to_dict:
        :param path_to_train;
        :return:
        """
        if os.path.exists(path=path_to_dict):
            print("Loading venues from file. ")
            return Dictionary.load(path_to_dict)
        else:
            print("Cry. ")
            exit(1)
            # y_values = set()
            # with open(path_to_train, "r", encoding="utf-8") as f:
            #     for line in f:
            #         line = json.loads(line)
            #         y_values.update([line['venue'].lower()])
            # ret_dict = Dictionary([list(y_values)])
            # ret_dict.save(path_to_dict)
            # return ret_dict

    def train_epochs(self):
        """
        :param self:
        :return:
        """
        while self.epoch < args['epochs']:
            # Reset after old epoch
            losses = []
            val_losses = []
            accuracies = []
            self.model.train()
            for batch in tqdm(list(iter(self.train_loader)), "Epoch {}".format(self.epoch + 1)):
                # Predict next value
                pred = self.model.forward_batch(batch)

                # Calculate loss
                y = batch[3].to(self.device)

                loss = self.loss_function(pred, y)
                losses.append(loss.item())

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Calculate performance on validation set
            with torch.no_grad():
                for batch in tqdm(list(iter(self.val_loader)), "Calculating train and validation loss"):
                    # Predict next value
                    pred = self.model.forward_batch(batch)

                    # Calculate loss
                    y = batch[3].to(self.device)

                    # Calculate los
                    loss = self.loss_function(pred, y)
                    val_losses.append(loss.item())
                    y_bin = label_binarize(y.cpu(), numpy.arange(len(self.output_vectors.token2id)))
                    pred_bin = label_binarize(torch.max(pred, 1).indices.cpu(), numpy.arange(len(self.output_vectors.token2id)))
                    accuracies.append(accuracy_score(y_true=y_bin, y_pred=pred_bin))
            # Advance epoch
            if self.args['verbose']:
                print("Epoch: ", self.epoch)
                print("Loss: {:.4f}".format(numpy.average(losses)))
                print("Validation Loss: {:.4f}".format(numpy.average(val_losses)))
                print("Validation Accuracy: {:.4f}".format(numpy.average(accuracies)))
            self.train_loss_all.append(numpy.average(losses))
            self.val_loss_all.append(numpy.average(val_losses))
            self.epoch += 1
            self.early_stopping(numpy.average(val_losses), [self.model])
            if self.early_stopping.early_stop:
                break
        self.save_model()

    def save_model(self) -> None:
        # Save best model & print loss curves
        if self.args['verbose']:
            print("Saving network to CoMa.model")
        self.model.load_state_dict(torch.load(args['corpus_path'] + "models/" + '/checkpoint1.pt'))
        torch.save(self.model.state_dict(), args['corpus_path'] + "models/" + '/CoMa.model')
        if self.args['verbose']:
            print("Training Duration: {:.2f} minutes".format((time.time() - self.start_time) / 60))

    def calculate_scores(self) -> (float, float, float, float):
        """
        :return:
        """
        # Load model
        self.model.load_state_dict(torch.load(args['corpus_path'] + "models/" + '/CoMa.model',
                                              map_location=self.device))
        self.model.eval()
        # Setup score calculation
        predictions, probabilities, truths, top_five_accuracies, top_five_idx, top_idx = [], [], [], [], [], []
        with torch.no_grad():
            for batch in self.test_loader:
                # Predict a batch
                prediction, top_five_accuracy, _ = predict_batch(self.model, batch)
                for pred in prediction:
                    tmp = torch.argsort(pred.clone().detach(), descending=True).tolist()
                    top_five_idx.append(tmp[0:5])
                    top_idx.append(tmp[0])
                predictions.extend(torch.max(prediction, 1).indices.cpu())
                probabilities.extend(prediction)
                top_five_accuracies.append(top_five_accuracy)
                # Save truth
                truth = batch[3].to(self.device)
                truths.extend(truth.tolist())
        id2token = {v: k for k, v in self.output_vectors.token2id.items()}
        top_five_classes = pd.DataFrame(data=[(id2token[a], 1) for b in top_five_idx for a in b],
                                        columns=["class", "count"]).\
            groupby("class").agg('sum').sort_values(by=["count"], ascending=False)
        print("### Top 5 predicted classes from ", len(top_five_idx), " total predictions (total ",
              len(top_five_classes), " classes predicted ### \n", top_five_classes)
        top_classes = pd.DataFrame(data=[(id2token[a], 1) for a in top_idx],
                                   columns=["class", "count"]).\
            groupby("class").agg('sum').sort_values(by=["count"], ascending=False)
        print("### Top predicted classes from ", len(top_idx), " total predictions (total ",
              len(top_classes), " classes predicted### \n", top_classes)

        # Print scores
        print("F1 Score: {:.4f}".format(f1_score(truths, predictions, average="weighted")))
        truths_bin = label_binarize(truths, numpy.arange(len(self.output_vectors.token2id)))
        predictions_bin = label_binarize(predictions, numpy.arange(len(self.output_vectors.token2id)))
        probabilities = torch.stack(probabilities).cpu().numpy()
        print("Mean Reciprocal Rank: {:.4f}".format(label_ranking_average_precision_score(truths_bin, probabilities)))
        print("Top 5 MRR: {:.4f}".format(mean_reciprocal_rank_at_5(probabilities, truths, self.args)))
        print("Accuracy: {:.4f}".format(accuracy_score(truths_bin, predictions_bin)))
        print("Top 5 Accuracy: {:.4f}".format(numpy.average(top_five_accuracies)))
        return f1_score(truths, predictions, average="weighted"), \
               label_ranking_average_precision_score(truths_bin, probabilities), \
               mean_reciprocal_rank_at_5(probabilities, truths, args), \
               accuracy_score(truths_bin, predictions_bin), numpy.average(top_five_accuracies)


if __name__ == '__main__':
    # load config
    if sys.argv[1]:
        config_path = sys.argv[1]
    else:
        config_path = "../data/config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = f.readlines()
    config = [json.loads(line) for line in config]
    args = config[0]  # 1 for medline dataset
    TEST_ONLY = True if 'test_only' in args else False
    if TEST_ONLY:
        print("Testing the model only.")
        args['test_only'] = True
    else:
        print("Training new model. ")
    print("Args:")
    for k, v in args.items():
        print(k, ": ", v)
    print("\n")
    print("##############################")
    print("###      Training CoMa     ###")
    print("##############################")
    train_class = TrainCoMaModel(args=args)
    ret_args = train_class.get_result_dict()
    with open(args['corpus_path'] + args['dataset'] + "/results.json", "a+", encoding="utf-8") as f:
        json.dump(ret_args, f)
        f.write("\n")
    print("Finished. ")
