import re
import gensim
import spacy
import torch
from torch import nn
from gensim.corpora import Dictionary
from torchtext.vocab import Vectors
from captum.attr import IntegratedGradients, configure_interpretable_embedding_layer
from util.utils import add_attributions_to_visualizer, check_top
from captum.attr import TokenReferenceBase


class CoMaModel(nn.Module):
    def __init__(self, args: dict):
        super(CoMaModel, self).__init__()
        # Prepare weights
        model = gensim.models.KeyedVectors.load_word2vec_format(
            args['corpus_path'] + args['dataset'] + "/word_embeddings.bin")
        weights = torch.FloatTensor(model.vectors)
        weights = torch.cat((weights, torch.zeros(1, weights.size()[1])), 0)

        self.pad_length_titles = args['pad_length_titles']
        self.pad_length_keywords = args['pad_length_keywords']
        self.pad_length_abstracts = args['pad_length_abstracts']

        # Word Embeddings for Abstracts
        self.embedding_abstracts = nn.Embedding.from_pretrained(weights)
        self.embedding_abstracts.weight.requires_grad = True
        self.embedding_abstracts.padding_idx = weights.size()[0]-1

        # Word Embeddings for Titles
        self.embedding_titles = nn.Embedding.from_pretrained(weights)
        self.embedding_titles.weight.requires_grad = True
        self.embedding_titles.padding_idx = weights.size()[0]-1

        # Word Embeddings for Keywords
        self.embedding_keywords = nn.Embedding.from_pretrained(weights)
        self.embedding_keywords.weight.requires_grad = True
        self.embedding_keywords.padding_idx = weights.size()[0]-1

        # Convolution for Abstracts
        self.conv_abstracts = nn.ModuleList(
            [nn.Conv2d(1, args['kernel_n_abstracts'], (i, self.embedding_abstracts.embedding_dim)) for i in
             args['kernel_sizes_abstracts']])
        self.conv_bn_abstracts = nn.BatchNorm1d(args['kernel_n_abstracts'])

        # Convolution for Titles
        self.conv_titles = nn.ModuleList(
            [nn.Conv2d(1, args['kernel_n_titles'], (i, self.embedding_titles.embedding_dim)) for i in
             args['kernel_sizes_titles']])
        self.conv_bn_titles = nn.BatchNorm1d(args['kernel_n_titles'])

        # Convolution for Keywords
        self.conv_keywords = nn.ModuleList(
            [nn.Conv2d(1, args['kernel_n_keywords'], (i, self.embedding_keywords.embedding_dim)) for i in
             args['kernel_sizes_keywords']])
        self.conv_bn_keywords = nn.BatchNorm1d(args['kernel_n_keywords'])

        # Fully connected layer at the end
        self.tokens_n = args['output_classes']
        self.input_n = len(args['kernel_sizes_abstracts']) * args['kernel_n_abstracts'] + len(
            args['kernel_sizes_titles']) * args['kernel_n_titles'] + len(args['kernel_sizes_keywords']) * args[
                           'kernel_n_keywords']
        self.hidden_n = int(self.input_n * (2 / 3) + self.tokens_n)

        self.fc = nn.Linear(self.input_n, self.hidden_n)
        self.fc2 = nn.Linear(self.hidden_n, self.tokens_n)

        # Dropout & Softmax
        self.dropout = nn.Dropout(args['dropout'])
        self.Softmax = nn.Softmax(dim=1)

        self.field_name = "abstracts, titles and keywords"
        self.nlp = spacy.load('en')
        self.interp = ""

        self.using_device = torch.device(args['device'])

    def forward(self, data: tuple, additional1, additional2):
        """
        Overwrite the forward method from nn.Module
        :param data: Tuple of Abstracts, titles and keywords
        :param additional1:
        :param additional2:
        :return:
        """
        x_abstracts = data
        x_titles = data
        x_keywords = data

        if self.interp == "abstracts":
            x_titles = additional1
            x_keywords = additional2
        elif self.interp == "titles":
            x_abstracts = additional1
            x_keywords = additional2
        elif self.interp == "keywords":
            x_titles = additional1
            x_keywords = additional2
        else:
            print("Somethings fishy")

        return self.forward_internal(x_abstracts, x_titles, x_keywords)

    def forward_batch(self, data: tuple) -> torch.tensor:
        """

        :param data:
        :return:
        """
        # Split data in abstracts, titles & keywords
        x_abstracts = data[1]
        x_titles = data[0]
        x_keywords = data[2]

        return self.forward_internal(x_abstracts, x_titles, x_keywords)

    def forward_internal(self, x_abstracts: torch.tensor, x_titles: torch.tensor,
                         x_keywords: torch.tensor) -> torch.tensor:
        """
        Internal forward pass, split three inputs
        :param x_abstracts: tensor of abstract embeddings
        :param x_titles: tensor of title embeddings
        :param x_keywords: tensor of keyword embeddings
        :param running_cpu:
        :return: y tensor
        """
        # Send to cuda if available
        x_abstracts = x_abstracts.to(self.using_device)
        x_titles = x_titles.to(self.using_device)
        x_keywords = x_keywords.to(self.using_device)

        # Abstracts
        x_abstracts = self.embedding_abstracts(x_abstracts)
        x_abstracts = x_abstracts.unsqueeze(1)
        x_abstracts = [self.conv_bn_abstracts(torch.relu(conv(x_abstracts)).squeeze(3)) for conv in self.conv_abstracts]
        x_abstracts = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in x_abstracts]
        x_abstracts = torch.cat(x_abstracts, 1)  # todo check if needed (torch.tensor)

        # Titles
        x_titles = self.embedding_titles(x_titles).unsqueeze(1)
        x_titles = [self.conv_bn_titles(torch.relu(conv(x_titles)).squeeze(3)) for conv in self.conv_titles]
        x_titles = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in x_titles]
        x_titles = torch.cat(x_titles, 1)  # todo check if needed (torch.tensor)

        # Keywords
        x_keywords = self.embedding_keywords(x_keywords).unsqueeze(1)
        x_keywords = [self.conv_bn_keywords(torch.relu(conv(x_keywords)).squeeze(3)) for conv in self.conv_keywords]
        x_keywords = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in x_keywords]
        x_keywords = torch.cat(x_keywords, 1)  # todo check if needed (torch.tensor)

        # Combine the vectors
        x = torch.cat([x_abstracts, x_titles, x_keywords], 1)

        # Train Fully Connected Layers with Dropout
        y = torch.relu(self.dropout(self.fc(x)))
        y = self.fc2(y)

        if not self.training:
            y = self.Softmax(y)

        return y

    def forward_with_sigmoid(self, x_abstracts: torch.tensor, x_titles: torch.tensor,
                             x_keywords: torch.tensor) -> torch.tensor:
        """
        Used for Captum interpretable
        :param x_abstracts: tensor of abstract embeddings
        :param x_titles: tensor of title embeddings
        :param x_keywords: tensor of keyword embeddings
        :return: y tensor with sigmoid activation
        """
        return torch.sigmoid(self.forward_internal(x_abstracts, x_titles, x_keywords).type(torch.FloatTensor))


def predict_batch(model: nn.Module, batch, print_top_five: bool = False, get_top: int = 5) -> (torch.tensor, int, list):
    """
    :param model:
    :param batch:
    :param print_top_five:
    :param get_top:
    :return:
    """
    model.eval()
    if len(batch) == 4:
        truth = batch[3].cuda() if torch.cuda.is_available() else batch[3]
    else:
        truth = None
    predictions = model.forward_batch(batch)
    in_top_five, top_list, _ = check_top(predictions, model.field_name, print_top_five, truth, get_top)
    top_five_accuracy = in_top_five / len(predictions)
    return predictions, top_five_accuracy, top_list


def get_input_indices(model, input_text, vocab, min_len, device, input_type):
    # text = [tok.text for tok in model.nlp.tokenizer(input)]
    text = re.split('[^\w-]+', input_text.lower())

    indexed = []
    for t in text[:200]:
        tokenized = vocab.get(t.lower())
        if tokenized is not None:
            indexed.append(tokenized)
        else:
            indexed.append(0)

    if len(text) < min_len:
        indexed += [len(vocab)] * (min_len - len(text))
    
    input_indices = torch.LongTensor(indexed)
    input_indices = input_indices.unsqueeze(0).to(device)
    return text, input_indices


def interpret_sentence(model: nn.Module, input_abstract: str, input_title: str, input_keywords: str, vectors: Vectors,
                       interpretable_embedding_abstracts: configure_interpretable_embedding_layer,
                       interpretable_embedding_titles: configure_interpretable_embedding_layer,
                       interpretable_embedding_keywords: configure_interpretable_embedding_layer,
                       ig: IntegratedGradients, vis_data_records_ig: list, output_vectors: Dictionary,
                       device: torch.device, min_len: int = 200):
    model.eval()
    model.zero_grad()

    abstract_token_reference = TokenReferenceBase(reference_token_idx=len(vectors))
    title_token_reference = TokenReferenceBase(reference_token_idx=len(vectors))
    keywords_token_reference = TokenReferenceBase(reference_token_idx=len(vectors))

    abstract_text, abstract_indices = get_input_indices(model, input_abstract, vectors, min_len, device, input_type='abstract')
    title_text, title_indices = get_input_indices(model, input_title, vectors, min_len, device, input_type='title')
    keywords_text, keywords_indices = get_input_indices(model, input_keywords, vectors, min_len, device, input_type='keywords')

    # input_indices dim: [sequence_length]
    seq_length = min_len

    abstract_indices = abstract_indices.to(device)
    title_indices = title_indices.to(device)
    keywords_indices = keywords_indices.to(device)

    # pre-computing word embeddings
    input_embedding_abstracts = interpretable_embedding_abstracts.indices_to_embeddings(abstract_indices)
    input_embedding_titles = interpretable_embedding_titles.indices_to_embeddings(title_indices)
    input_embedding_keywords = interpretable_embedding_keywords.indices_to_embeddings(keywords_indices)

    # predict
    pred = model.forward_internal(input_embedding_abstracts, input_embedding_titles, input_embedding_keywords)
    in_top_five, top_list, top_values = check_top(pred, model.field_name, False)

    for i in range(len(top_list)):
        model.interp = "titles"
        interpret_subsystem(top_list[i], top_values[i], input_embedding_titles, input_embedding_abstracts,
                            input_embedding_keywords, interpretable_embedding_titles, title_text,
                            title_token_reference, seq_length, ig, vis_data_records_ig, output_vectors, device)
        model.interp = "abstracts"
        interpret_subsystem(top_list[i], top_values[i], input_embedding_abstracts, input_embedding_titles,
                            input_embedding_keywords, interpretable_embedding_abstracts, abstract_text,
                            abstract_token_reference, seq_length, ig, vis_data_records_ig, output_vectors, device)
        model.interp = "keywords"
        interpret_subsystem(top_list[i], top_values[i], input_embedding_keywords, input_embedding_abstracts,
                            input_embedding_titles, interpretable_embedding_keywords, keywords_text,
                            keywords_token_reference, seq_length, ig, vis_data_records_ig, output_vectors, device)


def interpret_subsystem(pred_ind, pred_value, input_embedding, input_emb2, input_emb3, embedding, text, token_reference,
                        seq_length, ig, vis_data_records_ig, output_vectors, device):
    # generate reference for each sample
    reference_indices = token_reference.generate_reference(seq_length, device=device).unsqueeze(0)
    reference_embedding = embedding.indices_to_embeddings(reference_indices)

    # compute attributions and approximation delta using integrated gradients
    attributions_ig, delta = ig.attribute(input_embedding, reference_embedding,
                                          additional_forward_args=tuple([input_emb2, input_emb3]),
                                          target=pred_ind, n_steps=50, return_convergence_delta=True)

    add_attributions_to_visualizer(attributions_ig, text, pred_value, pred_ind, delta, vis_data_records_ig,
                                   output_vectors)
