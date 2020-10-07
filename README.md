# Where to Submit?
## Helping Researchers to Choose the Right Venue

In this repository you find the code for our paper "Where to Submit? Helping Researchers to Choose the Right Venue" 
that will be published in "Findings of EMNLP 2020".

> Whenever researchers write a paper, the same question occurs: "Where to submit?"
> In this work, we introduce WTS, an open and interpretable NLP system that recommends conferences and journals to researchers based on the title, abstract, and/or keywords of a given paper.
> We adapt the TextCNN architecture and automatically analyze its predictions using the Integrated Gradients method to highlight words and phrases that led to the recommendation of a scientific venue.
> We train and test our method on publications from the fields of artificial intelligence (AI) and medicine, both derived from the Semantic Scholar dataset.
> WTS achieves an Accuracy@5 of approximately 83 % for AI papers and 95 % in the field of medicine.
> It is open source and available for testing on [https://wheretosubmit.ml](https://wheretosubmit.ml).

To run this code, clone this repository and `cd` into the directory in you terminal.
The easiest way to run the code is to use [Docker](https://www.docker.com), so make sure that it is installed on your machine.
To reproduce our results, please go through the following steps:

1. [Create the Dataset](./dataset/README.md)
2. [Train and Test the Model](./model/README.md)
3. [Run the Website (optional)](./website/README.md)

# Citation

If you use the preprocessed data or code, please cite the paper:

```
TO BE ANNOUNCED
```
