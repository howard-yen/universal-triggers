# Universal Adversarial Triggers for Text Generation Models

This repository is forked from the paper that we are replicating for our class project. It contains both the official code from the EMNLP 2019 paper, Universal Adversarial Triggers for Attacking and Analyzing NLP, as well the code used for text generation models. This repository contains the code for replicating our experiments and creating universal triggers.

Read their [blog](http://www.ericswallace.com/triggers) and their [paper](https://arxiv.org/abs/1908.07125) for more information on the method.

We also wrote a paper with our analysis of this method of text generation models, which are available upon request.

## Dependencies

This code is written using PyTorch. The code for GPT-2 is based on [HuggingFace's Transformer repo](https://github.com/huggingface/pytorch-transformers) and the experiments on SQuAD, SNLI, and SST use [AllenNLP](https://github.com/allenai/allennlp/). The code is flexible and should be generally applicable to most models (especially if its in AllenNLP), i.e., you can easily extend this code to work for the model or task you want. 

To run the text generation models, we use the newest version of the Huggingface library (`transformers=4.5.0`), as well as the newest versions of pytorch, which are not the same version as the code in the original paper.

The code is made to run on GPU, and a GPU is likely necessary due to the costs of running the larger models. They used one GTX 1080 for all the experiments in the original paper; most experiments run in a few minutes. It is possible to run the SST and SNLI experiments without a GPU. For text generation, we used a 11G GPU, and the run time of the experiments depends on the model. Some models like DistilGPT2 runs within a couple minutes with GPU, and can easily run without a GPU. On the other hand, other models like GPT-neo takes up to an hour to run with the GPU and impossible to run without GPU, even the smaller 125M parameters version!

## Installation

An easy way to install the code is to create a fresh anaconda environment:

```
conda create -n triggers python=3.7
source activate triggers
pip install -r requirements.txt
```
Now you should be ready to go!

## Getting Started

The repository is broken down by task: 
+ `sst` attacks sentiment analysis using the SST dataset (AllenNLP-based).
+ `snli` attacks natural language inference models on the SNLI dataset (AllenNLP-based).
+ `squad` attacks reading comprehension models using the SQuAD dataset (AllenNLP-based).
+ `gpt2` attacks the GPT-2 language model using HuggingFace's model.
+ `text-generation` attacks the text-generation models: GPT2, DistilGPT2, GPT-Neo, Xlnet, and more.

To get started, we recommend you start with `snli` or `sst`. In `snli`, we download pre-trained models (no training required) and create the triggers for the hypothesis sentence. In `sst`, we walk through training a simple LSTM sentiment analysis model in AllenNLP. It then creates universal adversarial triggers for that model. The code is well documented and walks you through the attack methodology.

The gradient-based attacks are written in `attacks.py`. The file `utils.py` contains the code for evaluating models, computing gradients, and evaluating the top candidates for the attack. `utils.py` is only used by the AllenNLP models (i.e., not for GPT-2).

## References

Please consider citing our work if you found this code or our paper beneficial to your research.
```
@inproceedings{Wallace2019Triggers,
  Author = {Eric Wallace and Shi Feng and Nikhil Kandpal and Matt Gardner and Sameer Singh},
  Booktitle = {Empirical Methods in Natural Language Processing},                            
  Year = {2019},
  Title = {Universal Adversarial Triggers for Attacking and Analyzing {NLP}}
}    
```

## Contributions and Contact

Text generation code is written by Howard Yen and Andrew Chen, reachable at hyen@princeton.edu and arc4@princeton.edu.

This original paper was developed by Eric Wallace, contact available at ericwallace@berkeley.edu.

If you'd like to contribute code, feel free to open a [pull request](https://github.com/Eric-Wallace/universal-triggers/pulls). If you find an issue with the code, please open an [issue](https://github.com/Eric-Wallace/universal-triggers/issues).
