# Optimised generator continuous NMT

**This project is part of my thesis and not actively maintained.**

## Introduction

This repository hosts the **Optimised Continuous NMT Generator**, a custom implementation designed to enhance the inference speed of sequence generation in Neural Machine Translation (NMT) tasks without losing translation quality, built on top of [Evgeniia's continuous model](https://github.com/afeena/approx_knn). 

The default sequence generator in Fairseq requires the calculated normalised probabilities in the shape of `(batch size, beam size, vocabulary size)` to determine the next token. However, when working with top-K representations provided by efficient similarity search libraries like FAISS, the default generator becomes inefficient. It necessitates expensive transformations to convert the top-K results into the required format, resulting in computational overhead. The optimised generator directly utilises the top-K representation, eliminating the need for costly conversions. This modification significantly improves the efficiency and performance of continuous NMT sequence generation.

## Requirements
1. Install fairseq by following the official installation guide provided [here](https://fairseq.readthedocs.io/en/latest/index.html).
2. Install PyTorch by referring to the official installation guide available [here](https://pytorch.org/get-started/locally/).
3. Install FAISS, a library for efficient similarity search and clustering, by following the installation instructions provided [here](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md).

## Implementation details
- A [new task](https://github.com/izak0s/optimised_continuous_nmt/blob/main/fairseq_easy_extend/tasks/OptimisedTranslationTask.py#L13) has been created on top of the default translation task
- The modified generator can be found [here](https://github.com/izak0s/optimised_continuous_nmt/blob/main/fairseq_easy_extend/generators/OptimisedSequenceGenerator.py#L56)
- The decoder component of the generator requires an implementation to return the indices and distances tensor from the top-k representation. Please ensure you implement this functionality accordingly ([see example](https://github.com/izak0s/optimised_continuous_nmt/blob/main/fairseq_easy_extend/models/transformer/optimised_decoder_continuous.py#L30-L35)).

## Running the Optimised Generator
```bash
python decode.py [FAIRSEQ_DISJOINT_PATH] \
--task optimised_translation \
--model-overrides '{"_name": "continuous_transformer_optimised"}' \
--path [CHECKPOINT_PATH] \
--[ANY OTHER PARAMETERS FOR YOUR MODEL]
```
