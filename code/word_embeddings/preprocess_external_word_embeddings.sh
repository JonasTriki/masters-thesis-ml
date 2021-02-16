#!/bin/bash
python preprocess_external_word_embeddings.py \
--raw_data_dir raw_data \
--output_dir data \
--annoy_index_n_trees 250