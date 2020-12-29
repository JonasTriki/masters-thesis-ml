#!/bin/bash
python train_word2vec.py \
--text_data_dir data/nowiki-20200901_phrases/epoch_2 \
--dataset_name nowiki \
--output_dir ../output/word2vec_training \
--save_to_tokenizer_filepath data/nowiki-20200901.tokenizer \
--intermediate_embedding_weights_saves 10 \
--train_logs_to_file \
--tensorboard_logs_dir tensorboard_logs \
--max_vocab_size -1 \
--min_word_count 5 \
--batch_size 256 \
--n_epochs 5 \
--learning_rate 0.025 \
--min_learning_rate 0.0000025 \
--embedding_dim 300 \
--max_window_size 5 \
--num_negative_samples 5 \
--sampling_factor 0.00001 \
--unigram_exponent_negative_sampling 0.75