python train_word2phrase.py \
--text_data_dir data/nowiki-20200901 \
--dataset_name nowiki-20200901 \
--n_epochs 2 \
--max_vocab_size -1 \
--min_word_count 5 \
--threshold 200 \
--threshold_decay 0.5 \
--phrase_sep _ \
--output_dir data