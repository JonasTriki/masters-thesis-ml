#/bin/bash
python evaluate_word2vec.py \
--model_dir ../output/word2vec_training/03-Nov-2020_11-01-00 \
--model_name word2vec \
--dataset_name enwiki \
--sswr_dataset_filepath data/sswr.pkl \
--msr_dataset_filepath data/msr.pkl \
--vocab_size -1 \
--top_n_prediction 1 \
--output_dir ../output/word2vec_eval_analogies
