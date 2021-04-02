#!/bin/bash
screen -dmS estimate_num_meanings_supervised -L -Logfile estimate_num_meanings_supervised.logs python estimate_num_meanings_supervised.py \
--train_data_filepath data/word_meaning_train_data.csv \
--output_dir data
screen -r estimate_num_meanings_supervised