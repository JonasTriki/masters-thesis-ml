#!/bin/bash
python preprocess_wikipedia_data.py \
--language norwegian \
--wiki_dump_time 20210101 \
--raw_data_dir raw_data \
--output_dir data/nowiki-20210101 \
--min_sent_word_count 5 \
--num_output_files -1 \
--max_wikipedia_files -1
