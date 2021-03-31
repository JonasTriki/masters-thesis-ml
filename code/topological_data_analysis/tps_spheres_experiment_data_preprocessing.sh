#!/bin/bash
screen -dmS tps_spheres_data_experiment -L -Logfile tps_spheres_data_experiment.logs python tps_spheres_experiment_data_preprocessing.py \
--tps_neighbourhood_size 50 \
--output_dir data
screen -r tps_spheres_data_experiment