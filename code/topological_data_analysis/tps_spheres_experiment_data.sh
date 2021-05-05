#!/bin/bash
screen -dmS tps_spheres_experiment_data -L -Logfile tps_spheres_experiment_data.logs python tps_spheres_experiment_data.py \
--tps_neighbourhood_size 50 \
--output_dir data
screen -r tps_spheres_experiment_data