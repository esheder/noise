#!/bin/bash

shift=$1

./analog_timing.py $((SLURM_ARRAY_TASK_ID + shift)) ;
