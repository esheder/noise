#!/bin/bash

shift=$1

./sde_timing.py $((SLURM_ARRAY_TASK_ID + shift)) ;
