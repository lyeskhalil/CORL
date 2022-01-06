#!/bin/bash

salloc --account=def-dkundur --gres=gpu:v100l:1 --cpus-per-task=6 --mem=32000M --time=$1
