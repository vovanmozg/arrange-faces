#!/bin/bash

docker run -v ./images:/images -it face-aligner python create_collages.py
