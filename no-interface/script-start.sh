#!/bin/bash

docker build -t reliability_app .

docker run --rm -it reliability_app