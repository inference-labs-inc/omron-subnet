#!/bin/bash

# Define variables
IMAGE_NAME="intelanalytics/bigdl-ppml-trusted-bigdl-llm-base"
IMAGE_TAG="2.4.0-SNAPSHOT"
DOCKERFILE="./Dockerfile"
NO_PROXY="x.x.x.x"

# Build Docker image
sudo docker build \
    --build-arg no_proxy="$NO_PROXY" \
    -t "$IMAGE_NAME:$IMAGE_TAG" \
    -f "$DOCKERFILE" .
