#!/usr/bin/env bash

# download SQuAD dataset
DATA_DIR=./data/squad1.0/
mkdir -p "$DATA_DIR"

wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O "$DATA_DIR"/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O "$DATA_DIR"/dev-v1.1.json

# download embedding
EMBEDDING_DIR=./data/embedding/
mkdir -p "$EMBEDDING_DIR"

wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O "$EMBEDDING_DIR"/glove.840B.300d.zip
unzip "$EMBEDDING_DIR"/glove.840B.300d.zip -d "$EMBEDDING_DIR"

mkdir -p ./data/output