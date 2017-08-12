#!/bin/bash

# Get trees
data=trainDevTestTrees_PTB.zip
curl -O https://nlp.stanford.edu/sentiment/$data
unzip $data
rm -f $data

