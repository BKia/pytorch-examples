#!/bin/bash
wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
tar -zxvf simple-examples.tgz
rm simple-examples.tgz
mv simple-examples/data/ptb.train.txt train.txt
mv simple-examples/data/ptb.valid.txt valid.txt
mv simple-examples/data/ptb.test.txt test.txt
rm -rf simple-examples/
