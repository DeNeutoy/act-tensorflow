# act-tensorflow
Adaptive Computation Time algorithm in Tensorflow

This repo contains a ACTCell.py file impementing ACT and inheriting from the abstract RNN class in the Tensorflow doccumentation [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell.py). It implements the Adaptive Computation Time Algorithm, described in [this paper](http://arxiv.org/pdf/1603.08983v4.pdf).

Also included in the repo is a model which uses this cell for language modelling on the Penn Tree Bank,
a common dataset for evaluating language models. To run the ACT_Training.py file, you will need to download
the dataset, which can be found [here](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz). The files
you need are in the data/ directory.

In order to run the code in this repo you will need to have downloaded one of the daily Tensorflow binaries which you can find on their homepage.


To run:

run ACT_Training.py with the following parameters:

    * model_size, Default: "small", Size of model to train, either small, medium or large(from the config.py file)
    * data_path, Default: None, full path to the data directory containing the ptb files
    * model_path, Default: None, the full path of a pickled dictionary of weights(such as from a previously trained model)
     saved using saveload.py to reload
    * weights_dir, Default:None, full directory path to save weights into per epoch


Note - you will probably need some fairly heavy machinery to run the large config. The small model
will comfortably run with around 4G of memory, although it will be pretty slow.

