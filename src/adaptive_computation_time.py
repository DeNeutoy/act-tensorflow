#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from act_cell import ACTCell
from tensorflow.python.ops.nn import rnn_cell, rnn, seq2seq
from tensorflow.python.ops import array_ops


class ACTModel(object):

    def __init__(self, config, is_training=False):
        self.config = config
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.hidden_size = hidden_size = config.hidden_size
        self.num_layers = 1
        vocab_size = config.vocab_size
        self.max_grad_norm = config.max_grad_norm
        self.use_lstm = config.use_lstm

        # Placeholders for inputs.
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.initial_state = array_ops.zeros(
                array_ops.pack([self.batch_size, self.num_steps]),
                 dtype=tf.float32).set_shape([None, self.num_steps])

        embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.hidden_size])

        # Set up ACT cell and inner rnn-type cell for use inside the ACT cell.
        with tf.variable_scope("rnn"):
            if self.use_lstm:
                inner_cell = rnn_cell.BasicLSTMCell(self.config.hidden_size)
            else:
                inner_cell = rnn_cell.GRUCell(self.config.hidden_size)

        with tf.variable_scope("ACT"):

            act = ACTCell(self.config.hidden_size, inner_cell, config.epsilon,
                          max_computation=config.max_computation, batch_size=self.batch_size)

        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        inputs = [tf.squeeze(single_input, [1]) for single_input in tf.split(1, self.config.num_steps, inputs)]

        self.outputs, final_state = rnn(act, inputs, dtype = tf.float32)

        # Softmax to get probability distribution over vocab.
        output = tf.reshape(tf.concat(1, self.outputs), [-1, hidden_size])
        softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
        self.logits = tf.matmul(output, softmax_w) + softmax_b   # dim (numsteps*batchsize, vocabsize)

        loss = seq2seq.sequence_loss_by_example(
                [self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([batch_size * num_steps])],
                vocab_size)

        # Add up loss and retrieve batch-normalised ponder cost: sum N + sum Remainder.
        ponder_cost = act.calculate_ponder_cost(time_penalty=self.config.ponder_time_penalty)
        self.cost = (tf.reduce_sum(loss) / batch_size) + ponder_cost
        self.final_state = self.outputs[-1]

        if is_training:
            self.lr = tf.Variable(0.0, trainable=False)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.max_grad_norm)
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
