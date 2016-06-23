#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import os
import pickle, os, sys
from datetime import datetime
from collections import defaultdict
import tensorflow as tf
from epoch import run_epoch
from config import SmallConfig
from variable_batch_masking import ACTCellMasking
import random
try:
    from tensorflow.python.ops.nn import rnn_cell, rnn, seq2seq
except:
    sys.path.insert(0, os.environ['HOME'])
    from Project_RNN_Enhancement.rnn_enhancement import rnn_cell_enhanced as rnn_cell
    from Project_RNN_Enhancement.rnn_enhancement import rnn_enhanced as rnn
    from Project_RNN_Enhancement.rnn_enhancement import victory_music

    seq2seq = tf.nn.seq2seq
from tensorflow.python.ops import array_ops, functional_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops.tensor_array_ops import TensorArray
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs


class ACTModel(object):

    def __init__(self, config, is_training=False):
        self.config = config
        self.batch_size = batch_size = 20
        self.num_steps = num_steps = config.num_steps
        self.hidden_size = hidden_size =  config.hidden_size
        self.num_layers = 1
        vocab_size = config.vocab_size
        self.max_grad_norm = config.max_grad_norm

        # placeholders for inputs
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        self.initial_state = array_ops.zeros(
                array_ops.pack([self.batch_size, self.num_steps]),
                 dtype=tf.float32).set_shape([None, self.num_steps])


        with tf.variable_scope("rnn"):
            inner_cell = rnn_cell.BasicRNNCell(self.config.hidden_size)
            #inner_cell = rnn_cell.MultiRNNCell([mrnn]* self.config.num_layers)
            rnn_state = inner_cell.zero_state(self.batch_size, dtype=tf.float32)

        with tf.variable_scope("ACT"):
            act = ACTCellMasking(self.config.hidden_size, inner_cell, 0.01, 10, self.batch_size)
            #act = ACTCell_VariableBatchSize(self.config.hidden_size, inner_cell, 0.01, 10, batch_size = self.batch_size)

        embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.hidden_size])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        inputs = [tf.squeeze(single_input, [1]) for single_input in tf.split(1, self.config.num_steps, inputs)]
        #inputs = tf.pack(inputs)

        self.outputs, final_state = rnn(act, inputs, #initial_state=rnn_state,
                                            dtype = tf.float32)



        '''###### Softmax to get distribution over vocab #######'''
        output = tf.reshape(tf.concat(1, self.outputs), [-1, hidden_size])
        softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
        self.logits = tf.matmul(output, softmax_w) + softmax_b   # dim (400, 10,000)(numsteps*batchsize, vocabsize)

        loss = seq2seq.sequence_loss_by_example(
                [self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([batch_size * num_steps])],
                vocab_size)


        self.cost = tf.reduce_sum(loss) / batch_size + act.CalculatePonderCost(time_penalty = 0.01)
        self.final_state = self.outputs[-1]

        if is_training:

            self.lr = tf.Variable(0.0, trainable=False)

            tvars = tf.trainable_variables()  # all params inside model which are trainable
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.max_grad_norm)

            optimizer = tf.train.GradientDescentOptimizer(self.lr)
            #optimizer = tf.train.AdamOptimizer(self.lr)

            self.train_op = optimizer.apply_gradients(zip(grads, tvars))



class ACTCell_BatchSize1(rnn_cell.RNNCell):

    """An RNN cell implementing Graves' Adaptive Computation Time algorithm"""


    def __init__(self, num_units, cell, epsilon, max_computation):

        self.one_minus_eps = tf.constant(1.0 - epsilon)
        self._num_units = num_units
        self.cell = cell
        self.N = tf.constant(max_computation)

    @property
    def input_size(self):
        return self._num_units
    @property
    def output_size(self):
        return self._num_units
    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, timestep = 0, scope=None):
        timestep = random.randint(1, 10000)

        with vs.variable_scope(scope or type(self).__name__):

            # define within cell constants/ counters used to control while loop
            prob = tf.constant(0.0,tf.float32, name="prob")
            counter = tf.constant(0, dtype=tf.int32, name="counter")
            acc_outputs = tf.zeros_like(state, tf.float32, name="output_accumulator")
            acc_states = tf.zeros_like(state, tf.float32, name="state_accumulator")

            # the predicate for stopping the while loop. Tensorflow demands that we have
            # all of the variables used in the while loop in the predicate.
            pred = lambda prob,counter,state,input,acc_output,acc_state:\
                tf.logical_and(tf.less(prob,self.one_minus_eps), tf.less(counter,self.N))

            prob,iterations,_,_,acc_outputs,acc_states = control_flow_ops.while_loop(

                pred,self.ACTStep,[prob,counter,state,inputs, acc_outputs, acc_states])


        output = acc_outputs
        next_state = acc_states
        tf.add_to_collection("ACT_remainder", 1.0-prob)  # TODO: check if this is right
        tf.add_to_collection("ACT_iterations", iterations)

        print('got through one complete timestep')
        return output, next_state

    def ACTStep(self,prob,counter,state,input,acc_outputs,acc_states):

        #you may need to change this to rnn.rnn depending on tensorflow versions
        output, new_state = rnn(self.cell, [input], state, scope=type(self.cell).__name__)

        with tf.variable_scope('sigmoid_activation_for_pondering'):
            p = tf.squeeze(tf.nn.rnn_cell._linear(new_state, 1, True))
            prob += p

        def use_remainder():
            remainder = 1.0 - prob
            acc_state = tf.add(tf.mul(new_state,remainder), acc_states)
            acc_output = tf.add(tf.mul(output[0], remainder), acc_outputs)
            return acc_state, acc_output

        def normal():
            acc_state = tf.add(tf.mul(new_state,p), acc_states)
            acc_output = tf.add(tf.mul(output[0], p), acc_outputs)
            return acc_state, acc_output

        acc_state, acc_output = tf.cond(tf.less(prob,self.one_minus_eps), normal, use_remainder)

        return [prob,counter + 1,new_state, input, acc_output,acc_state]

    def get_ponder_cost(self, epsilon):

        n_iterations = tf.get_collection_ref("ACT_iterations")
        remainder = tf.get_collection_ref("ACT_remainder")
        return tf.reduce_sum(tf.to_float(n_iterations) + remainder)








class ACTCell_VariableBatchSize(rnn_cell.RNNCell):

    """An RNN cell implementing Graves' Adaptive Computation Time algorithm"""


    def __init__(self, num_units, cell, epsilon, max_computation, batch_size = 7):
        self.batch_size = batch_size
        self.constant_batch_size = [self.batch_size, 1]
        self.one_minus_eps = tf.constant(1.0 - epsilon, shape = self.constant_batch_size)
        self.num_units = num_units
        self.cell = cell
        self.N = tf.constant(max_computation, shape = self.constant_batch_size)

    @property
    def input_size(self):
        return self.num_units
    @property
    def output_size(self):
        return self.num_units
    @property
    def state_size(self):
        return self.num_units

    def __call__(self, inputs, state, timestep = 0, scope=None):
        timestep = random.randint(1, 10000)

        with vs.variable_scope(scope or type(self).__name__):

            # define within cell constants/ counters used to control while loop
            prob = tf.constant(0.0,tf.float32, shape = self.constant_batch_size, name="prob")
            counter = tf.constant(0, dtype=tf.int32, shape = self.constant_batch_size, name="counter")
            acc_outputs = tf.zeros([self.batch_size, self.num_units], tf.float32, name="output_accumulator")
            acc_states = tf.zeros([self.batch_size, self.num_units], tf.float32, name="state_accumulator")

            pred = lambda prob,counter,state,input,acc_output,acc_state:\
                tf.logical_and(tf.less(prob,self.one_minus_eps), tf.less(counter,self.N))

            #need to unpack state and inputs
            inputs_unpacked = self.UnpackAndExpand(inputs) #this would be a list of len(batch_size), each item is 1d [hidden_size]
            states_unpacked = self.UnpackAndExpand(state)

            #i think we should unpack everything
            prob_unpacked = self.UnpackAndExpand(prob)
            counter_unpacked = self.UnpackAndExpand(counter)
            acc_outputs_unpacked = self.UnpackAndExpand(acc_outputs)
            acc_states_unpacked = self.UnpackAndExpand(acc_states)

            prob_list, iterations_list, acc_outputs_list, acc_states_list = [], [], [], []

            for b in range(self.batch_size):
                prob_list[b],iterations_list[b],_,_,acc_outputs_list[b],acc_states_list[b] = control_flow_ops.while_loop(

                    pred,self.ACTStep,[prob_unpacked[b],counter_unpacked[b],states_unpacked[b],inputs_unpacked[b], acc_outputs_unpacked[b], acc_states_unpacked[b]])


        output = self.CompressAndPack(acc_outputs_list)
        next_state = self.CompressAndPack(acc_states_list)
        tf.add_to_collection("ACT_remainder", self.batch_size-tf.reduce_sum(prob))  # TODO: check if this is right
        tf.add_to_collection("ACT_iterations", tf.reduce_sum(iterations))

        print('got through one complete timestep')
        return output, next_state

    def UnpackAndExpand(self, tensor):
        return [tf.expand_dims(x,1) for x in tf.unpack(tensor)]

    def CompressAndPack(self, list_tensor):
        return tf.pack([tf.squeeze(x) for x in list_tensor])


    def ACTStep(self,prob,counter,state,input,acc_outputs,acc_states):

        #you may need to change this to rnn.rnn depending on tensorflow versions
        output, new_state = rnn(self.cell, [input], state, scope=type(self.cell).__name__)

        with tf.variable_scope('sigmoid_activation_for_pondering'):
            p = tf.squeeze(tf.nn.rnn_cell._linear(new_state, 1, True))
            prob += p

        def use_remainder():
            remainder = 1.0 - prob
            acc_state = tf.add(tf.mul(new_state,remainder), acc_states)
            acc_output = tf.add(tf.mul(output[0], remainder), acc_outputs)
            return acc_state, acc_output

        def normal():
            acc_state = tf.add(tf.mul(new_state,p), acc_states)
            acc_output = tf.add(tf.mul(output[0], p), acc_outputs)
            return acc_state, acc_output

        acc_state, acc_output = tf.cond(tf.less(prob,self.one_minus_eps), normal, use_remainder)

        return [prob,counter + 1,new_state, input, acc_output,acc_state]

    def get_ponder_cost(self, epsilon):

        n_iterations = tf.get_collection_ref("ACT_iterations")
        remainder = tf.get_collection_ref("ACT_remainder")
        return tf.reduce_sum(tf.to_float(n_iterations) + remainder)
