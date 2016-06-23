#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import pickle, os, sys
import tensorflow as tf
try:
    from tensorflow.python.ops.nn import rnn_cell, rnn, seq2seq
except:
    sys.path.insert(0, os.environ['HOME'])
    from Project_RNN_Enhancement.rnn_enhancement import rnn_cell_enhanced as rnn_cell
    from Project_RNN_Enhancement.rnn_enhancement import rnn_enhanced as rnn
    from Project_RNN_Enhancement.rnn_enhancement import victory_music

    seq2seq = tf.nn.seq2seq
from tensorflow.python.ops import control_flow_ops, gen_math_ops
from tensorflow.python.ops import variable_scope as vs




class ACTCellMasking(rnn_cell.RNNCell):

    """An RNN cell implementing Graves' Adaptive Computation Time algorithm"""


    def __init__(self, num_units, cell, epsilon, max_computation, batch_size):

        self.batch_size = batch_size
        self.one_minus_eps = tf.constant(1.0 - epsilon, tf.float32,[self.batch_size])
        self._num_units = num_units
        self.cell = cell
        self.N = tf.constant(max_computation, tf.float32,[self.batch_size])
        self.ACT_remainder = []
        self.ACT_iterations = []

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

        with vs.variable_scope(scope or type(self).__name__):

            # define within cell constants/ counters used to control while loop
            prob = tf.constant(0.0,tf.float32,[self.batch_size], name="prob")
            counter = tf.constant(0.0, tf.float32,[self.batch_size], name="counter")
            acc_outputs = tf.zeros_like(state, tf.float32, name="output_accumulator")
            acc_states = tf.zeros_like(state, tf.float32, name="state_accumulator")


            # the predicate for stopping the while loop. Tensorflow demands that we have
            # all of the variables used in the while loop in the predicate.
            pred = lambda prob,counter,state,input,acc_output,acc_state:tf.logical_and(
                #
                gen_math_ops._all(tf.less(prob,self.one_minus_eps),0),
                               # only stop if all of the batch have passed either threshold
                               gen_math_ops._all(tf.less(counter,self.N),0))

            prob,iterations,_,_,output,next_state = control_flow_ops.while_loop(
                pred,self.ACTStep,[prob,counter,state,inputs, acc_outputs, acc_states])


        '''Calculate ponder cost parts. Reduce mean is used to normalize cost by the batch size'''
        self.ACT_remainder.append(tf.reduce_mean(1 - prob)) #TODO: double check this
        self.ACT_iterations.append(tf.reduce_mean(iterations))

        print('got through one complete timestep in variable batch masking')
        return output, next_state

    def CalculatePonderCost(self, time_penalty):
        '''returns tensor of shape [1] which is the total ponder cost'''

        return time_penalty * tf.reduce_sum(
            tf.add_n(self.ACT_remainder) + tf.to_float(tf.add_n(self.ACT_iterations)))

    def ACTStep(self,prob,counter,state,input,acc_outputs,acc_states):

        # input is now [batch_size, hidden_size]
        output, new_state = rnn(self.cell, [input], state, scope=type(self.cell).__name__)

        with tf.variable_scope('sigmoid_activation_for_pondering'):
            p = tf.squeeze(tf.sigmoid(tf.nn.rnn_cell._linear(new_state, 1, True)))

        # here we create a mask on the p vector, which we then multiply with the state/output.
        # if p[i] = 0, then we have passed the remainder point for that example, so we multiply
        # the state/output vector by this masked probability(which has zeros if the prob for
        # a batch has passed the stopping point) so we carry none of it forward.
        #  If, by adding p, we pass the boundary, we don't add p onto prob - this allows us to
        # use the use_remainder() as normal for all steps after ALL examples have taken their max time.
        batch_mask = tf.cast(tf.less(prob + p,self.one_minus_eps), tf.float32)
        p_masked = tf.mul(p, batch_mask)

        # only increase the prob accumulator for the examples
        # which haven't already passed the threshold. This
        # means that we can just use the final prob value per
        # example to determine the remainder.
        prob += p_masked

        def use_remainder():
            remainder = tf.constant(1.0, tf.float32,[self.batch_size]) - prob
            remainder_expanded = tf.expand_dims(remainder,1)
            tiled_remainder = tf.tile(remainder_expanded,[1,self.output_size])

            acc_state = tf.add(tf.mul(new_state,tiled_remainder), acc_states)
            acc_output = tf.add(tf.mul(output[0], tiled_remainder), acc_outputs)
            return acc_state, acc_output

        def normal():

            p_expanded = tf.expand_dims(p_masked,1)
            tiled_p = tf.tile(p_expanded,[1,self.output_size])

            acc_state = tf.add(tf.mul(new_state,tiled_p), acc_states)
            acc_output = tf.add(tf.mul(output[0], tiled_p), acc_outputs)
            return acc_state, acc_output

        # halting condition: if the batch mask is all zeros, then all batches have finished.
        # therefore, if the sum of the mask = 0, then we use the remainder.

        acc_state, acc_output = tf.cond(tf.equal(tf.reduce_sum(batch_mask),0.0), use_remainder, normal)

        # only increment the counter for the examples which are still running
        counter += tf.mul(tf.constant(1.0,tf.float32,[self.batch_size]),batch_mask)

        return [prob,counter,new_state, input, acc_output,acc_state]

