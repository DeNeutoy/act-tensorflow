#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import pickle, os, sys
import tensorflow as tf
from tensorflow.python.ops.nn import rnn_cell, rnn, seq2seq
from tensorflow.python.ops import control_flow_ops, gen_math_ops
from tensorflow.python.ops import variable_scope as vs


class ACTCell(rnn_cell.RNNCell):

    """An RNN cell implementing Graves' Adaptive Computation Time algorithm"""


    def __init__(self, num_units, cell, epsilon, max_computation, batch_size, sigmoid_output=False):

        self.batch_size = batch_size
        self.one_minus_eps = tf.constant(1.0 - epsilon, tf.float32,[self.batch_size])
        self._num_units = num_units
        self.cell = cell
        self.N = tf.constant(max_computation, tf.float32,[self.batch_size])
        self.ACT_remainder = []
        self.ACT_iterations = []
        self.sigmoid_output = sigmoid_output

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

            # define within cell constants/ counters used to control while loop for ACTStep
            prob = tf.constant(0.0,tf.float32,[self.batch_size], name="prob")
            prob_compare = tf.constant(0.0,tf.float32,[self.batch_size], name="prob_compare")
            counter = tf.constant(0.0, tf.float32,[self.batch_size], name="counter")
            acc_outputs = tf.zeros_like(state, tf.float32, name="output_accumulator")
            acc_states = tf.zeros_like(state, tf.float32, name="state_accumulator")
            batch_mask = tf.constant(True, tf.bool,[self.batch_size])

            # While loop stops when this predicate is FALSE.
            # Ie all (probability < 1-eps AND counter < N) are false.

            #x = self.ACTStep(batch_mask,prob_compare,prob,counter,state,inputs,acc_outputs,acc_states)


            pred = lambda batch_mask,prob_compare,prob,\
                          counter,state,input,acc_output,acc_state:\
                tf.reduce_any(
                    tf.logical_and(
                        tf.less(prob_compare,self.one_minus_eps),
                        tf.less(counter,self.N)))
                # only stop if all of the batch have passed either threshold

            # Do while loop iterations until predicate above is false.
            _,_,remainders,iterations,_,_,output,next_state = \
                control_flow_ops.while_loop(pred,self.ACTStep,
                [batch_mask,prob_compare,prob,
                 counter,state,inputs, acc_outputs, acc_states])

        #accumulate remainder  and N values
        self.ACT_remainder.append(tf.reduce_mean(1 - remainders))
        self.ACT_iterations.append(tf.reduce_mean(iterations))

        if self.sigmoid_output:
            output = tf.sigmoid(tf.nn.rnn_cell._linear(output,self.batch_size,0.0))

        return output, next_state



    def CalculatePonderCost(self, time_penalty):
        '''returns tensor of shape [1] which is the total ponder cost'''

        return time_penalty * tf.reduce_sum(
            tf.add_n(self.ACT_remainder)/len(self.ACT_remainder) +
            tf.to_float(tf.add_n(self.ACT_iterations)/len(self.ACT_iterations)))

    def ACTStep(self,batch_mask,prob_compare,prob,counter,state,input,acc_outputs,acc_states):

        # General idea: generate halting probabilites and accumulate them. Stop when the accumulated probs
        # reach a halting value, 1-eps. At each timestep, multiply the prob with the rnn output/state.
        # There is a subtlety here regarding the batch_size, as clearly we will have examples halting
        # at different points in the batch. This is dealt with using logical masks to protect accumulated
        # probabilities, states and outputs from a timestep t's contribution if they have already reached
        # 1-es at a timstep s < t. On the last timestep, the remainder of every example in the batch is
        # multiplied with the state/output, having been accumulated over the timesteps and correctly carried
        # through for all examples, regardless of #overall batch timesteps.


        # if all the probs are zero, we are seeing a new input => binary flag := 1, else 0.
        binary_flag = tf.cond(tf.reduce_all(tf.equal(prob,0.0)),
                              lambda: tf.ones([self.batch_size,1],dtype=tf.float32),
                              lambda: tf.zeros([self.batch_size,1],tf.float32))

        input_with_flags = tf.concat(1, [binary_flag,input])
        output, new_state = rnn(self.cell, [input_with_flags], state, scope=type(self.cell).__name__)


        with tf.variable_scope('sigmoid_activation_for_pondering'):
            p = tf.squeeze(tf.sigmoid(tf.nn.rnn_cell._linear(new_state, 1, True)))

        # multiply by the previous mask as if we stopped before, we don't want to start again
        # if we generate a p less than p_t-1 for a given example.
        new_batch_mask = tf.logical_and(tf.less(prob + p,self.one_minus_eps),batch_mask)

        new_float_mask = tf.cast(new_batch_mask, tf.float32)


        # only increase the prob accumulator for the examples
        # which haven't already passed the threshold. This
        # means that we can just use the final prob value per
        # example to determine the remainder.
        prob += p * new_float_mask

        # this accumulator is used solely in the While loop condition.
        # we multiply by the PREVIOUS batch mask, to capture probabilities
        # that have gone over 1-eps THIS iteration.
        prob_compare += p * tf.cast(batch_mask, tf.float32)

        def use_remainder():

            # runs on the last iteration of while loop. prob now contains
            # exactly the probability at N-1, ie the timestep before we
            # go over 1-eps for all elements of the batch.

            remainder = tf.constant(1.0, tf.float32,[self.batch_size]) - prob
            remainder_expanded = tf.expand_dims(remainder,1)
            tiled_remainder = tf.tile(remainder_expanded,[1,self.output_size])

            acc_state = (new_state * tiled_remainder) + acc_states
            acc_output = (output[0] * tiled_remainder) + acc_outputs
            return acc_state, acc_output

        def normal():

            # accumulate normally, by multiplying the batch
            # probs with the output and state of the rnn.
            # If we passed the 1-eps threshold this round, we
            # have a zero in the batch mask, so we add no contribution
            # to acc_state or acc_output

            p_expanded = tf.expand_dims(p * new_float_mask,1)
            tiled_p = tf.tile(p_expanded,[1,self.output_size])

            acc_state = (new_state * tiled_p) + acc_states
            acc_output = (output[0] * tiled_p) + acc_outputs
            return acc_state, acc_output

        # only increase the counter for those probabilities that
        # did not go over 1-eps in this iteration.
        counter += tf.constant(1.0,tf.float32,[self.batch_size]) * new_float_mask


        # halting condition(halts, and uses the remainder when this is FALSE):
        # if the batch mask is all zeros, then all batches have finished.
        # if any batch element still has both a prob < 1-eps AND counter< N we continue.

        counter_condition = tf.less(counter,self.N)
        condition = tf.reduce_any(tf.logical_and(new_batch_mask,counter_condition))

        acc_state, acc_output = tf.cond(condition, normal, use_remainder)


        return [new_batch_mask,prob_compare,prob,counter,new_state, input, acc_output,acc_state]
