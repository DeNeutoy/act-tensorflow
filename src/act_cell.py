#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.contrib.rnn import static_rnn
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl


class ACTCell(RNNCell):
    """
    A RNN cell implementing Graves' Adaptive Computation Time algorithm
    """
    def __init__(self, num_units, cell, epsilon,
                 max_computation, batch_size, sigmoid_output=False):

        self.batch_size = batch_size
        self.one_minus_eps = tf.fill([self.batch_size], tf.constant(1.0 - epsilon, dtype=tf.float32))
        self._num_units = num_units
        self.cell = cell
        self.max_computation = max_computation
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

    def __call__(self, inputs, state, timestep=0, scope=None):

        self.state_is_tuple = False
        if isinstance(state, (tuple, list)):
            state = tf.concat(state, 1)
            self.state_is_tuple = True

        with vs.variable_scope(scope or type(self).__name__):
            # define within cell constants/ counters used to control while loop for ACTStep
            prob = tf.fill([self.batch_size], tf.constant(0.0, dtype=tf.float32), "prob")
            prob_compare = tf.zeros_like(prob, tf.float32, name="prob_compare")
            counter = tf.zeros_like(prob, tf.float32, name="counter")
            acc_outputs = tf.fill([self.batch_size, self.output_size], 0.0, name='output_accumulator')
            acc_states = tf.zeros_like(state, tf.float32, name="state_accumulator")
            batch_mask = tf.fill([self.batch_size], True, name="batch_mask")


            # While loop stops when this predicate is FALSE.
            # Ie all (probability < 1-eps AND counter < N) are false.
            def halting_predicate(batch_mask, prob_compare, prob,
                          counter, state, input, acc_output, acc_state):
                return tf.reduce_any(tf.logical_and(
                        tf.less(prob_compare,self.one_minus_eps),
                        tf.less(counter, self.max_computation)))

            # Do while loop iterations until predicate above is false.
            _,_,remainders,iterations,_,_,output,next_state = \
                tf.while_loop(halting_predicate, self.act_step,
                              loop_vars=[batch_mask, prob_compare, prob,
                                         counter, state, inputs, acc_outputs, acc_states])

        #accumulate remainder  and N values
        self.ACT_remainder.append(tf.reduce_mean(1 - remainders))
        self.ACT_iterations.append(tf.reduce_mean(iterations))

        if self.sigmoid_output:
            output = tf.sigmoid(tf.contrib.rnn.BasicRNNCell._linear(output,self.batch_size,0.0))

        if self.state_is_tuple:
            next_c, next_h = tf.split(next_state, 2, 1)
            next_state = tf.contrib.rnn.LSTMStateTuple(next_c, next_h)

        return output, next_state

    def calculate_ponder_cost(self, time_penalty):
        '''returns tensor of shape [1] which is the total ponder cost'''
        return time_penalty * tf.reduce_sum(
            tf.add_n(self.ACT_remainder)/len(self.ACT_remainder) +
            tf.to_float(tf.add_n(self.ACT_iterations)/len(self.ACT_iterations)))

    def act_step(self,batch_mask,prob_compare,prob,counter,state,input,acc_outputs,acc_states):
        '''
        General idea: generate halting probabilites and accumulate them. Stop when the accumulated probs
        reach a halting value, 1-eps. At each timestep, multiply the prob with the rnn output/state.
        There is a subtlety here regarding the batch_size, as clearly we will have examples halting
        at different points in the batch. This is dealt with using logical masks to protect accumulated
        probabilities, states and outputs from a timestep t's contribution if they have already reached
        1 - es at a timstep s < t. On the last timestep for each element in the batch the remainder is
        multiplied with the state/output, having been accumulated over the timesteps, as this takes
        into account the epsilon value.
        '''

        # If all the probs are zero, we are seeing a new input => binary flag := 1, else 0.
        binary_flag = tf.cond(tf.reduce_all(tf.equal(prob, 0.0)),
                              lambda: tf.ones([self.batch_size, 1], dtype=tf.float32),
                              lambda: tf.zeros([self.batch_size, 1], tf.float32))

        input_with_flags = tf.concat([binary_flag, input], 1)

        if self.state_is_tuple:
            (c, h) = tf.split(state, 2, 1)
            state = tf.contrib.rnn.LSTMStateTuple(c, h)

        output, new_state = static_rnn(cell=self.cell, inputs=[input_with_flags], initial_state=state, scope=type(self.cell).__name__)

        if self.state_is_tuple:
            new_state = tf.concat(new_state, 1)

        with tf.variable_scope('sigmoid_activation_for_pondering'):
            p = tf.squeeze(tf.layers.dense(new_state, 1, activation=tf.sigmoid, use_bias=True), squeeze_dims=1)

        # Multiply by the previous mask as if we stopped before, we don't want to start again
        # if we generate a p less than p_t-1 for a given example.
        new_batch_mask = tf.logical_and(tf.less(prob + p, self.one_minus_eps), batch_mask)
        new_float_mask = tf.cast(new_batch_mask, tf.float32)

        # Only increase the prob accumulator for the examples
        # which haven't already passed the threshold. This
        # means that we can just use the final prob value per
        # example to determine the remainder.
        prob += p * new_float_mask

        # This accumulator is used solely in the While loop condition.
        # we multiply by the PREVIOUS batch mask, to capture probabilities
        # that have gone over 1-eps THIS iteration.
        prob_compare += p * tf.cast(batch_mask, tf.float32)

        # Only increase the counter for those probabilities that
        # did not go over 1-eps in this iteration.
        counter += new_float_mask

        # Halting condition (halts, and uses the remainder when this is FALSE):
        # If any batch element still has both a prob < 1 - epsilon AND counter < N we
        # continue, using the outputed probability p.
        counter_condition = tf.less(counter, self.max_computation)

        final_iteration_condition = tf.logical_and(new_batch_mask, counter_condition)
        use_remainder = tf.expand_dims(1.0 - prob, -1)
        use_probability = tf.expand_dims(p, -1)
        update_weight = tf.where(final_iteration_condition, use_probability, use_remainder)
        float_mask = tf.expand_dims(tf.cast(batch_mask, tf.float32), -1)

        acc_state = (new_state * update_weight * float_mask) + acc_states
        acc_output = (output[0] * update_weight * float_mask) + acc_outputs

        return [new_batch_mask, prob_compare, prob, counter, new_state, input, acc_output, acc_state]