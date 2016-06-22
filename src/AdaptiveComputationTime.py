from __future__ import division
from __future__ import print_function

import os
import pickle, os, sys
from datetime import datetime
from collections import defaultdict
import tensorflow as tf
from epoch import run_epoch
from config import SmallConfig
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
        self.batch_size = batch_size = 1
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
            # act = inner_cell
            act = ACTCell_TensorArray(self.config.hidden_size, inner_cell, 0.01, 10)

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


        self.cost = tf.reduce_sum(loss) / batch_size #+ act.get_ponder_cost(0.01)
        self.final_state = self.outputs[-1]

        if is_training:

            self.lr = tf.Variable(0.0, trainable=False)

            tvars = tf.trainable_variables()  # all params inside model which are trainable
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.max_grad_norm)

            optimizer = tf.train.GradientDescentOptimizer(self.lr)
            #optimizer = tf.train.AdamOptimizer(self.lr)

            self.train_op = optimizer.apply_gradients(zip(grads, tvars))




class ACTCell(rnn_cell.RNNCell):

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

        with vs.variable_scope(scope or type(self).__name__):

            # define within cell constants/ counters used to control while loop
            prob = tf.constant(0.0,tf.float32, name="prob")
            counter = tf.constant(0, dtype=tf.int32, name="counter")
            acc_probs, acc_outputs, acc_states = [],[],[]

            # the predicate for stopping the while loop. Tensorflow demands that we have
            # all of the variables used in the while loop in the predicate.
            pred = lambda prob,counter,state,input,acc_output,acc_state,acc_probs:\
                tf.logical_and(tf.less(prob,self.one_minus_eps), tf.less(counter,self.N))

            _,iterations,_,_,acc_outputs,acc_states,acc_probs = control_flow_ops.while_loop(

                pred,self.ACTStep,[prob,counter,state,inputs, acc_outputs, acc_states, acc_probs])


        # TODO:fix last part of this, need to use the remainder.
        # TODO: find a way to accumulate the regulariser

        # outputs = acc_outputs.pack()
        # probs = acc_probs.pack()
        # states = acc_states.pack()

        #outputs.set_shape([self.count_its,self.output_size])
        #probs.set_shape([self.count_its,])
        #states.set_shape([self.count_its,self.output_size])

        # acc_probs = tf.split(0,iterations,probs)
        # acc_outputs = tf.split(0,iterations,outputs)
        # acc_states = tf.split(0,iterations,states)
        #
        # for p,o,s in zip(acc_probs,acc_outputs, acc_states):
        #     p.set_shape([])
        #     o.set_shape([1,self.output_size])
        #     s.set_shape([1, self.output_size])

        remainder = 1.0 - tf.add_n(acc_probs[:-1])
        acc_probs[-1] = remainder

        next_state = tf.add_n([tf.mul(x,y) for x,y in zip(acc_probs,acc_states)])
        output = tf.add_n([tf.mul(x,y) for x,y in zip(acc_probs,acc_outputs)])

        tf.add_to_collection("ACT_remainder", remainder)
        tf.add_to_collection("ACT_iterations", iterations)

        return output, next_state

    def ACTStep(self,prob,counter,state,input,acc_outputs,acc_states,acc_probs):

        output, new_state = rnn(self.cell, [input], state, scope=type(self.cell).__name__)

        prob_w = tf.get_variable("prob_w", [self.cell.state_size,1], trainable = False) #corrected to state_size
        prob_b = tf.get_variable("prob_b", [1], trainable = False)
        p = tf.squeeze(tf.nn.sigmoid(tf.matmul(new_state,prob_w) + prob_b))

        acc_outputs.append(output[0])
        acc_states.append(new_state)
        acc_probs.append(p)


        return [prob + p,counter + 1,new_state, input, acc_outputs,acc_states,acc_probs]

    def get_ponder_cost(self, epsilon):

        n_iterations = tf.get_collection_ref("ACT_iterations")
        remainder = tf.get_collection_ref("ACT_remainder")
        return tf.reduce_sum(n_iterations + remainder)


class ACTCell_TensorArray(rnn_cell.RNNCell):

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
            acc_probs, acc_outputs, acc_states = [],[],[]


            #dynamic set to true because we don't know how many N iterations will occur
            acc_probs = TensorArray(dtype = tf.float32, size = 1, dynamic_size = True, clear_after_read = False,
                tensor_array_name = "accumulated_probabilities_{}".format(timestep), infer_shape = True)
            acc_outputs = TensorArray(dtype = tf.float32, size = 1, dynamic_size = True, clear_after_read = False,
                tensor_array_name = "accumulated_outputs_{}".format(timestep), infer_shape = True)
            acc_states = TensorArray(dtype = tf.float32, size = 1, dynamic_size = True, clear_after_read = False,
                tensor_array_name = "accumulated_states_{}".format(timestep), infer_shape = True)

            # the predicate for stopping the while loop. Tensorflow demands that we have
            # all of the variables used in the while loop in the predicate.
            pred = lambda prob,counter,state,input,acc_output,acc_state,acc_probs:\
                tf.logical_and(tf.less(prob,self.one_minus_eps), tf.less(counter,self.N))

            _,iterations,_,_,acc_outputs,acc_states,acc_probs = control_flow_ops.while_loop(
                
                pred,self.ACTStep,[prob,counter,state,inputs, acc_outputs, acc_states, acc_probs])

        #nick as a reminder, you can pack the tensorarray to convert it to a regular tensor
        remainder = 1.0 - acc_probs.read(iterations)
        acc_probs.write(iterations, remainder) #not sure if this command is correct

        acc_probs = acc_probs.pack()
        acc_states = acc_states.pack()
        acc_outputs = acc_outputs.pack()

        # expand the dimensions of acc_probs to (1,iterations,1) so the broadcasting works
        next_state = tf.reduce_sum(tf.expand_dims(tf.expand_dims(acc_probs,0),2) * acc_states, reduction_indices = 0) #check reduction indices
        output = tf.reduce_sum(tf.expand_dims(tf.expand_dims(acc_probs,0),2) * acc_outputs, reduction_indices = 0)

        next_state.set_shape([None, self.cell.state_size])
        output.set_shape([None, self.cell.state_size])

        tf.add_to_collection("ACT_remainder", remainder)
        tf.add_to_collection("ACT_iterations", iterations)

        print('got through one complete timestep')
        return output, next_state

    def ACTStep(self,prob,counter,state,input,acc_outputs,acc_states,acc_probs):

        #you may need to change this to rnn.rnn depending on tensorflow versions
        output, new_state = rnn(self.cell, [input], state, scope=type(self.cell).__name__)

        with tf.variable_scope('sigmoid_activation_for_pondering'):
            # prob_w = tf.get_variable("prob_w", [self.cell.state_size,1]) #corrected to state_size
            # prob_b = tf.get_variable("prob_b", [1])
            # p = tf.squeeze(tf.nn.sigmoid(tf.matmul(new_state,prob_w) + prob_b))
            # p = tf.nn.rnn_cell._linear(new_state[0], 1, True)
            p = tf.squeeze(tf.nn.rnn_cell._linear(new_state, 1, True))


        acc_outputs = acc_outputs.write(counter, output[0])
        acc_states = acc_states.write(counter, new_state)
        acc_probs = acc_probs.write(counter, p)

        return [prob + p,counter + 1,new_state, input, acc_outputs,acc_states,acc_probs]

    def get_ponder_cost(self, epsilon):

        n_iterations = tf.get_collection_ref("ACT_iterations")
        remainder = tf.get_collection_ref("ACT_remainder")
        return tf.reduce_sum(n_iterations + remainder)