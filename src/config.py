from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5     # changed from 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200 # should be 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000


  """Debugging code for models"""
  # import sys
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     m = self.lstm_model
#     n = self.rnn_model
#     o = self.linear_model
#     from numpy.random import randint
#     x = randint(0,10,[batch_size, num_steps])
#     y = randint(0,10,[batch_size, num_steps])
#
#     out = sess.run(self.train_op,
#                    {m.input_data: x,
#                     m.targets: y,
#                     m.initial_state: self.lstm_model.initial_state.eval(),
#                     n.input_data: x,
#                     n.targets: y,
#                     n.initial_state: self.rnn_model.initial_state.eval(),
#                     o.input_data: x,
#                     o.targets: y,
#                     o.initial_state: self.linear_model.initial_state.eval()})
#     print(out)
# sys.exit()

