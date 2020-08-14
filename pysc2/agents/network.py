from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.layers as layers


def build_net(minimap, screen, info, msize, ssize, num_action, ntype):
  if ntype == 'atari':
    return build_atari(minimap, screen, info, msize, ssize, num_action)
  elif ntype == 'fcn':
    return build_fcn(minimap, screen, info, msize, ssize, num_action)
  else:
    raise print('FLAGS.net must be atari or fcn')


def build_atari(minimap, screen, info, msize, ssize, num_action):
  #raise Exception("Atari net not supported at the moment")
  mconv1 = layers.conv2d(tf.transpose(minimap, [0, 2, 3, 1]),
                         num_outputs=16,
                         kernel_size=8,
                         stride=4,
                         scope='mconv1')
  mconv2 = layers.conv2d(mconv1,
                         num_outputs=32,
                         kernel_size=4,
                         stride=2,
                         scope='mconv2')
  sconv1 = layers.conv2d(tf.transpose(screen, [0, 2, 3, 1]),
                         num_outputs=16,
                         kernel_size=8,
                         stride=4,
                         scope='sconv1')
  sconv2 = layers.conv2d(sconv1,
                         num_outputs=32,
                         kernel_size=4,
                         stride=2,
                         scope='sconv2')
  info_fc = layers.fully_connected(layers.flatten(info),
                                   num_outputs=256,
                                   activation_fn=tf.tanh,
                                   scope='info_fc')

  # Compute spatial actions, non spatial actions and value
  feat_fc = tf.concat([layers.flatten(mconv2), layers.flatten(sconv2), info_fc], axis=1)
  feat_fc = layers.fully_connected(feat_fc,
                                   num_outputs=256,
                                   activation_fn=tf.nn.relu,
                                   scope='feat_fc')

  spatial_action_x = layers.fully_connected(feat_fc,
                                            num_outputs=ssize,
                                            activation_fn=tf.nn.softmax,
                                            scope='spatial_action_x')
  spatial_action_y = layers.fully_connected(feat_fc,
                                            num_outputs=ssize,
                                            activation_fn=tf.nn.softmax,
                                            scope='spatial_action_y')
  spatial_action_x = tf.reshape(spatial_action_x, [-1, 1, ssize])
  spatial_action_x = tf.tile(spatial_action_x, [1, ssize, 1])
  spatial_action_y = tf.reshape(spatial_action_y, [-1, ssize, 1])
  spatial_action_y = tf.tile(spatial_action_y, [1, 1, ssize])
  spatial_action = layers.flatten(spatial_action_x * spatial_action_y)

  non_spatial_action = layers.fully_connected(feat_fc,
                                              num_outputs=num_action,
                                              activation_fn=tf.nn.softmax,
                                              scope='non_spatial_action')
  value = tf.reshape(layers.fully_connected(feat_fc,
                                            num_outputs=1,
                                            activation_fn=None,
                                            scope='value'), [-1])

  return spatial_action, non_spatial_action, value



def build_fcn(minimap, screen, info, msize, ssize, num_action):
  # Extract features
  mconv1 = layers.conv2d(tf.transpose(minimap, [0, 2, 3, 1]),
                         num_outputs=16,
                         kernel_size=5,
                         stride=1,
                         scope='mconv1')
  mconv2 = layers.conv2d(mconv1,
                         num_outputs=32,
                         kernel_size=3,
                         stride=1,
                         scope='mconv2')
  sconv1 = layers.conv2d(tf.transpose(screen, [0, 2, 3, 1]),
                         num_outputs=16,
                         kernel_size=5,
                         stride=1,
                         scope='sconv1')
  sconv2 = layers.conv2d(sconv1,
                         num_outputs=32,
                         kernel_size=3,
                         stride=1,
                         scope='sconv2')
  info_fc = layers.fully_connected(layers.flatten(info),
                                   num_outputs=256,
                                   activation_fn=tf.tanh,
                                   scope='info_fc')

  #made this a fully connected since details aren't very clear on original paper
  info_projection = layers.fully_connected(layers.flatten(info),
                                   num_outputs=ssize**2,
                                   activation_fn=tf.tanh,
                                   scope='info_projection')
  
  # Compute spatial actions
  feat_conv = tf.concat([mconv2, sconv2, tf.reshape(info_projection, [-1,64,64,1])]
                        , axis=3)
  spatial_choice = layers.conv2d(feat_conv,
                                 num_outputs=1,
                                 kernel_size=1,
                                 stride=1,
                                 activation_fn=None,
                                 scope='spatial_choice')
  outputs={}
  outputs['spatial_choice'] = tf.nn.softmax(layers.flatten(spatial_choice))

  # Compute non spatial actions and value
  feat_fc = tf.concat([layers.flatten(mconv2), layers.flatten(sconv2), info_fc], axis=1)
  feat_fc = layers.fully_connected(feat_fc,
                                   num_outputs=256,
                                   activation_fn=tf.nn.relu,
                                   scope='feat_fc')
  
  outputs['action_choice'] = layers.fully_connected(feat_fc,
                                              num_outputs=num_action,
                                              activation_fn=tf.nn.softmax,
                                              scope='action_choice')
  outputs['queued'] = layers.fully_connected(feat_fc,
                                      num_outputs=2,
                                      activation_fn=tf.nn.softmax,
                                      scope='queued')
  outputs['control_group_act'] = layers.fully_connected(feat_fc,
                                      num_outputs=5,
                                      activation_fn=tf.nn.softmax,
                                      scope='control_group_act')
  outputs['control_group_id'] = layers.fully_connected(feat_fc,
                                      num_outputs=10,
                                      activation_fn=tf.nn.softmax,
                                      scope='control_group_id')
  outputs['select_point_act'] = layers.fully_connected(feat_fc,
                                      num_outputs=4,
                                      activation_fn=tf.nn.softmax,
                                      scope='select_point_act')
  outputs['select_add'] = layers.fully_connected(feat_fc,
                                      num_outputs=2,
                                      activation_fn=tf.nn.softmax,
                                      scope='select_add')
  outputs['select_unit_act'] = layers.fully_connected(feat_fc,
                                      num_outputs=4,
                                      activation_fn=tf.nn.softmax,
                                      scope='select_unit_act')
  outputs['select_unit_id'] = layers.fully_connected(feat_fc,
                                      num_outputs=500,
                                      activation_fn=tf.nn.softmax,
                                      scope='select_unit_id')
  outputs['select_worker'] = layers.fully_connected(feat_fc,
                                      num_outputs=4,
                                      activation_fn=tf.nn.softmax,
                                      scope='select_worker')
  outputs['build_queue_id'] = layers.fully_connected(feat_fc,
                                      num_outputs=10,
                                      activation_fn=tf.nn.softmax,
                                      scope='build_queue_id')
  outputs['unload_id'] = layers.fully_connected(feat_fc,
                                      num_outputs=500,
                                      activation_fn=tf.nn.softmax,
                                      scope='unload_id')
  outputs['value'] = tf.reshape(layers.fully_connected(feat_fc,
                                            num_outputs=1,
                                            activation_fn=None,
                                            scope='value'), [-1])

  return outputs
