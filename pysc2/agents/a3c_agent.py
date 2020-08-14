from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import tensorflow as tf
from pysc2.lib import actions
from pysc2.lib import features

from pysc2.agents.network import build_net
import pysc2.utils as U


class A3CAgent(object):
  """An agent specifically for solving the mini-game maps."""
  def __init__(self, training, msize, ssize, name='A3C/A3CAgent'):
    self.name = name
    self.training = training
    self.summary = []
    # Minimap size, screen size and info size
    assert msize == ssize
    self.msize = msize
    self.ssize = ssize
    #TODO calculate this automatically
    self.isize = len(actions.FUNCTIONS)+11+7+30*7+8*7+5*7+2+2*10

  def setup(self, sess, summary_writer):
    self.sess = sess
    self.summary_writer = summary_writer


  def initialize(self):
    init_op = tf.global_variables_initializer()
    self.sess.run(init_op)


  def reset(self):
    pass


  def build_model(self, reuse, dev, ntype):
    with tf.variable_scope(self.name) and tf.device(dev):
      if reuse:
        tf.get_variable_scope().reuse_variables()
        assert tf.get_variable_scope().reuse

      # Set inputs of networks
      self.minimap = tf.placeholder(tf.float32, [None, U.minimap_channel(), self.msize, self.msize], name='minimap')
      self.screen = tf.placeholder(tf.float32, [None, U.screen_channel(), self.ssize, self.ssize], name='screen')
      self.info = tf.placeholder(tf.float32, [None, self.isize], name='info')

      # Build networks
      self.outputs = build_net(self.minimap, self.screen, self.info, self.msize, self.ssize, len(actions.FUNCTIONS), ntype)
      self.spatial_action, self.non_spatial_action, self.value = self.outputs['spatial_choice'], self.outputs['action_choice'], self.outputs['value']

      # Set targets and masks

      # 가능한 비공간적 행동
      self.valid_non_spatial_action = tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)], name='valid_non_spatial_action')

      # 선택된 비공간적 행동
      self.non_spatial_action_selected = tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)], name='non_spatial_action_selected')
      self.value_target = tf.placeholder(tf.float32, [None], name='value_target')
      
      self.valid_args = tf.placeholder(tf.float32, [None, len(actions.TYPES)])
      self.valid_units=tf.placeholder(tf.float32, [None, actions.TYPES[9].sizes[0]]) #'select_unit_id'
      self.valid_queue=tf.placeholder(tf.float32, [None, actions.TYPES[11].sizes[0]]) #'build_queue_id'
      self.valid_unload=tf.placeholder(tf.float32, [None, actions.TYPES[12].sizes[0]]) #'unload_id'
    
      args_dict={}
      translations={}
      for act_type in actions.TYPES:
        if act_type.name in ['screen', 'screen2', 'minimap']:
          args_dict[act_type.id] = tf.placeholder(tf.float32, [None, self.ssize**2])
          translations[act_type.id] = 'spatial_choice'
        else:
          args_dict[act_type.id] = tf.placeholder(tf.float32, [None, act_type.sizes[0]])
          translations[act_type.id] = act_type.name

      self.args_selected = tuple(args_dict[i] for i in range(len(actions.TYPES)))
      args_prob = []

      #takes into account only valid actions
      for i in range(len(actions.TYPES)):
        
        if translations[i]=='select_unit_id':
          sel_prob=tf.reduce_sum(self.args_selected[i] * self.outputs[translations[i]], axis=1)
          valid_prob=tf.clip_by_value(tf.reduce_sum(self.valid_units * self.outputs[translations[i]], axis=1),1e-10,1)
          args_prob.append(sel_prob/valid_prob)
        elif translations[i]=='build_queue_id':
          sel_prob=tf.reduce_sum(self.args_selected[i] * self.outputs[translations[i]], axis=1)
          valid_prob=tf.clip_by_value(tf.reduce_sum(self.valid_queue * self.outputs[translations[i]], axis=1),1e-10,1)
          args_prob.append(sel_prob/valid_prob)
        elif translations[i]=='unload_id':
          sel_prob=tf.reduce_sum(self.args_selected[i] * self.outputs[translations[i]], axis=1)
          valid_prob=tf.clip_by_value(tf.reduce_sum(self.valid_unload * self.outputs[translations[i]], axis=1),1e-10,1)
          args_prob.append(sel_prob/valid_prob)
        else:
          args_prob.append(tf.reduce_sum(self.args_selected[i] * self.outputs[translations[i]], axis=1))
          
      #concatenate args_probs together
      args_prob=tf.concat([tf.expand_dims(prob,axis=1) for prob in args_prob],axis=1)
      args_log_prob = tf.log(tf.clip_by_value(args_prob, 1e-10, 1.))
      
      # Compute log probability

      non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.non_spatial_action_selected, axis=1)
      valid_non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.valid_non_spatial_action, axis=1)
      valid_non_spatial_action_prob = tf.clip_by_value(valid_non_spatial_action_prob, 1e-10, 1.)
      non_spatial_action_prob = non_spatial_action_prob / valid_non_spatial_action_prob
      non_spatial_action_log_prob = tf.log(tf.clip_by_value(non_spatial_action_prob, 1e-10, 1.))

      self.summary.append(tf.summary.histogram('non_spatial_action_prob', non_spatial_action_prob))


      # Compute losses, more details in https://arxiv.org/abs/1602.01783
      # Policy loss and value loss
      final_arg_prob= tf.reduce_sum(self.valid_args * args_log_prob, axis=1)
      action_log_prob = non_spatial_action_log_prob + final_arg_prob
      
      advantage = tf.stop_gradient(self.value_target - self.value)
      policy_loss = - tf.reduce_mean(action_log_prob * advantage)
      value_loss = - tf.reduce_mean(self.value * advantage)
      
      entropy_loss = 0
      for key in self.outputs:
        if key == 'value':
          continue
        pred = tf.clip_by_value(self.outputs[key], 1e-10, 1.)
        entropy_loss += tf.reduce_sum(pred*tf.log(pred), axis=1)
                                      
      self.summary.append(tf.summary.scalar('policy_loss', policy_loss))
      self.summary.append(tf.summary.scalar('value_loss', value_loss))

      #entropy_loss coefficient 1e-3 on original paper
      loss = policy_loss + value_loss + 1e-4*entropy_loss

      # Build the optimizer
      self.learning_rate = tf.placeholder(tf.float32, None, name='learning_rate')
      opt = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.99, epsilon=1e-10)
      grads = opt.compute_gradients(loss)
      cliped_grad = []
      for grad, var in grads:
        self.summary.append(tf.summary.histogram(var.op.name, var))
        self.summary.append(tf.summary.histogram(var.op.name+'/grad', grad))
        grad = tf.clip_by_norm(grad, 10.0)
        cliped_grad.append([grad, var])
      self.train_op = opt.apply_gradients(cliped_grad)
      self.summary_op = tf.summary.merge(self.summary)

      self.saver = tf.train.Saver(max_to_keep=5)


  def step(self, obs):
    '''for key in obs.observation:
      try:
        print(key,obs.observation[key].max(),obs.observation[key])
      except(ValueError):
        print(key,"Unable ",obs.observation[key])'''
    #could add support for rgb features
    minimap, screen, info = U.preprocess(obs.observation)

    feed = {self.minimap: minimap,
            self.screen: screen,
            self.info: info}
    outputs = self.sess.run(
       self.outputs,
      feed_dict=feed)

    # Select an action and a spatial target
    
    non_spatial_action = outputs['action_choice'].ravel()
    spatial_action = outputs['spatial_choice'].ravel()
    valid_actions = obs.observation['available_actions']
    act_id = random.choices(np.arange(len(valid_actions)),weights=non_spatial_action[valid_actions])[0]
    act_id=valid_actions[act_id]


    # Set act_id and act_args
    act_args = []
    for arg in actions.FUNCTIONS[act_id].args:
      #print(act_id,actions.FUNCTIONS[act_id].args)
      if arg.name in ('screen', 'minimap', 'screen2'):
        #moved here to make it randomize each individually, switched to weighted random
        target = random.choices(np.arange(len(spatial_action)),weights=spatial_action)[0]
        target = [int(target // self.ssize), int(target % self.ssize)]
        act_args.append([target[1], target[0]])

      else:
        max_ind=-1
        if arg.name == 'select_unit_id':
          max_ind=obs.observation['multi_select'].shape[0]
        elif arg.name == 'build_queue_id':
          max_ind=obs.observation['build_queue'].shape[0]
        elif arg.name == 'unload_id':
          max_ind=obs.observation['cargo'].shape[0]
        #print("OUTPUT SHAPE: ", np.shape(outputs[arg.name]))
        choice=random.choices(np.arange(len(outputs[arg.name][0,:max_ind])),weights=outputs[arg.name][0,:max_ind])[0]
        act_args.append([choice])
        
    return actions.FunctionCall(act_id, act_args)


  def update(self, rbs, disc, lr, cter):
    # Compute R, which is value of the last observation
    obs = rbs[-1][-1]
    if obs.last():
      R = 0
    else:
      minimap, screen, info = U.preprocess(obs.observation)

      feed = {self.minimap: minimap,
              self.screen: screen,
              self.info: info}
      R = self.sess.run(self.value, feed_dict=feed)[0]

    # Compute targets and masks
    minimaps = []
    screens = []
    infos = []

    value_target = np.zeros([len(rbs)], dtype=np.float32)
    value_target[-1] = R

    valid_non_spatial_action = np.zeros([len(rbs), len(actions.FUNCTIONS)], dtype=np.float32)
    non_spatial_action_selected = np.zeros([len(rbs), len(actions.FUNCTIONS)], dtype=np.float32)
    
    args_selected={}
    #a dict from action id to action name
    translations={}
    for act_type in actions.TYPES:
      if act_type.name in ('screen', 'minimap', 'screen2'):
        args_selected[act_type.name]=np.zeros([len(rbs),self.ssize**2], dtype=np.float32)
        translations[act_type.id]=act_type.name
      else:
        #works because all arg.sizes here are 1d
        args_selected[act_type.name]=np.zeros([len(rbs),act_type.sizes[0]], dtype=np.float32)
        translations[act_type.id]=act_type.name
      
    valid_args=np.zeros([len(rbs),len(actions.TYPES)], dtype=np.float32)
    valid_units=np.zeros([len(rbs),actions.TYPES[9].sizes[0]]) #'select_unit_id'
    valid_queue=np.zeros([len(rbs),actions.TYPES[11].sizes[0]]) #'build_queue_id'
    valid_unload=np.zeros([len(rbs),actions.TYPES[12].sizes[0]]) #'unload_id'

    rbs.reverse()
    for i, [obs, action, next_obs] in enumerate(rbs):
      minimap, screen, info = U.preprocess(obs.observation)

      minimaps.append(minimap)
      screens.append(screen)
      infos.append(info)

      reward = obs.reward
      act_id = action.function
      act_args = action.arguments

      value_target[i] = reward + disc * value_target[i-1]

      valid_actions = obs.observation["available_actions"]
      valid_non_spatial_action[i, valid_actions] = 1
      non_spatial_action_selected[i, act_id] = 1

      args = actions.FUNCTIONS[act_id].args
      
      for arg, act_arg in zip(args, act_args):
        valid_args[i,arg.id]=1 
        if arg.name in ('screen', 'minimap', 'screen2'):
          ind = act_arg[1] * self.ssize + act_arg[0]
          args_selected[arg.name][i, ind] = 1
        else:
          args_selected[arg.name][i,act_arg]=1

        #keep track of valid options
        if arg.name == 'select_unit_id':
          max_ind=obs.observation['multi_select'].shape[0]
          valid_units[i,:max_ind]=1
        if arg.name == 'build_queue_id':
          max_ind=obs.observation['build_queue'].shape[0]
          valid_queue[i,:max_ind]=1
        if arg.name == 'unload_id':
          max_ind=obs.observation['cargo'].shape[0]
          valid_unload[i,:max_ind]=1
    
    minimaps = np.concatenate(minimaps, axis=0)
    screens = np.concatenate(screens, axis=0)
    infos = np.concatenate(infos, axis=0)

    # Train
    feed = {self.minimap: minimaps,
            self.screen: screens,
            self.info: infos,
            self.value_target: value_target,
            self.valid_non_spatial_action: valid_non_spatial_action,
            self.non_spatial_action_selected: non_spatial_action_selected,
            self.learning_rate: lr,
            self.valid_args: valid_args,
            self.args_selected: tuple(args_selected[translations[i]] for i in range(len(actions.TYPES))),
            self.valid_units: valid_units,
            self.valid_queue: valid_queue,
            self.valid_unload: valid_unload
            }
    
    _, summary = self.sess.run([self.train_op, self.summary_op], feed_dict=feed)
    self.summary_writer.add_summary(summary, cter)


  def save_model(self, path, count):
    self.saver.save(self.sess, path+'/model.pkl', count)


  def load_model(self, path):
    ckpt = tf.train.get_checkpoint_state(path)
    self.saver.restore(self.sess, ckpt.model_checkpoint_path)
    return int(ckpt.model_checkpoint_path.split('-')[-1])
