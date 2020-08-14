from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pysc2.lib import actions
from pysc2.lib import features


'''got rid of expanding cateegorical features in the channel dimension, since that's
not what the original paper did, instead use the numerical value of the category'''

def preprocess_minimap(minimap):
  layers = []
  assert minimap.shape[0] == len(features.MINIMAP_FEATURES)
  for i in range(len(features.MINIMAP_FEATURES)):
    if features.MINIMAP_FEATURES[i].type == features.FeatureType.SCALAR:
      #convert to logarithmic, have to add 1 and take max to avoid high negative values
      layers.append(np.log(np.maximum(minimap[i:i+1]+1,1)))
    else:
      layers.append(minimap[i:i+1])
  return np.concatenate(layers, axis=0)


def preprocess_screen(screen):
  layers = []
  assert screen.shape[0] == len(features.SCREEN_FEATURES)
  for i in range(len(features.SCREEN_FEATURES)):
    
    if features.SCREEN_FEATURES[i].type == features.FeatureType.SCALAR:
      layers.append(np.log(np.maximum(screen[i:i+1]+1,1)))
    else:
      
      layers.append(screen[i:i+1])
  return np.concatenate(layers, axis=0)

def preprocess_structured(data,key):
  '''preprocesses structured array "data", key is the name of that data in observation'''
  #TODO maybe add last_actions, action_result, cargo_slots_available
  if key=="player": #(11,)
    for i in range(len(data)):
      player_id=features.Player['player_id']
      if i!=player_id:
        data[i]=np.log(np.maximum(data[i]+1,1))
    return data

  if key=="single_select": #(1,7)
    return get_unit_infos(data,1)
  
  #only show info for at most 30 selected units, actual cap 255
  if key=="multi_select": #(n,7)
    return get_unit_infos(data,30)

  #only show for first 8 since thats the most common case, can be much higher in nudys
  if key=="cargo": #(n,7)
    return get_unit_infos(data,8)

  #only show for first 5, pretty sure thats hard cap
  if key=="build_queue": #(n,7)
    return get_unit_infos(data,5)

  if key=="alerts": #(n,) at most 2
    return np.pad(data,[(0,2-len(data))],'constant',constant_values=0)

  if key=="control_groups": #(10,2)
    non_scalar=[features.UnitCounts['unit_type']]
    for i in range(len(data[0])):
      if i not in non_scalar:
        data[:,i]=np.log(np.maximum(data[:,i]+1,1))
    return data.reshape([-1])

def get_unit_infos(data,max_units):
  '''helper func for observations that are similar to single_select
  max_units=maximum amount of spots reserved for different unit types'''
  non_scalar=[features.UnitLayer['unit_type'],features.UnitLayer['player_relative']]
  num_units=np.shape(data)[0]
  if num_units>=max_units:
    data=data[:max_units] 
  else:
    data=np.pad(data,[(0,max_units-num_units),(0,0)],'constant',constant_values=0)
    
  for i in range(len(data[0])):
    if i not in non_scalar:
      data[:,i]=np.log(np.maximum(data[:,i]+1,1))
  return data.reshape([-1])

def preprocess(observation):
  '''
  Preprocess observation object
  
  Returns:
  (minimap,screen,info) tuple of preprocessed np arrays
  '''
  #could add support for rgb features
  minimap = np.array(observation['feature_minimap'], dtype=np.float32)  
  minimap = np.expand_dims(preprocess_minimap(minimap), axis=0)
  screen = np.array(observation['feature_screen'], dtype=np.float32)
  screen = np.expand_dims(preprocess_screen(screen), axis=0)

  info=[]
  acts_info = np.zeros([len(actions.FUNCTIONS)], dtype=np.float32)
  acts_info[observation['available_actions']] = 1
  info.append(acts_info)
  keys=('player','single_select','multi_select','cargo',
        'build_queue','alerts','control_groups')
  for key in keys:
    info.append(preprocess_structured(observation[key],key))
    
  info = np.expand_dims(np.concatenate(info,axis=0), axis=0)
  return (minimap, screen, info)
  
def minimap_channel():
  return len(features.MINIMAP_FEATURES)

def screen_channel():
  return len(features.SCREEN_FEATURES)
