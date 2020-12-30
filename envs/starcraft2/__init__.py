from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .starcraft2 import StarCraft2Env

from absl import flags

FLAGS = flags.FLAGS
FLAGS(['main.py'])