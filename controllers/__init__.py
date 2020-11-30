REGISTRY = {}

from .basic_controller import BasicMAC
from .separate_controller import SeparateMAC
from .rode_controller import RODEMAC
from .noise_controller import NoiseMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["separate_mac"] = SeparateMAC
REGISTRY["rode_mac"] = RODEMAC
REGISTRY["noise_mac"] = NoiseMAC