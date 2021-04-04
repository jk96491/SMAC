REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .episode_runner_V2 import EpisodeRunnerV2
REGISTRY["episode_V2"] = EpisodeRunnerV2

from .parallel_runner_V2 import ParallelRunnerV2
REGISTRY["parallel_V2"] = ParallelRunnerV2

from .episode_runner_V3 import EpisodeRunnerV3
REGISTRY["episode_V3"] = EpisodeRunnerV3
