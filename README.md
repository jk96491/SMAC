# StarCraft II Multi Agent Challenge

## **The algorithms provided are _QMIX, COMA, LIIR, G2ANet, QTRAN, VDN, Central V, IQL, ROMA and RODE_ .**

This repository has been edited for convenient execution in Windows OS.

<img src="https://user-images.githubusercontent.com/17878413/97562077-72142600-1a24-11eb-962c-39df6423ced4.png" width="30%"></img>

<img src="https://user-images.githubusercontent.com/17878413/97562396-e8b12380-1a24-11eb-92ba-c9a05ec3630b.png" width="30%"></img>

<img src="https://user-images.githubusercontent.com/17878413/97562276-b69fc180-1a24-11eb-997c-e5feeff6a30b.png" width="30%"></img>


First you need to install the StarCraft 2 game. Trial version does not matter. Download it from the link below
 
 ```shell
https://starcraft2.com/ko-kr/
```

After installation, you should download the map required for the minigame from the link below.

 ```shell
https://github.com/oxwhirl/smac/tree/master/smac/env/starcraft2/maps/SMAC_Maps
```

You can move all downloaded files to the path below.

 ```shell
C:\Program Files (x86)\StarCraft II\Maps\SMAC_Maps
```

# From now on, this is the environment setting.
Enter the following command to install the packages you need first.
 ```shell
pip install -r requirements.txt
```
Unfortunately, you need to install the two below yourself (it is not difficult).

You need to install cloudpickle.
 ```shell
pip install cloudpickle
```
You also need to install pytorch.
 ```shell
conda install pytorch==1.2.0 torchvision==0.4.0 -c pytorch
```
Finally, run "main.py"

# Python MARL framework

PyMARL is [WhiRL](http://whirl.cs.ox.ac.uk)'s framework for deep multi-agent reinforcement learning and includes implementations of the following algorithms:
- [**QMIX**: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [**COMA**: Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)
- [**VDN**: Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296) 
- [**IQL**: Independent Q-Learning](https://arxiv.org/abs/1511.08779)
- [**QTRAN**: QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1905.05408)
- [**LIIR**: LIIR: Learning Individual Intrinsic Reward in Multi-Agent Reinforcement Learning](https://papers.nips.cc/paper/2019/file/07a9d3fed4c5ea6b17e80258dee231fa-Paper.pdf)
- [**ROMA**: ROMA: Multi-Agent Reinforcement Learning with Emergent Roles](https://arxiv.org/abs/2003.08039)
- [**RODE**: RODE: Learning Roles to Decompose Multi-Agent Tasks](https://arxiv.org/abs/2010.01523)

PyMARL is written in PyTorch and uses [SMAC](https://github.com/oxwhirl/smac) as its environment.

## Installation instructions

Build the Dockerfile using 
```shell
cd docker
bash build.sh
```

Set up StarCraft II and SMAC:
```shell
bash install_sc2.sh
```

This will download SC2 into the 3rdparty folder and copy the maps necessary to run over.

The requirements.txt file can be used to install the necessary packages into a virtual environment (not recomended).

## Run an experiment 

```shell
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=2s3z
```

The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

To run experiments using the Docker container:
```shell
bash run.sh $GPU python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=2s3z
```

All results will be stored in the `Results` folder.

The previous config files used for the SMAC Beta have the suffix `_beta`.

## Saving and loading learnt models

### Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called *models*. The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

### Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep. 

## Watching StarCraft II replays

`save_replay` option allows saving replays of models which are loaded using `checkpoint_path`. Once the model is successfully loaded, `test_nepisode` number of episodes are run on the test mode and a .SC2Replay file is saved in the Replay directory of StarCraft II. Please make sure to use the episode runner if you wish to save a replay, i.e., `runner=episode`. The name of the saved replay file starts with the given `env_args.save_replay_prefix` (map_name if empty), followed by the current timestamp. 

The saved replays can be watched by double-clicking on them or using the following command:

```shell
python -m pysc2.bin.play --norender --rgb_minimap_size 0 --replay NAME.SC2Replay
```

**Note:** Replays cannot be watched using the Linux version of StarCraft II. Please use either the Mac or Windows version of the StarCraft II client.

## Documentation/Support

Documentation is a little sparse at the moment (but will improve!). Please raise an issue in this repo, or email [Tabish](mailto:tabish.rashid@cs.ox.ac.uk)

## Citing PyMARL 

If you use PyMARL in your research, please cite the [SMAC paper](https://arxiv.org/abs/1902.04043).

*M. Samvelyan, T. Rashid, C. Schroeder de Witt, G. Farquhar, N. Nardelli, T.G.J. Rudner, C.-M. Hung, P.H.S. Torr, J. Foerster, S. Whiteson. The StarCraft Multi-Agent Challenge, CoRR abs/1902.04043, 2019.*

In BibTeX format:

```tex
@article{samvelyan19smac,
  title = {{The} {StarCraft} {Multi}-{Agent} {Challenge}},
  author = {Mikayel Samvelyan and Tabish Rashid and Christian Schroeder de Witt and Gregory Farquhar and Nantas Nardelli and Tim G. J. Rudner and Chia-Man Hung and Philiph H. S. Torr and Jakob Foerster and Shimon Whiteson},
  journal = {CoRR},
  volume = {abs/1902.04043},
  year = {2019},
}
```

## License

Code licensed under the Apache License v2.0
