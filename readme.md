## π-eLight: Programmatic Interpretable Reinforcement Learning for Effective Traffic Signal Control

π-eLight is an improvement over our previous work, π-Light. Similar to the previous work, we represent the policy using programs. We propose a new program framework that includes an additional program to determine whether to maintain the current phase. Additionally, we introduce an improved program search algorithm to explore program combinations. π-eLight outperforms π-Light and achieved first place in the Traffic Signal Control track of the 2024 Tencent Kaiwu Global AI Competition.

Note that we have also fixed some minor bugs in the environment from the previous code.



## Dependencies

- python=3.8.10
- torch=1.7.1+cu110
- numpy=1.21.2
- CityFlow=0.1.0 

You need to install a modified version of [CityFlow](https://github.com/dxing-cs/TinyLight#dependencies) to run the code.

Then you need to unzip the data file.



### Run $\pi$-eLight

```shell
python 02_run_MCTS.py --dataset=Jinan --p_mode=two
```

### Evaluate generalization performance of $\pi$-eLight

```shell
python 02_run_MCTS.py --dataset=Hangzhou1 --generalization=True target=Manhattan
```

### Run Tinylight

```shell
python 00_run_tiny_light.py --dataset=Jinan
```

### Run other baselines

```shell
python 01_run_baseline.py --dataset=Jinan
```

### Evaluate generalization performance of other baselines

```shell
python 015_baseline_transfer.py --dataset=Jinan
```

### Run UniTSA 

```shell
python 04_run_universal_light.py --dataset=Jinan
```



### Run [VIPER](https://arxiv.org/abs/1805.08328)

We also compare imitation learning-based VIPER, which distills the neural policy into a decision tree. We utilized MPLight as a teacher to generate state-action pairs for training the decision tree.
Overall, VIPER's performance is close to that of MPlight.

```shell
python 03_run_viper.py
```



## 关于腾讯开悟AI全球公开赛-智能交通信号灯调度赛道-第一名的方案

[智能交通信号灯调度赛道 - 腾讯开悟](https://aiarena.tencent.com/aiarena/zh/match/open-competition-2024/open-competition-2024-3/)

腾讯比赛中的环境比cityflow更复杂一些，每次决策需要指定相位和持续时间。你可以将config中的action_interval调小，然后运行 `python 02_run_MCTS.py --p_mode=two`，这能大体上还原比赛方案。



## Acknowledgments

This codebase is based on [Tinylight](https://github.com/dxing-cs/TinyLight)'s code.