# Enhancing Deep Reinforcement Learning with Accurate N-Step Methods: Bias and Variance Reduction Strategies

PyTorch implementation of multi-step TD3 algorithm. If you use our code or data please cite the paper.

Method is tested on [MuJoCo](http://www.mujoco.org/) continuous control tasks in [OpenAI gym](https://github.com/openai/gym). 
Networks are trained using [torch 1.3.0](https://github.com/pytorch/pytorch) and Python 3.7.

### Usage
The paper results can be reproduced by running:
```
python main.py  --policy TD3_N --env Ant-v2 --N-step 100 --alpha 0.005 --lambda 0.1 --seed 0
python main.py  --policy TD3_N --env Hopper-v2  --N-step 100 --alpha 0.0005 --lambda 0.9 --seed 0
python main.py  --policy TD3_N --env HalfCheetah-v2   --N-step 300 --alpha 0.01 --lambda 0.9 --seed 0
python main.py  --policy TD3_N --env Swimmer-v2  --N-step 100 --alpha 0.005 --lambda 0.8 --seed 0
python main.py  --policy TD3_N --env Walker2d-v2  --N-step 150 --alpha 0.001 --lambda 0.9  --seed 0
python main.py  --policy TD3_N --env LunarLanderContinuopus-v2   --N-step 20 --alpha 0.005 --lambda 0.4 --seed 0
 
```
Hyper-parameters can be modified with different arguments to main.py. 