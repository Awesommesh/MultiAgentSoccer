# MultiAgentSoccer
Repository for training multi-agent soccer agents across 2v2, 3v3 and 4v4 soccer environments created using Unity and ML-Agents.

NOTE: It should be noted that currently, we only support training and evaluation on Apple Silicon Mac OS devices.

[Footage of General Agents against random agents]()

[Footage of Specialized Agents against random agents](https://drive.google.com/file/d/1iO3S3lXvTKDk2iwQOxgBYJfYFirvFs5G/view?usp=sharing)

# Environment setup
Use ```conda create --name [ENV_NAME] python=3.10``` to create new conda env with python 3.10. Replace ENV_NAME with desired environment name.
Use ```pip install -r requirements.txt``` to install dependencies

# Training

## Training PPO
First checkout to the ```main``` branch

Before training, edit the approrpiate training scripts (```train.py``` or ```train_specialized.py```) save file path to be based on your desired save directory. This can be done by searching for ```storage_path=``` and replacing the value after the equality sign to the desired global path for saved outputs.

Simply run ```ray start --head``` followed by ```python train.py -n [NUM_AGENTS]``` to train generalized PPO agents. Set NUM_AGENTS to be 2, 3 or 4 which will select the appropriate soccer environment with NUM_AGENTS players on each team. Other PPO hyperparamters like learning rate can be edited by simply looking at the arguments in the train.py script and adjusting them.

After stopping the training process or after it is completed, run ```ray stop``` to end the ray processes.

Additionally, by passing the ```--use_lstm``` flag, the PPO agents will train using a LSTM policy.

In order to train specialized agents in 2v2 setting, repeat the same process with ray start and stop but use ```python train_specialized.py -n 2```. Note that currently we have only tested and support specialized training for 2v2 setting.

## Training IMPALA
First checkout to the ```APPO``` branch

Before training, edit the ```train_impala.py``` save file path to be based on your desired save directory. This can be done by searching for ```storage_path=``` and replacing the value after the equality sign to the desired global path for saved outputs.

Simply run ```ray start --head``` followed by ```python train_impala.py -n [NUM_AGENTS]``` to train generalized PPO agents. Set NUM_AGENTS to be 2, 3 or 4 which will select the appropriate soccer environment with NUM_AGENTS players on each team.

After stopping the training process or after it is completed, run ```ray stop``` to end the ray processes.

Additionally, by passing the ```--use_lstm``` flag, the IMPALA agents will train using a LSTM policy.

# Evaluation
Checkout to the ```main``` branch

## Evaluating Genearl FFN Agent against each other or against Random agents
Run the following command to test how your trained agents perform against random agents or against other trained agents: ```python test_policy_against_random.py --timescale [TIMESCALE] -n [NUM_AGENTS] --ckpt_path1 [CKPT_1] --ckpt_path2 [CKPT_2]```

Set TIMESCALE FROM 0 to 100 to select simulation speed. To render in realtime set TIMESCALE=1.
Set NUM_AGENTS to 2, 3 or 4 to select the soccer environmnet based on number of agents per team.
Set CKPT_1 to the directory path of saved checkpoint folder or to "random" to select the policy for the blue team. If "random" is selected, the blue agents will sample all actions randomly.
Set CKPT_2 to the directory path of saved checkpoint folder or to "random" to select the policy for the purple team. If "random" is selected, the purple agents will sample all actions randomly.

Note that if a checkpoint directory is specified it must be a general policy with FFN network architecture.

Additionally, supply the ```--dont_render``` flag to not render the environment for faster execution.

## Evaluating LSTM Agent against other FFN or Random agents
Run the following command to test how your trained agents perform against random agents or against other trained agents: ```python test_lstm_policy_against_random.py --timescale [TIMESCALE] -n [NUM_AGENTS] --ckpt_path1 [CKPT_1] --ckpt_path2 [CKPT_2]```

Set TIMESCALE FROM 0 to 100 to select simulation speed. To render in realtime set TIMESCALE=1.
Set NUM_AGENTS to 2, 3 or 4 to select the soccer environmnet based on number of agents per team.
Set CKPT_1 to the directory path of LSTM based saved checkpoint folder or to "random" to select the policy for the blue team. If "random" is selected, the blue agents will sample all actions randomly.
Set CKPT_2 to the directory path of FFN based saved checkpoint folder or to "random" to select the policy for the purple team. If "random" is selected, the purple agents will sample all actions randomly.

Note that the CKPT_1 must be a LSTM policy or random and that CKPT_2 mustn't be a LSTM policy.

Additionally, supply the ```--dont_render``` flag to not render the environment for faster execution.

## Evaluating Specialized Agent against other General FFN or Random agents
Run the following command to test how your trained agents perform against random agents or against other trained agents: ```python test_policy_against_specialized.py --timescale [TIMESCALE] -n [NUM_AGENTS] --ckpt_path1 [CKPT_1] --ckpt_path2 [CKPT_2]```

Set TIMESCALE FROM 0 to 100 to select simulation speed. To render in realtime set TIMESCALE=1.
Set NUM_AGENTS to 2, 3 or 4 to select the soccer environmnet based on number of agents per team.
Set CKPT_1 to the directory path of general FFN based saved checkpoint folder or to "random" to select the policy for the blue team. If "random" is selected, the blue agents will sample all actions randomly.
Set CKPT_2 to the directory path of specialized FFN based saved checkpoint folder or to "random" to select the policy for the purple team. If "random" is selected, the purple agents will sample all actions randomly.

Note that the CKPT_1 must be a general FFN policy or random and that CKPT_2 must be a specialized FFN policy or random.

Additionally, supply the ```--dont_render``` flag to not render the environment for faster execution.
