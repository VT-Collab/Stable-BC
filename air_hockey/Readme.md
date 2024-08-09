This directory provides the implemetnation details for our real world air hockey experiment

## Environment
The environment consists of a 7-DoF Franka Emika Panda robot arm trying to play a simplified game of air hockey. The goal of the robot is to hit the puck as many times as possible without missing it.

The state of the robot $x \in \mathbb{R}^2$ is the position of the robot's end effector on the air hockey table and the action $u \in \mathbb{R}^2$ is the velocity of the robot's end effector. The environment state $y \in \mathbb{R}^2$ is the position of the puck in the camera frame. In this experiment, the robot knows its dynamics $f(x, u)$, but does not have access to the dynamics of the puck on the air hockey table $g(x, y, u)$.

## Implementation
A list of all the arguments used in this experiment can be found in `cfg/config.yaml`

The detailed implementation for the environment is provided below

### Collect Demos
To collect demos in the air hockey experiment run the following script
```
python3 main.py get_demo=True user=<user_number>
```
This script will initiate communication with the robot and prompt the user to start providing the demonstration. When the user ends the interaction, the script will create a folder `data/user_<user_number>` and store the recorded demonstration in `data/user_<user_number>/demo.json`. The script will then process the collected demos to get them in the format for needed for training and store the processed demos in `data/user_<user_number>/demo_processed.json`.

### Training the policy
In this experiment, since the robot does not have access to the environment dynamics (dynamics of the simulated human car), we use the loss function in Equation 11 to learn a policy for Stable-BC:

$$
\mathcal L(\theta) = \sum_{(x, y, u) \in \mathcal D}\Big [ \|\|u - \pi_\theta(x, y)\|\|^2 + \lambda_1 \|\|A_2\|\| + \lambda_2 \sum_{\sigma_i \in eig(A_1)} ReLU(Re(\sigma_i)) \Big]
$$

with $\lambda_1 = 0.1$ and $\lambda_2=10.0$. 

To train the policy, run the following script
```
python3 main.py train=True user=<user_number> num_dp=<datapoints> alg=<algorithm> 
```
Choose the algorithm from 'bc' and 'stable'.

This will train a policy for the respective algorithm and number of datapoints and save the trained policy in `data/user_<user_number>/{datapoints}_dp/model_<algorithm>.pt`

### Evaluation


To run the evaluations, run the following command:
```
python3 main.py rollout=True user=<user_number> num_dp=<datapoints> alg=<algorithm> eval_num=<evaluation_number>
```
The script will initiate communication with the robot and prompt the user to start the evaluaiton. When the robot misses the puck and the user ends the evaluation, the script will save the recorded data for the puck position and the robot position in `data/user_<user_number>/<datapoints>_dp/eval<algorithm>_<evaluation_number>.json`.
