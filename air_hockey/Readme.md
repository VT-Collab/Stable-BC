This directory provides the implemetnation details for our real world air hockey experiment

## Environment and Procedure
In this experiment, we first fixed the air hockey table in front of the robot so that the robot could move its arm to hit the puck. The side of the air hockey opposite to the robot was blocked using a foam wall to facilitate the return of the puck after hitting the wall. 

The environment consists of a 7-DoF Franka Emika Panda robot arm trying to play a simplified game of air hockey. The goal of the robot is to hit the puck as many times as possible without missing it.

The state of the robot $x \in \mathbb{R}^2$ is the position of the robot's end effector on the air hockey table and the action $u \in \mathbb{R}^2$ is the velocity of the robot's end effector. To track the position of the puck on the air hockey table, we used a camera mounted directly above the air hockey table, with the region of interest cropped to only capture the air hockey table. The camera tracked the position of the puck at a frequency of 20 Hz. The environment state $o \in \mathbb{R}^2$ is the position of the puck in the camera frame, and is comprised of comprised of the position of the puck at the current timestep t and the position of the puck at the previous timestep $t-1$ given as $o = (o^t_{puck}, o^{t-1}_{puck})$. . In this experiment, the robot knows its dynamics $f(x, u)$, but does not have access to the dynamics of the puck on the air hockey table $g(x, o, u)$.

To collect the offline training data, we next recruited 10 participants from the Virginia Tech community. Each participant was given 2 minutes to practice controlling the robot and hitting the puck. Once the practice was complete and the participants were comfortable teleoperating the robot to play air hockey, they used the same joystick to guide the robot to continuously hit the puck against the opposite side of the table for ~2.5 minutes (i.e., the human demonstrated the desired robot behavior). This data was recorded at a frequency of 20 Hz resulting in a total of ~3000 state-action pairs collected from each participant. We recognize that each user could behave differently when playing a dynamic game like air hockey. To evaluate if our approach is able to learn from this diverse range of human behaviors, we stored the training data collected from each user separately, resulting in 10 different training datasets $D_1, D_2, \cdots D_{10}$.  

Finally, after collecting the training datasets from all ten of the users, we trained separate policies for the data collected from each user. To test the performance of our approach with different amounts of training data, we trained the robot with varying amounts of training data. Specifically, we trained the policy with 15 seconds, 60 seconds and 120 seconds of training data collected from each participant, leading to a total of 30 trained policies (10 for each size of training data). Additionally, we tested the performance of our approach against Behavior Cloning when the data collected from all participants was made available for training. That is, we combined the datasets of demonstrations collected from all participants to form one large dataset with 24000 state-action pairs, and trained a policy using this large dataset of aggregated demonstrations. We evaluated each of the trained policies across 10 independent rollouts. In each rollout, the proctor started the evaluation by pushing the puck towards the robot and the robot executed its learned policy to try and continuously hit the puck.


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
\mathcal L(\theta) = \sum_{(x, o, u) \in \mathcal D}\Big [ \|\|u - \pi_\theta(x, o)\|\|^2 + \lambda_1 \|\|A_2\|\| + \lambda_2 \sum_{\sigma_i \in eig(A_1)} ReLU(Re(\sigma_i)) \Big]
$$

with $\lambda_1 = 0.1$ and $\lambda_2=10.0$. 

To train the policy, run the following script
```
python3 main.py train=True user=<user_number> num_dp=<datapoints> alg=<algorithm> 
```
Choose the algorithm from 'bc' and 'stable'.

This will train a policy for the respective algorithm and number of datapoints and save the trained policy in `data/user_<user_number>/<datapoints>_dp/model_<algorithm>.pt`

### Evaluation


To run the evaluations, run the following command:
```
python3 main.py rollout=True user=<user_number> num_dp=<datapoints> alg=<algorithm> eval_num=<evaluation_number>
```
The script will initiate communication with the robot and prompt the user to start the evaluaiton. When the robot misses the puck and the user ends the evaluation, the script will save the recorded data for the puck position and the robot position in `data/user_<user_number>/<datapoints>_dp/eval<algorithm>_<evaluation_number>.json`.
