This directory provides the implemetnation details for the interactive intersection environment.

## Environment
The enviornment consists of a simulated human car and a autonomous car trying to cross an intersection, while avoiding a collision.

The autonomous car state $x \in \mathbb{R}^2$ is its position and the action $u \in \mathbb{R}^2$ is the autonomous car's velocity. The environment state $y \in \mathbb {R}^2$ is the state of the simulated human car. In this simulation, the robot knows its dynamics $f(x, u)$, but does not have access to the dynamics of the human car $g(x, y, u)$.

To collect demonstrations in this environment, both the agents optimize for the following cost function:
$$
Cost(x, y, c) = \|x(t + \Delta t - c)\| - \|x(t) - x\| + 0.75 \cdot \|x(t) - y(t)\| - 0.75 \cdot \|x(t + \Delta t) - y(t)\|
$$
where $c$ is the position of the constant goal in the environment (across the intersection).

## Implementation
A list of all the arguments used in this simulation can be found in `cfg/config.yaml`

To collect demos, train and evaluate all the algorithms over 10 independent runs run the following command
```
bash run.sh
```

The detailed implementation for the environment is provided below

### Collecting Demos
To collect demos in the intersection environment using the cost function defined above run the following script
```
python3 main.py get_demo=True num_demos=<number of demos>
```
This script will create a folder `data/` and generate demonstration datasets for both *BC* and *CCIL*. The dataset for *BC* will be stored in `data/data.json` and that for *CCIL* will be stored in `data/data_ccil.json`.

### Training the policy
In this environment, since the robot does not have access to the environment dynamics (dynamics of the simulated human car), we use the loss function in Equation 11 to learn a policy for Stable-BC:
$$
\mathcal L(\theta) = \sum_{(x, y, u) \in \mathcal D}\Big [ \|u - \pi_\theta(x, y)\|^2 + \lambda_1 \|A_2\| + \lambda_2 \sum_{\sigma_i \in eig(A_1)} ReLU(Re(\sigma_i)) \Big]
$$
with $\lambda_1 = 0.1$ and $\lambda_2=10.0$. 

To train the policy, run the following script
```
python3 main.py train=True alg=<algorithm>
```
Choose the algorithm from 'bc', 'ccil', 'stable_bc', 'stable_ccil'.

This will train a policy for the respective algorithm and save the trained policy in `data/model_<algorithm>.pt`

### Evaluation
We consider 3 different test conditions for evaluation of the learned policy
- 1: The testing distribution matches the training distribution
- 2: The human behavior is different than that seen during training (simulated human optimizes a different cost function)
- 3: The robot car's initial position is chosen from a distribution different from the one seen during training

To run the evaluations, run the following command:
```
python3 main.py rollout=True alg=<algorithm> test_case=<1, 2 or 3>
```
By default, this code will evaluate the selected algorithm for the chosen test case over 100 rollouts. The raw cost data for the 100 rollouts will be stored in `data/results_<algorithm>_<test_case>.json` and the average of the costs will be stored in `data/results_<algorithm>_<test_case>.csv`