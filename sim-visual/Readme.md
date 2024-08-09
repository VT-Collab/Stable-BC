This directory provides the implemetnation details for the vision-based environment.


## Environment
The environemnt consists of a point mass robot trying to reach a goal position. 

The robot state $x \in \mathbb{R}^2$ is its position in a 2D plane. The robot receives a $21\times 21$ image of the environment showing the goal location $y \in \mathbb {R}^{21\times 21}$. A simulated human provides demonstrations that show the robot the correct actions to be taken given the observation images. After collecting the set of expert demonstrations, we first train an autoencoder $\mathcal{E}(y)$ that embeds the $21\times 21$ image in a 10-D latent space. We then train a policy $\pi(x, \mathcal{E}(y))$ to imitate the expert behavior.

## Implementation
A list of all the arguments used in this simulation can be found in `cfg/config.yaml`

To collect demos, train and evaluate both BC and Stable-BC over 10 independent runs use the following command 
```
bash run.sh
```
The detailed implementation for the enviornment is provided below

### Collecting Demos
To collect demos in the visual environment with point mass robot run the following script
```
python3 main.py get_demo=True num_dp=<number_of_datapoints>
```
This script will create a folder `data/` and generate a demonstration dataset. The dataset will be stored in `data/data.json`.

This script will also train an autoencoder $\mathcal{E}$ on the collected data and save the autoencoder model in `data/autoencoder.pt`.

### Training the policy
In this environment, we use the loss function in Equation 11 to learn a policy for Stable-BC:

$$
\mathcal L(\theta) = \sum_{(x, y, u) \in \mathcal D}\Big [ \|\|u - \pi_\theta(x, y)\|\|^2 + \lambda_1 \|\|A_2\|\| + \lambda_2 \sum_{\sigma_i \in eig(A_1)} ReLU(Re(\sigma_i)) \Big]
$$

with $\lambda_1 = 0.1$ and $\lambda_2=10.0$. 

To train the policy, run the following script
```
python3 main.py train=True alg=<algorithm>
```
Choose the algorithm from 'bc' and 'stable'.

This will train a policy for the respective algorithm and save the trained policy in `data/model_<algorithm>.pt`

### Evaluation
To run the evaluations, run the following command:
```
python3 main.py rollout=True alg=<algorithm> test_case=<1, 2 or 3>
```
By default, this code will evaluate the selected algorithm for the chosen test case over 1000 rollouts. The raw error data for the 1000 rollouts will be stored in `data/results_<algorithm>.json` and the average error will be stored in `data/results_<algorithm>.csv`