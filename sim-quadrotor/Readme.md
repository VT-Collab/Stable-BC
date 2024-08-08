This directory provides the implemetnation details for the nonlinear quadrotor environment.

## Environment
The enviornment consists of a quadrotor that must navigate across a room with spherical obstacles.
The quadrotor's state $x \in \mathbb{R}^6$ includes its position $(p_x, p_y, p_z)$ and velocity $(v_x, v_y, v_z)$, and the quadrotor's action $u \in \mathbb{R}^3$ includes its acceleration $u_T$, roll $u_\phi$, and pitch $u_\theta$. 

## Expert
The expert for this environment is the MPPI controller which minimizes the quadrotors distance to the goal position while avoiding obstacles. The MPPI controller uses a Gaussian distribution to sample actions.

## Implementation
To collect demos, train and evaluate all the algorithms over 10 independent runs run the following command
```
bash run.sh
```

The detailed implementation for the environment is provided below

### Collecting Demos
To collect demonstrations in the quadrotor environment using the MPPI expert, run the following command
```
python3 get_data.py
```
This script will create a folder `data/` and generate demonstration dataset stored in `sim10_quadrotor/data/data_0.pkl`.

### Training the policy
In this environment, since the robot has access to the full state information, we use the loss function in Equation 7 to learn a policy for Stable-BC:

$$
\mathcal L(\theta) = \sum_{(x, y, u) \in \mathcal D}\Big [ \|\|u - \pi_\theta(x, y)\|\|^2 + \lambda \sum_{\sigma_i \in eig(A_1)} ReLU(Re(\sigma_i)) \Big]
$$

with $\lambda = 0.0001$. 

To train the BC and Stable-BC policies, run the following script
```
python3 run_train.py
```
To train the DART policy, run the following script
```
python3 dart.py
```

This will train 10 policies for each number of demonstrations and algorithm.

### Evaluation
Gaussian noise with standard deviation 0.1 is added to the actions of the quadrotor during evaluation to simulate real-world conditions and test the robustness of the learned policies. 

To run the evaluations, run the following command:
```
python3 run_test.py
```

This script will evaluate the policies over 100 test rollouts stating from randomly sampled initial states.

### Results
To plot the success rate of the policies, run the following command
```
python3 plot_sem.py
```
