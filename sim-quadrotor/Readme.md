

\subsection{Nonlinear Quadrotor Navigation} \label{sec:sim2}

In our second simulation we apply Stable-BC to a nonlinear system.
Specifically, we consider a quadrotor that must navigate across a room with spherical obstacles (see \fig{sim2}).
The quadrotor's state $x \in \mathbb{R}^6$ includes its position $(p_x, p_y, p_z)$ and velocity $(v_x, v_y, v_z)$, and the quadrotor's action $u \in \mathbb{R}^3$ includes its acceleration $u_T$, roll $u_\phi$, and pitch $u_\theta$. 
State $x$ evolves with nonlinear dynamics:
\begin{gather*}
   \dot{p}_x = v_x, \quad \dot{p}_y = v_y, \quad \dot{p}_z = v_z \\
    \dot{v}_x = a_g \tan u_\theta, \quad \dot{v}_y = -a_g \tan u_\phi, \quad \dot{v}_z = u_T - a_g 
\end{gather*}
At the start of each interaction the quadrotor is uniformly randomly initialized on one side of the room. 
There are seven static obstacles that the robot must avoid as it navigates to its fixed goal location on the opposite side of the room. The interaction ends when the quadrotor either reaches within $0.5$ units of the goal (a success) or collides with an obstacle or wall of the room (a failure).

\begin{figure}
    \centering
    \includegraphics[width=1\columnwidth]{Figures/sim2.pdf}
    \caption{Simulation results for nonlinear quadrotor navigation. (Left) An example trajectory of the quadrotor flying around the $3$D obstacles to reach its goal position. (Right) Average success rate of the quadrotor. We trained the system end-to-end $10$ separate times, and then performed $100$ test rollouts with each trained model. Shaded regions show SEM.}
    \vspace{-1.5em}
    \label{fig:sim2}
\end{figure}

\p{Methods}
Because the obstacles and goal locations are fixed, and the quadrotor knows its dynamics, in this simulation we apply our \textit{model-based approach} for \textbf{Stable-BC}.
We compare Stable-BC to two baselines: \textbf{BC} and \textbf{DART} \cite{laskey2017dart}.
DART is a state-of-the-art data collection approach that perturbs the expert while they provide demonstrations to increase dataset diversity.
As the robot collects expert demonstrations offline, DART iteratively estimates the errors between the expert's actions and its current policy.
DART then injects noise based on these errors when collecting new demonstrations from the expert; this causes the expert to show the robot more diverse and corrective behaviors.
BC and Stable-BC are trained using the same offline dataset that does not include DART's perturbation procedure.
To test the robustness of the learned policies and simulate real-world conditions, we inject Gaussian noise into quadrotor's actions at test time.


\p{Results}
Our results are summarized in \fig{sim2}. 
We report the success rate, i.e., the fraction of trials where the quadrotor reached its goal without collisions.
For all methods the success rate increases when the robot is given more expert demonstrations.
However, Stable-BC achieves a higher success rate with fewer demonstrations as compared to the baselines.
Looking specifically at DART and Stable-BC, we find that Stable-BC with the original offline dataset converges to best-case performance more rapidly than robots which use DART to perturb the expert and collect more diverse data.
These results demonstrate that Stable-BC can be effectively applied to nonlinear systems.








This directory provides the implemetnation details for the nonlinear quadrotor environment.

## Environment
The enviornment consists of a quadrotor that must navigate across a room with spherical obstacles.
The quadrotor's state $x \in \mathbb{R}^6$ includes its position $(p_x, p_y, p_z)$ and velocity $(v_x, v_y, v_z)$, and the quadrotor's action $u \in \mathbb{R}^3$ includes its acceleration $u_T$, roll $u_\phi$, and pitch $u_\theta$. 
State $x$ evolves with nonlinear dynamics:
\begin{gather*}
   \dot{p}_x = v_x, \quad \dot{p}_y = v_y, \quad \dot{p}_z = v_z \\
    \dot{v}_x = a_g \tan u_\theta, \quad \dot{v}_y = -a_g \tan u_\phi, \quad \dot{v}_z = u_T - a_g 
\end{gather*}

## Expert
The expert for this environment is the MPPI controller which minimizes the cost function:

<!-- def get_cost(x, u, dist_to_goal, dist_to_obs, dist_to_map_boundaries, **kwargs):
    '''
    x: (number_of_samples, horizon, state_dim) (px, py, pz, vx, vy, vz)
    u: (number_of_samples, horizon, control_dim) ( thrust, roll, pitch )
    x_goal: (px, py, pz)
    obstacle_list: (number of obstacles, 4) (px, py, pz, radius)
    map_boundaries: (px_bound, py_bound, pz_bound)
    '''

    # # distance to goal
    # dist_to_goal = get_dist_to_goal(x, x_goal)

    # # distance to obstacles
    # dist_to_obs = get_dist_to_obs(x, obstacle_list)
    # # distance to map boundaries
    # dist_to_map_boundaries = get_dist_to_map_boundaries(x, map_boundaries)
    min_dist_to_obstacles = np.minimum(dist_to_obs, dist_to_map_boundaries)

    # goal cost
    goal_cost = np.sum(dist_to_goal * kwargs['goal_cost_weight'], axis=-1)

    # obstacle cost
    obstacle_cost = np.sum(np.exp(-min_dist_to_obstacles * kwargs['obstacle_cost_exponential_weight']) * kwargs['obstacle_cost_weight'], axis=-1)

    # control cost
    control_cost = np.sum(np.sum( ((u - np.array([A_G, 0, 0]))/ np.array([f_g_diff_max, roll_max, pitch_max]) ) **2, axis=-1) * kwargs['control_cost_weight'], axis=-1)   

    # total cost
    total_cost = goal_cost + obstacle_cost + control_cost
    return total_cost -->
<!-- convert this getting cost function into an equation -->
$$
\mathcal L(\theta) = \sum_{(x, y, u) \in \mathcal D}\Big [ \|\|u - \pi_\theta(x, y)\|\|^2 + \lambda \sum_{\sigma_i \in eig(A_1)} ReLU(Re(\sigma_i)) \Big]
$$
    <!-- kwargs = {'goal_cost_weight': 1.5, 'obstacle_cost_weight': 1e1, 'obstacle_cost_exponential_weight': 1e1, 'control_cost_weight': 1e-2} -->
where 


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
python3 plot_sam.py
```
