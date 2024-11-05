# Stable-BC: Bounding Covariate Shift with Stable Behavior Cloning
This repository provides our implementation of Stable-BC in three different simulation environments and in a real world air hockey game using a 7-DoF  Franka Emika robot arm. The videos for our experiments can be found [here](youtube link)

 ## Installation and Setup

To install Stable-BC, clone the repo using
```
git clone https://github.com/VT-Collab/Stable_BC.git
```

Install the required packages and dependencies for running Stable-BC
- PyPI(pip):
```
pip install -r requirements.txt
```

- Conda Environment
```
conda env create -f stable_bc.yml
```

## Implementation
The implementation details for each experiment are provided in the respective directories:
- [Intersection Environment](https://github.com/VT-Collab/Stable_BC/tree/master/sim-intersection)


- [Quadrotor Simulation](https://github.com/VT-Collab/Stable_BC/tree/master/sim-quadrotor)

- [Vision-Based Environment](https://github.com/VT-Collab/Stable_BC/tree/master/sim-visual)

- [Air Hockey Experiments](https://github.com/VT-Collab/Stable_BC/tree/master/air_hockey)

## Comparison with Offline-RL
In addition to the comparison with state-of-the-art imitation learning algorithms performed in the manuscript, we also compare our proposed approach to an Offline-RL algorithm in the intersection environment. The implementation details for the Offline-RL algorithm and a detailed comparison of performance against Stable-BC is provided in:
- [Intersection Environment/CQL](https://github.com/VT-Collab/Stable-BC/tree/master/sim-intersection/CQL)