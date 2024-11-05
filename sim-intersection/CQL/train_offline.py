import numpy as np
from collections import deque
import torch
import wandb
import argparse
import glob
from utils import save
import random
from agent import CQLSAC
from torch.utils.data import DataLoader, TensorDataset
import csv
import os

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="CQL", help="Run name, default: CQL")
    parser.add_argument("--env", type=str, default="halfcheetah-medium-v2", help="Gym environment name, default: Pendulum-v0")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes, default: 100")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--log_video", type=int, default=0, help="Log agent behaviour to wanbd when set to 1, default: 0")
    parser.add_argument("--save_every", type=int, default=100, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size, default: 256")
    parser.add_argument("--hidden_size", type=int, default=256, help="")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="")
    parser.add_argument("--temperature", type=float, default=1.0, help="")
    parser.add_argument("--cql_weight", type=float, default=1.0, help="")
    parser.add_argument("--target_action_gap", type=float, default=10, help="")
    parser.add_argument("--with_lagrange", type=int, default=0, help="")
    parser.add_argument("--tau", type=float, default=5e-3, help="")
    parser.add_argument("--eval_every", type=int, default=100, help="")
    parser.add_argument("--num_demos", type=int, default=10)
    parser.add_argument("--test_case", type=int, default=1)
    parser.add_argument("--num_runs", type=int, default=10)
    
    args = parser.parse_args()
    return args

def expert_agent(x, y, goal, beta=20.0):
        U = np.random.uniform(-1, 1, (100, 2))
        P = []
        R = []
        for u in U:
            if np.linalg.norm(u) > 1.0:
                u /= np.linalg.norm(u)
            C_goal = np.linalg.norm((x + u) - goal) - np.linalg.norm(x - goal)
            C_avoid = np.linalg.norm(x - y) - np.linalg.norm((x + u) - y)
            C = C_goal + 0.75 * C_avoid
            R.append(-1.*C)
            P.append(np.exp(-beta * C))
        P /= np.sum(P)
        idx = np.random.choice(len(U), p=P)
        return U[idx, :], R[idx]

def new_agent(y, goal, beta=20.0):
    U = np.random.uniform(-1, 1, (100, 2))
    P = []
    R = []
    for u in U:
        if np.linalg.norm(u) > 1.0:
            u /= np.linalg.norm(u)
        C = np.linalg.norm((y + u) - goal) - np.linalg.norm(y - goal)
        R.append(-1.0 * C)
        P.append(np.exp(-beta * C))
    P /= np.sum(P)
    idx = np.random.choice(len(U), p=P)
    return U[idx, :], R[idx]

def get_dataset(num_traj=10):
    num_trajectories = num_traj
    dataset = {"actions": [], "observations": [], "next_observations": [], "rewards": [], "terminals": []}
    goal_x = np.array([10., 0.])
    goal_y = np.array([0., 10.])
    for _ in range(num_trajectories):
        x = np.random.uniform([-10, -10], [0, 10], 2)
        y = np.random.uniform([-10, -10], [10, 0], 2)
        tau = []
        for idx in range(20):
            u1, R = expert_agent(x, y, goal_x)
            u2, _ = expert_agent(y, x, goal_y)
            state = np.array([x[0], x[1], y[0], y[1]])
            next_x = x + u1
            next_y = y + u2
            next_state = np.array([next_x[0], next_x[1], next_y[0], next_y[1]])

            dataset["actions"].append(u1.tolist())
            dataset["observations"].append(state.tolist())
            dataset["next_observations"].append(next_state.tolist())
            dataset["rewards"].append(R)
            dataset["terminals"].append(0 if idx < 19 else 1)

            x = next_x
            y = next_y
    dataset["actions"] = np.array(dataset["actions"])
    dataset["observations"] = np.array(dataset["observations"])
    dataset["next_observations"] = np.array(dataset["next_observations"])
    dataset["rewards"] = np.array(dataset["rewards"])
    dataset["terminals"] = np.array(dataset["terminals"])
    return dataset


def prep_dataloader(config, seed=1):
    dataset = get_dataset(config.num_demos)
    batch_size = int(len(dataset["actions"])/10.)
    tensors = {}
    for k, v in dataset.items():
        if k in ["actions", "observations", "next_observations", "rewards", "terminals"]:
            if  k is not "terminals":
                tensors[k] = torch.from_numpy(v).float()
            else:
                tensors[k] = torch.from_numpy(v).long()

    tensordata = TensorDataset(tensors["observations"],
                               tensors["actions"],
                               tensors["rewards"][:, None],
                               tensors["next_observations"],
                               tensors["terminals"][:, None])
    dataloader  = DataLoader(tensordata, batch_size=batch_size, shuffle=True)
    
    return dataloader

def evaluate(config, policy, eval_runs=100): 
    """
    Makes an evaluation run with the current policy
    """
    reward_batch = []
    goal_x = np.array([10., 0.])
    goal_y = np.array([0., 10.])
    for i in range(eval_runs):
        x = np.random.uniform([-10, -10], [0, 10], 2)
        y = np.random.uniform([-10, -10], [10, 0], 2)
        if config.test_case == 3:
            if np.random.rand() < 0.5:
                x = np.random.uniform([-10, 10], [0, 15], 2)
            else:
                x = np.random.uniform([-10, -15], [0, -10], 2)
        state = np.array([x[0], x[1], y[0], y[1]])
        
        rewards = 0
        for idx in range(20):
            u1 = policy.get_action(state, eval=True)
            u2, _ = expert_agent(y, x, goal_y)
            if config.test_case == 2:
                u2, _ = new_agent(y, goal_y)

            C_goal = np.linalg.norm((x + u1) - goal_x) - np.linalg.norm(x - goal_x)
            C_avoid = np.linalg.norm(x - y) - np.linalg.norm((x + u1) - y)
            C = C_goal + 0.75 * C_avoid

            x += u1
            y += u2

            rewards += -1. * C


            state = np.array([x[0], x[1], y[0], y[1]])

        reward_batch.append(rewards)
    return np.mean(reward_batch)

def train(config):
    seed  = np.random.randint(10000)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    dataloader = prep_dataloader(config, seed=config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    batches = 0
    average10 = deque(maxlen=10)
    
        
    agent = CQLSAC(state_size=4,
                    action_size=2,
                    tau=config.tau,
                    hidden_size=config.hidden_size,
                    learning_rate=config.learning_rate,
                    temp=config.temperature,
                    with_lagrange=config.with_lagrange,
                    cql_weight=config.cql_weight,
                    target_action_gap=config.target_action_gap,
                    device=device)

    for i in range(1, config.episodes+1):

        for batch_idx, experience in enumerate(dataloader):
            states, actions, rewards, next_states, dones = experience
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)
            policy_loss, alpha_loss, bellmann_error1, bellmann_error2, cql1_loss, cql2_loss, current_alpha, lagrange_alpha_loss, lagrange_alpha = agent.learn((states, actions, rewards, next_states, dones))
            batches += 1

        if i % config.eval_every == 0:
            eval_reward = evaluate(config, agent)

            average10.append(eval_reward)
            print("Episode: {} | Reward: {} | Polciy Loss: {} | Batches: {}".format(i, eval_reward, policy_loss, batches,))
            
            if i % config.save_every == 0:
                save(config, save_name=config.test_case, model=agent.actor_local, wandb=wandb, ep=0)
    return eval_reward

if __name__ == "__main__":
    config = get_config()
    if not os.path.exists('data/'):
        os.makedirs('data/')

    for idx in range(config.num_runs):
        print(idx)
        eval_reward = train(config)
        with open('data/results_rl{}_{}.csv'.format(config.test_case, config.num_demos), 'a') as myfile:
            writer = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            datarow = [eval_reward]
            writer.writerow(datarow)