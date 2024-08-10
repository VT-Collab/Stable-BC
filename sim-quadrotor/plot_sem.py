import numpy as np
import csv
import matplotlib.pyplot as plt


algs = [0, 1, 2]
num_demos = [5, 10, 20, 40, 60, 100]
random_seeds = [i for i in range(10)]


algs = [0, 1, 2]
num_demos = [5, 10, 20, 40, 60, 100]
random_seeds = [i for i in range(10)]
mean_bc = []
mean_dart = []
mean_stable = []

std_bc = []
std_dart = []
std_stable = []

for n in num_demos:
    success_rate = np.zeros((10, 3)).tolist()
    for seed in random_seeds:
        for alg in algs:
            file = open('results_0.001lr_1000epoch/lamda_0.0001/{}dems/{}/training_region_noise0.1/im_test_results_{}.txt'.format(n, seed, alg))
            text = file.readlines()
            success_rate[seed][alg] = float(text[11][15:19])
    mean_bc.append(np.mean(np.array(success_rate)[:, 0]))
    mean_dart.append(np.mean(np.array(success_rate)[:, 2]))
    mean_stable.append(np.mean(np.array(success_rate)[:, 1]))

    std_bc.append((np.std(np.array(success_rate)[:, 0]))/np.sqrt(len(random_seeds)))
    std_dart.append((np.std(np.array(success_rate)[:, 2]))/np.sqrt(len(random_seeds)))
    std_stable.append((np.std(np.array(success_rate)[:, 1]))/np.sqrt(len(random_seeds)))

mean_bc = np.array(mean_bc)
mean_dart = np.array(mean_dart)
mean_stable = np.array(mean_stable)

std_bc = np.array(std_bc)
std_dart = np.array(std_dart)
std_stable = np.array(std_stable)
plt.figure()
X = np.arange(len(num_demos))
plt.plot(X, mean_bc)
plt.plot(X, mean_dart)
plt.plot(X, mean_stable)

plt.fill_between(X, mean_bc-std_bc, mean_bc+std_bc, alpha=0.5)
plt.fill_between(X, mean_dart-std_dart, mean_dart+std_dart, alpha=0.5)
plt.fill_between(X, mean_stable-std_stable, mean_stable+std_stable, alpha=0.5)
plt.xticks(X, num_demos)
plt.ylabel('Success Rate')
plt.xlabel('Num Demos')
plt.savefig('success.svg')