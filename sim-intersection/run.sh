for run in {1..10}; do
    python3 main.py get_demo=True num_demos=5
    wait
    python3 main.py train=True alg=bc
    python3 main.py train=True alg=ccil
    python3 main.py train=True alg=stable_bc
    python3 main.py train=True alg=stable_ccil
    wait
    python3 main.py rollout=True alg=bc
    python3 main.py rollout=True alg=ccil
    python3 main.py rollout=True alg=stable_bc
    python3 main.py rollout=True alg=stable_ccil
done