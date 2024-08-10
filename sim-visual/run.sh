for run in {1..10}; do
    python3 main.py get_demo=True num_dp=500
    wait
    python3 main.py train=True alg=bc
    python3 main.py train=True alg=stable
    wait
    python3 main.py rollout=True alg=bc
    python3 main.py rollout=True alg=stable
done