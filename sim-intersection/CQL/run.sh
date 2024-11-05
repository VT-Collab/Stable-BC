n_demos=(2 5 10 20 50 100 200 500)
for n in ${n_demos[@]}; do
echo $n " Demos"
    python3 train_offline.py --num_demos $n --test_case 2
    wait
done


n_demos=(2 5 10 20 50 100 200 500)
for n in ${n_demos[@]}; do
echo $n " Demos"
    python3 train_offline.py --num_demos $n --test_case 3
    wait
done