python3 bandit_experiment.py --dataset 'satimage' --history_sample_size 800 --policy_type 'TS' &
python3 bandit_experiment.py --dataset 'mnist' --history_sample_size 800 --policy_type 'TS' &
python3 bandit_experiment.py --dataset 'letter' --history_sample_size 800 --policy_type 'TS' &
python3 bandit_experiment.py --dataset 'Sensorless' --history_sample_size 800 --policy_type 'TS' &
python3 bandit_experiment.py --dataset 'connect-4' --history_sample_size 800 --policy_type 'TS' &

wait

python3 bandit_experiment.py --dataset 'satimage' --history_sample_size 1000 --policy_type 'TS' &
python3 bandit_experiment.py --dataset 'mnist' --history_sample_size 1000 --policy_type 'TS' &
python3 bandit_experiment.py --dataset 'letter' --history_sample_size 1000 --policy_type 'TS' &
python3 bandit_experiment.py --dataset 'Sensorless' --history_sample_size 1000 --policy_type 'TS' &
python3 bandit_experiment.py --dataset 'connect-4' --history_sample_size 1000 --policy_type 'TS' &

wait

python3 bandit_experiment.py --dataset 'satimage' --history_sample_size 1200 --policy_type 'TS' &
python3 bandit_experiment.py --dataset 'mnist' --history_sample_size 1200 --policy_type 'TS' &
python3 bandit_experiment.py --dataset 'letter' --history_sample_size 1200 --policy_type 'TS' &
python3 bandit_experiment.py --dataset 'Sensorless' --history_sample_size 1200 --policy_type 'TS' &
python3 bandit_experiment.py --dataset 'connect-4' --history_sample_size 1200 --policy_type 'TS' &

wait