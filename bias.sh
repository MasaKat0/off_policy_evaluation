python3 bias.py --history_sample_size 100 --policy_type 'TS' &
python3 bias.py --history_sample_size 250 --policy_type 'TS' &
python3 bias.py --history_sample_size 500 --policy_type 'TS' &
python3 bias.py --history_sample_size 100 --policy_type 'UCB' &
python3 bias.py --history_sample_size 250 --policy_type 'UCB' &
python3 bias.py --history_sample_size 500 --policy_type 'UCB' &

wait