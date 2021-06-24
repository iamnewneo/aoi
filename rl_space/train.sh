export ENV="prod"
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONWARNINGS="ignore"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/kunalg/software/anaconda3/lib/
python -u dqn_reward_change.py 2>&1 | tee -a out_reward_change_log.log