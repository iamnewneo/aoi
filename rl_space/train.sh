export ENV="prod"
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONWARNINGS="ignore"
python -u dqn.py 2>&1 | tee -a out_log.log