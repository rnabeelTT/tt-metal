# source ~/evals/usersim/env.sh   -> points tau2 / VITA (text2_run.sh) at the Claude user-simulator relay
export USER_SIM_BASE_URL=http://127.0.0.1:${USERSIM_PORT:-28400}/v1
export USER_SIM_MODEL=claude-usersim
export USER_SIM_TIMEOUT=${USER_SIM_TIMEOUT:-3600}   # tau2 litellm per-request timeout (> shim's 1800 s answer timeout)
