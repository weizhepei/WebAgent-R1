#!/bin/bash

PUBLIC_HOSTNAME="localhost"
test_config_base_dir='config_files/wa/test_webarena_lite'
# TODO: ip address of the model you are deploying (only if you are deploying your own model using e.g. vllm)
planner_ip='http://localhost:8000/v1' 

model='Qwen/Qwen2.5-3B-Instruct'
result_dir="eval_results/${model}"

SERVER=${PUBLIC_HOSTNAME}
MAP_SERVER=${PUBLIC_HOSTNAME}

# TODO: insert OpenAI API key here
OPENAI_API_KEY='xxxx'
OPENAI_API_URL="https://api.openai.com/v1"
OPENAI_ORGANIZATION=''

CONDA_ENV_NAME='webagent-r1'

ENV_VARIABLES="export DATASET='webarena'; export SHOPPING='http://${SERVER}:8082';export SHOPPING_ADMIN='http://${SERVER}:8083/admin';export REDDIT='http://${SERVER}:8080';export GITLAB='http://${SERVER}:9001';export MAP='http://${MAP_SERVER}:9003';export WIKIPEDIA='http://${SERVER}:8087/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing';export HOMEPAGE='http://${SERVER}:8077';export OPENAI_API_KEY=${OPENAI_API_KEY};export OPENAI_API_URL=${OPENAI_API_URL};export OPENAI_ORGANIZATION=${OPENAI_ORGANIZATION}"

trap keep_only_current_pane EXIT

# Close additional panes
keep_only_current_pane() {
    echo "Cleaning up: Keeping only the current pane..."
    local current_pane=$(tmux display-message -p '#{pane_id}')
    tmux list-panes -F '#{pane_id}' | 
    while read pane_id; do
        if [ "$pane_id" != "$current_pane" ]; then
            tmux kill-pane -t "$pane_id"
        fi
    done
    echo "Cleanup complete."
}

keep_only_current_pane

# Function to generate splits
generate_splits() {
    local total_items=$1    
    # local num_splits=$2     
    local num_splits=$(($2 - 1))

    # Ensure num_splits isn't larger than total_items
    if [ $num_splits -gt $total_items ]; then
        num_splits=$total_items
        echo "Warning: Reducing splits to $num_splits (can't have more splits than items)" >&2
    fi
    
    local splits=()
    # Calculate actual items per split
    local items_per_split=$((total_items / num_splits))
    local remainder=$((total_items % num_splits))
    
    local current=0
    for ((i=0; i<num_splits; i++)); do
        splits+=($current)
        # Add one extra item to early splits if there's remainder
        if [ $i -lt $remainder ]; then
            current=$((current + items_per_split + 1))
        else
            current=$((current + items_per_split))
        fi
    done
    
    # Add the final number
    splits+=($total_items)
    
    echo "${splits[@]}"
}

# NEW: Ask for desired number of panes
# read -p "Enter desired number of panes: " desired_panes
# desired_panes=${desired_panes:-33}  # Default to 33 if no input

desired_panes=33 # this will initialize (33 - 1) = 32 processes

# Function to create panes dynamically
create_panes() {
    local desired=$1
    local current=$(tmux list-panes | wc -l)
    
    # Subtract 1 from desired because we already have the initial pane
    local to_create=$((desired - 1))
    
    # First split horizontally for the first split
    if [ $to_create -gt 0 ]; then
        tmux split-window -h
        to_create=$((to_create - 1))
    fi
    
    # Then create remaining panes by splitting vertically
    for ((i=0; i<to_create; i++)); do
        # Find the largest pane and split it
        largest_pane=$(tmux list-panes -F "#{pane_id} #{pane_height}" | sort -k2 -nr | head -n1 | cut -d' ' -f1)
        tmux select-pane -t $largest_pane
        tmux split-window -v
    done
    
    # Verify the number of panes
    current=$(tmux list-panes | wc -l)
    if [ $current -ne $desired ]; then
        echo "Warning: Created $current panes instead of $desired" >&2
    fi
}


# NEW: Create the desired number of panes
create_panes $desired_panes

# Function to run a job
run_job() {
    tmux select-pane -t $1
    COMMAND="python run.py \
        --instruction_path 'agent/prompts/jsons/p_webrl_chat_think.json' \
        --test_start_idx $2 \
        --test_end_idx $3 \
        --result_dir ${result_dir} \
        --test_config_base_dir ${test_config_base_dir} \
        --provider 'openai' \
        --model ${model} \
        --mode 'chat' \
        --planner_ip ${planner_ip} \
        --stop_token \"<|eot_id|>\" \
        --temperature 1.0 \
        --max_obs_length 0 \
        --max_tokens 2048 \
        --viewport_width 1280 \
        --viewport_height 720 \
        --parsing_failure_th 5 \
        --repeating_action_failure_th 5 \
        --action_set_tag webrl_id  --observation_type webrl"
    tmux send-keys "tmux set mouse on; conda activate ${CONDA_ENV_NAME}; ${ENV_VARIABLES}; until ${COMMAND}; do echo 'crashed' >&2; sleep 1; done" C-m
    sleep 3
}

TOLERANCE=2
run_batch() {
    args=("$@") # save all arguments in an array
    num_jobs=${#args[@]} # get number of arguments
    max_panes=$(tmux list-panes | wc -l)

    local status=0  # Initialize status variable

    for ((i=1; i<$num_jobs; i++)); do
        if [ $i -le $max_panes ]; then
            run_job $i ${args[i-1]} ${args[i]}
        else
            echo "Warning: Skipping job $i as pane doesn't exist" >&2
        fi
    done
    
    # Wait for all jobs to finish
    while tmux list-panes -F "#{pane_pid} #{pane_current_command}" | grep -q python; do
        sleep 100  # wait for 10 seconds before checking again
    done

    # Run checker
    max_checks=2
    check_count=0
    while true; do
        if [ $check_count -ge $((max_checks - 1)) ]; then
            # Last round: don't use --delete_errors
            if python scripts/check_error_runs.py ${result_dir} --tolerance ${TOLERANCE}; then
                echo "Check passed on final attempt. Exiting..."
                break
            else
                echo "Check failed on final attempt. Exiting..."
                # exit 1
                status=1
                break

            fi
        elif ! python scripts/check_error_runs.py ${result_dir} --delete_errors --tolerance ${TOLERANCE}; then
            check_count=$((check_count + 1))
            
            if [ $check_count -ge $max_checks ]; then
                echo "Maximum number of checks ($max_checks) reached. Exiting..."
                # exit 1
                status=1
                break

            fi
            
            echo "Check failed (attempt $check_count of $max_checks), rerunning jobs..."
            for ((i=1; i<$num_jobs; i++)); do
                run_job $i ${args[i-1]} ${args[i]}
            done

            # Wait for all jobs to finish
            while tmux list-panes -F "#{pane_pid} #{pane_current_command}" | grep -q python; do
                sleep 100  # wait for 100 seconds before checking again
            done
        else
            echo "Check passed. Exiting..."
            break
        fi
    done

    return $status  # Return the status from the function

}

# NEW: Function to count test files
count_test_files() {
    local test_dir=$1
    local count=$(ls -1 "$test_dir"/*.json 2>/dev/null | wc -l)
    echo $count
}

total_items=$(count_test_files "$test_config_base_dir")
echo "Found $total_items test files in $test_config_base_dir"

# Generate splits and run batch
num_panes=$(tmux list-panes | wc -l)
splits=($(generate_splits $total_items $num_panes))
echo "Generated splits: ${splits[@]}"
run_batch "${splits[@]}"