#!/bin/bash

# Kill existing tmux session if it exists
tmux kill-session -t my_session 2>/dev/null

# Create a new tmux session
tmux new-session -d -s my_session

tmux send-keys -t my_session "conda activate har; cd backend; python main.py" C-m

tmux split-window -h

tmux send-keys -t my_session "cd frontend/human-action-intelligence/; npm run dev" C-m

tmux attach -t my_session
