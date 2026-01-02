#!/bin/bash
# Activate virtual environment
# Note: On Windows Git Bash, 'Scripts' is standard for venv created by Windows python
if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found."
    exit 1
fi

# Set PYTHONPATH to include src so imports work
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Run the harness GUI
python scripts/harness_gui.py

# Keep window open if double-clicked (optional, mostly for batch files, but useful if run in a terminal that closes)
read -p "Press enter to continue..."
