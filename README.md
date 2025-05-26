# Big 2 Server

Play Big 2 against agents trained with deep reinforcement learning (see <a href="https://github.com/henrycharlesworth/big2_PPOalgorithm">this project</a> for more details). Rules can be found <a href="https://github.com/henrycharlesworth/big2_PPOalgorithm/blob/master/rules.md">here</a>.


# Usage
```bash
# Create a virtual environment with conda
conda create -n big2 python=3.13
conda activate big2

# or create a virtual environment with venv
python3 -m venv big2
source big2/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run web game
```bash
python manage.py runserver
```
Once the server is running, open your browser and go to [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

### game simulation
```bash
python run_random_agents.py
```

### train CNN
Will load `cnn_agent_best.pt` and train a new CNN agent.
```bash
python trainCNN.py
```