# RL-SCS

A reinforcement learning implementation for Standard Combat Series (SCS) - a hexagonal tile-based strategy game.

## Installation

It is recommended to create a virtual environment before installing the dependencies:

**Create a virtual environment:**
```bash
python3 -m venv venv
```

**Activate the virtual environment:**

On Linux/macOS:
```bash
source venv/bin/activate
```

On Windows:
```bash
venv\Scripts\activate
```

**Install the required dependencies:**
```bash
pip install -r requirements.txt
```

## Dependencies

- numpy - For numerical computations and array operations
- torch - PyTorch for deep learning and neural network models
- PyYAML - For loading game configuration files
- termcolor - For colored terminal output
- pygame - For game rendering and visualization
- ray - For distributed computing and parallel processing


## Usage

Load and run a game using one of the configuration files:

```python
from SCS_Game import SCS_Game

game = SCS_Game("Game_configs/test_config.yml")
```

