# RL-SCS

A reinforcement learning implementation for Standard Combat Series (SCS) - a hexagonal tile-based strategy game.

**PettingZoo AEC API Compatible** ✅

## Installation

It is recommended to create a virtual environment before installing the dependencies:

### 1. Create a virtual environment

```bash
python3 -m venv venv
```

### 2. Activate the virtual environment

On Linux/macOS:
```bash
source venv/bin/activate
```


### 3. Install dependencies

For regular use:
```bash
pip install -r requirements.txt
```

For development:
```bash
pip install -r requirements-dev.txt
```

## Project Structure

```
RL-SCS/
├── src/                            # Package root
│   ├── SCS_Game.py                # Main game environment (PettingZoo AEC)
│   ├── SCS_Renderer.py            # Rendering logic
│   ├── Terrain.py                 # Terrain definitions
│   ├── Tile.py                    # Board tiles
│   ├── Unit.py                    # Game units
│   ├── _utils.py                  # Internal utilities
│   ├── tests/                     # Test suite
│   │   ├── test_pettingzoo.py    # Standard API tests
│   │   ├── test_comprehensive.py # Multi-config tests
│   │   └── test_stress.py         # Stress tests
│   ├── assets/                    # Game assets (images, etc.)
│   └── example_configurations/    # Game configuration files
├── requirements.txt               # Core dependencies
└── requirements-dev.txt           # Development dependencies
```

## Usage


## Development

### Running Tests During Development


# Run tests
pytest
```


## License

[Add your license here]

## Contributors

[Add contributors here]
