# Installation

## Setup

```bash

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install the package in editable mode
pip install -e .

# Install with test dependencies
pip install -e . --dependency-groups test
```

## Running Tests

```bash
# Run all tests except stress tests
pytest test/ -v -m "not stress"

# Run stress tests
pytest test/ -v -m stress
```
