### Triton benchmarks

### Installation

First, install [uv](https://github.com/astral-sh/uv) to avoid package version problems.

Then:
```sh
# Create virtualenv
uv venv venv
source venv/bin/activate

# Install requirements
uv pip install -r requirements.txt

# Override triton version
uv pip install triton==3.3.0
```

### Running

```sh
python main.py
```

This will run 3 successive benchmarks and report the timings.