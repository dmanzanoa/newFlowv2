# OpticalFlow

## Setup

### Python Setup Linux|Mac
```
python -m venv venv
source venv/bin/activate
pip install -e .
```

### NixOS
```
nix-shell
```

## Run Agent

**Run all scripts from the top level of the repo**

```
# process one file

hl agent run agents/my_agent.json -f inputs/test.mp4

```
