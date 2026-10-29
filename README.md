# rdbr_fall_2026

## Setup

Make sure you have `python` and `pip` installed.
```bash
python -V
pip -V
```

Create a virtual environment.
```bash
python venv -m .env
```

Activate the environment (command depends on operating system).
```bash
.env/Scripts/Activate # windows
source .env/bin/activate # linux
```

Install dependencies.
```bash
pip install pyvista fury imgui_bundle
```

## Run

Print the help message.

```bash
python src/main.py -h
```

Spin up a simulatation for a given model with the run subcommand.

```bash
python src/main.py run <model-name>
```

Currently, the following models are avaliable:
* DoublePendulum
* Quadruped
* Lever
* Slider

Models are located in `src/models`

## Playback

Record and playback a simulation run:
```bash
python src/main.py run <model-name> --record=<file-name>
python src/main.py playback <file-name>
```

## Testing

Tests are located in `src/tests`.

```bash
python -m unittest src/tests
```
