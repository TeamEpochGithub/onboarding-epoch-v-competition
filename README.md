# Onboarding Epoch V - Pokémon Type Classification

[![Epoch](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FJeffrey-Lim%2Fepoch-dvdscreensaver%2Fmaster%2Fbadge.json)](https://teamepoch.ai/)
[![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Rye](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/rye/main/artwork/badge.json)](https://rye-up.com)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with MyPy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/TeamEpochGithub/onboarding-epoch-v-competition/main.svg)](https://results.pre-commit.ci/latest/TeamEpochGithub/onboarding-epoch-v-competition/main)
[![codecov](https://codecov.io/gh/TeamEpochGithub/onboarding-epoch-v-competition/graph/badge.svg?token=gzOUyRJV5L)](https://codecov.io/gh/TeamEpochGithub/onboarding-epoch-v-competition)

Team Epoch's solution to the [Pokémon Type Classification Competition](https://www.kaggle.com/competitions/pokemon-type-classification-epoch-v-competition).

## Getting started

Clone the repository and navigate to the project directory.
Make sure [Rye](https://rye-up.com/guide/installation/) is installed on your machine and run:

```bash
rye sync
```

Alternatively, you can install the dependencies from `requirements-dev.lock` using the following command:

```bash
pip install -r requirements-dev.lock
```

## pre-commit

This repository uses [pre-commit](https://pre-commit.com/) for code quality checks and auto-formatting.
To install the pre-commit hooks, run:

```bash
pre-commit install
```

To run the pre-commit checks on all files, run:

```bash
pre-commit run --all-files
```

## Dataset

Dataset is available in the `data/raw` folder. It contains the following files:

### Metadata

`train_metadata.csv` contains various details about each Pokémon, such as its name, type, species, height, weight, abilities, EV yield, catch rate, base friendship, base exp, growth rate.

CSV Descriptors:

- `Pokemon`: Name of the Pokémon.
- `Type`: One or dual type determining STAB (same-type attack bonus) and weaknesses or resistances to incoming attacks.
- `Species`: Identifies the Pokémon based on defining biological characteristics.
- `Height`: Height of each Pokémon.
- `Weight`: Weight of each Pokémon.
- `Abilities`: Special attributes aiding Pokémon in battle, introduced in Generation 3.
- `EV Yield`: Stats gained by defeating specific Pokémon.
- `Catch Rate`: Chances of catching a Pokémon with a Poké Ball.
- `Base Friendship`: Default friendship value when encountering a Pokémon.
- `Base Exp`: EXP yield when defeating a Pokémon at level 1.
- `Growth Rate`: Amount of EXP needed for leveling up.
- `Egg Groups`: Classification used in Pokémon breeding.
- `Gender`: Chance of Pokémon being male or female.
- `Egg Cycles`: Time unit for hatching Pokémon eggs.
- `Base Stats` (HP, Attack, Defense, Special Attack, Special Defense, Speed): Determine Pokémon strengths and weaknesses.

### Images

Training images are located in the `data/raw/train_images` folder.
Test images are located in the `data/raw/test_images` folder.

These images are in-game sprites of the Pokémon, sources from all mainline Pokémon games.

## Submission

For making a submission you are expected to run inference on the `data/raw/test_images` and create a CSV file with the following columns:

```csv
id, Bug, Dark, Dragon, Electric, Fairy, Fighting, Fire, Flying, Ghost, Grass, Ground, Ice, Normal, Poison, Psychic, Rock, Steel, Water
```

An example submission file is provided in `submission/example_submission.csv`.
