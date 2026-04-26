# pinn-tutorial: An introductory course on Physics-Informed Neural Networkss


**Author:** Ivan Debono

This repository contains code and notebooks for learning and experimenting with Physics-Informed Neural Networks (PINNs).



## Setup Instructions

### 1. Clone the Repository

```
git clone <repo-url>
cd PINN_tutorial
```

### 2. Install Everything with Make

Simply run:

```
make setup
```

This will create a virtual environment, install all dependencies, and set up the Jupyter kernel.

### 3. Activate the Environment

```
source .venv/bin/activate
```

### 4. Run Notebooks or Scripts

You can now run the notebooks (e.g., `Introduction.ipynb`, `Session1.ipynb`, etc.) or Python scripts (e.g., `burgers_1d_eqn.py`).

---

## Project Structure

- `src/` — Source code for PINN models and trainers (I will add these in future)
- `*.ipynb` — Tutorial and exercise notebooks
- `Makefile` — Useful commands for setup and cleaning
- `pyproject.toml` — Project metadata and dependencies

## Cleaning Up

To remove temporary files and caches, or reinstall the repository:

```
make clean
```

## Troubleshooting

- Ensure you are using Python 3.14 or higher.
- If you encounter issues with `uv`, try upgrading it: `pip install -U uv`.
- For Jupyter, ensure the correct kernel is selected.

---

For more details, see the code and notebook files in this repository.
