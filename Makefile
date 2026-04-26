# Makefile for setting up the repository

setup:
	uv venv --python 3.14
	uv pip install -e. 
	uv add ipykernel
	uv add numpy torch matplotlib
	.venv/bin/python -m ipykernel install --user --name=pinn-tutorial --display-name="Python (pinn-tutorial)"

clean:
	rm -rf __pycache__ *.pyc *.pyo .pytest_cache .ipynb_checkpoints
