# d100_d400_income_predict

## OVERVIEW

This repository provides a reproducible Docker environment pre-configured with Python, Conda, and the [UCIMLRepo](https://github.com/uci-ml-repo/ucimlrepo) library used for simple data access.

It uses **Miniforge** to minimize image size and prioritize the `conda-forge` channel.

## Set-up

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop) installed on your machine (for running the application).
- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed on your machine (for local development and git hooks).

### 1. Clone the Repository

    git clone https://github.com/caitpj/d100_d400_income_predict.git
    cd d100_d400_income_predict

### 2. Local Development Setup (Git Hooks)
To ensure code quality, we use `pre-commit` hooks that run locally on your machine before every commit. This requires a local Conda environment on your host machine (not in Docker).

    # 1. Create/Update the local environment
    # This installs pre-commit, black, mypy, etc. based on environment.yml
    conda env update --file environment.yml --prune

    # 2. Activate the environment
    conda activate d100_d300_env

    # 3. Install the git hooks
    pre-commit install

*Now, every time you run `git commit`, checks for formatting, secrets, and linting will run automatically.*

### 3. Build the Docker Image (For Running Code)
The application logic runs inside a Docker container to ensure reproducibility.

    docker build -t conda-uciml .

### 4. Run the Container
This runs the container, prints the location of the installed package to verify success, and then exits.

    docker run --rm conda-uciml

## How to Run Python Scripts

Since the dependencies (`ucimlrepo`, etc.) live inside the Docker container, you cannot simply run `python script.py` on your local machine (unless you rely on your local conda env). To use the containerized environment, you must run your scripts **through** Docker.

### The Command
Use the following command to run any Python script in this repository:

    docker run --rm -v "$(pwd):/app" conda-uciml python your_script_name.py

For ease of use consider an alias e.g.

    alias dpython='docker run --rm -v "$(pwd):/app" conda-uciml python'

Now you can simply run `dpython script.py`

## AI Use
Some code was AI generated, notably:
- Visualisations
- Pandas vs Polars benchmark test
