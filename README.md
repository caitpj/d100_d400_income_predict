# d100_d400_income_predict
OVERVIEW


## Set-up

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop) installed on your machine.

This repository provides a reproducible Docker environment pre-configured with Python, Conda, and the [UCIMLRepo](https://github.com/uci-ml-repo/ucimlrepo) library used for simple data access.

It uses **Miniforge** to minimize image size and prioritize the `conda-forge` channel.

### 1. Clone the Repository
```bash
git clone https://github.com/caitpj/d100_d400_income_predict.git
cd d100_d400_income_predict
```

### 2. Build the Docker Image
Run the following command in your terminal to build the image. We are tagging it as conda-uciml for easy reference.
```bash
docker build -t conda-uciml .
```

### 3. Run the Container
This runs the container, prints the location of the installed package to verify success, and then exits.
```bash
docker run --rm conda-uciml
```

## How to Run Python Scripts

Since the dependencies (`ucimlrepo`, etc.) live inside the Docker container, you cannot simply run `python script.py` on your local machine. You must run your scripts **through** the Docker container.

### The Command
Use the following command to run any Python script in this repository:

```bash
docker run --rm -v "$(pwd):/app" conda-uciml python your_script_name.py
```

For ease of use consider an alias e.g.
```bash
alias dpython='docker run --rm -v "$(pwd):/app" conda-uciml python'
```
Now you can simply run `dpython script.py`


## Project Structure
