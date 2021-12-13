# Turbofan Engine Degradation Prognostics

This project aims to predict the Remaining Useful Life (RUL) for turbofan engines based on some sensor values.
The engine degradation was simulated using C-MAPSS under different conditions, and the dataset is provided by NASA [here](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan).

## Usage

Several notebooks are provided to explore this problem.

The first notebook  `EDA.ipynb` is for exploratory data analysis, for getting familiar with the
nature and shape of the data, and get an idea of how the sensors evolve for the different datasets.

The second notebook `regression.ipynb` applies different regressors to the RUL prediction problem,
while varying different parameters involved in building the train/test datasets.

The third notebook, `LSTM.ipynb` tackles the prediction problem with more complex models, namely
an LSTM and an GRU.


## Setup

## Quickstart

Using pip: `pip install -r requirements.txt`

Using Docker: `./run.sh`

Using Venv:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## General Setup

Generally, you need to have Python 3.5 or higher installed with the dependencies listed in
`requirements.txt` to run the code in this repo. Using `pip`, these dependencies can be installed
by running `pip install -r requirements.txt`.

**Note:** all references to `python` assume Python 3 which could be equivalently run as `python3`
depending on your setup.

**Note:** additional dependencies are needed to run the Jupyter notebooks in this repo, but
they are not included in the `requirements.txt`. If you want to install these dependencies,
you can run: `pip install jupyter jupyterlab`. However, these dependencies are contained
in the Docker image.

## Running with venv

You can create a virtual environment using `venv` and install the dependencies, for example:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running with Docker

### Docker setup

First and foremost, you need to have Docker installed.\
Build Docker image, containing all required dependencies with Python 3.9,
by changing directories to this repository and running:
```bash
docker build -t prognostics .
```

After the Docker image is built, it can be used by running:
```bash
docker run -it --rm -v $PWD:/work --entrypoint /bin/bash prognostics
```
This will launch a bash session inside the Docker container, which can then be used interactively.
The `bash` command can be replaced with any other command, as needed. The argument `-v $PWD:/src`
mounts your current path to the `/src` directory in the Docker container. As a consequence,
any changes made inside the `/src` directory in the Docker container will be reflected in your
local directory, and vice versa.

**Note:** changes made to non-mounted directories inside the Docker container, as well as any
system changes such as dependency installation to not persist between different sessions.

### Docker & Jupyter

The Docker image contains the dependencies needed for running a Jupyter server. In fact,
Jupyter is the default entrypoint. You can use the utility script which will launch build the Docker image and run a container in daemon which will launch a Jupyter server.
```bash
# make sure script is executable (only first time)
chmod +x run.sh
# run script
./run.sh
```
To stop the running container: `docker stop prognostics`
