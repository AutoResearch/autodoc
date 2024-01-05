# AutoDoc

[![ssec](https://img.shields.io/badge/SSEC-Project-purple?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAOCAQAAABedl5ZAAAACXBIWXMAAAHKAAABygHMtnUxAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAMNJREFUGBltwcEqwwEcAOAfc1F2sNsOTqSlNUopSv5jW1YzHHYY/6YtLa1Jy4mbl3Bz8QIeyKM4fMaUxr4vZnEpjWnmLMSYCysxTcddhF25+EvJia5hhCudULAePyRalvUteXIfBgYxJufRuaKuprKsbDjVUrUj40FNQ11PTzEmrCmrevPhRcVQai8m1PRVvOPZgX2JttWYsGhD3atbHWcyUqX4oqDtJkJiJHUYv+R1JbaNHJmP/+Q1HLu2GbNoSm3Ft0+Y1YMdPSTSwQAAAABJRU5ErkJggg==&style=plastic)](https://escience.washington.edu/software-engineering/ssec/)

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

<!-- [![PyPI](https://img.shields.io/pypi/v/autora-doc?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/autora-doc/) -->


[![GitHub Workflow Status](https://github.com/autoresearch/autodoc/actions/workflows/smoke-test.yml/badge.svg)](https://github.com/AutoResearch/autodoc/actions/workflows/smoke-test.yml)
[![codecov](https://codecov.io/gh/AutoResearch/autodoc/branch/main/graph/badge.svg)](https://codecov.io/gh/AutoResearch/autodoc)
<!-- [![Read the Docs](https://img.shields.io/readthedocs/autora-doc)](https://autora-doc.readthedocs.io/) -->

This project was automatically generated using the LINCC-Frameworks
[python-project-template](https://github.com/lincc-frameworks/python-project-template). For more information about the project template see the
[documentation](https://lincc-ppt.readthedocs.io/en/latest/).

## Dev Guide - Getting Started

Before installing any dependencies or writing code, it's a great idea to create a
virtual environment. We recommend using `conda` to manage virtual
environments. If you have conda installed locally, you can run the following to
create and activate a new environment.

```
>> conda create env -n <env_name> python=3.8
>> conda activate <env_name>
```

Once you have created a new environment, you can install this project for local
development using the following commands:

```
>> pip install -e .'[dev,train]'
>> pre-commit install
>> conda install pandoc
```

Notes:
1) The single quotes around `'[dev]'` may not be required for your operating system.
3) Look at `pyproject.toml` for other optional dependencies, e.g. you can do `pip install -e ."[dev,train,cuda]"` if you want to use CUDA.
2) `pre-commit install` will initialize pre-commit for this local repository, so
   that a set of tests will be run prior to completing a local commit. For more
   information, see the Python Project Template documentation on
   [pre-commit](https://lincc-ppt.readthedocs.io/en/latest/practices/precommit.html)
3) Install `pandoc` allows you to verify that automatic rendering of Jupyter notebooks
   into documentation for ReadTheDocs works as expected. For more information, see
   the Python Project Template documentation on
   [Sphinx and Python Notebooks](https://lincc-ppt.readthedocs.io/en/latest/practices/sphinx.html#python-notebooks)


## Running AzureML pipelines

This repo contains the evaluation and training pipelines for AutoDoc.

### Prerequisites

[Install Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli)

Add the ML extension:
```
az extension add --name ml
```

Configure the CLI:

```
az login
az account set --subscription "<your subscription name>"
az configure --defaults workspace=<aml workspace> group=<resource group> location=<location, e.g. westus3>
```


### Running jobs

Prediction
```sh
az ml job create -f azureml/eval.yml  --set display_name="Test prediction job" --set environment_variables.HF_TOKEN=<your huggingface token> --web
```

Notes:
- `--name` will set the mlflow run id
- `--display_name` becomes the name in the experiment dashboard
- `--web` argument will pop-up a browser window for tracking the job.
- The `HF_TOKEN` is required for gated repos, which need authentication


### Uploading data

Example:
```sh
az storage blob upload  --account-name <account> --container <container>> --file data/data.jsonl -n data/sweetpea/data.jsonl
 ```
