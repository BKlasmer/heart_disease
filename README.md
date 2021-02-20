# Heart Disease

Add description here.

## Creating & using the virtual environment
- First, check that you satisfy the [Dependencies](#dependencies) for this project.
- This project should run on its own virtual environment. To do so you need to run the following command (from the base of the module):
  ```bash
  foo@bar:~$ ./provisioning/provision_venv.sh
  ```
- If you want to work on it, you can run (from the base of the module):
  ```bash
  foo@bar:~$ source ./venv/bin/activate
  ```
- To exit from it:
  ```bash
  foo@bar:~$ deactivate
  ```
- You can create the virtual environment in another location by supplying it as the first input parameters (this is useful when deploying this software in production):
  ```bash
  foo@bar:~$ ./provisioning/provision_venv.sh /some/other/path/
  ```

## Jupyter Notebooks
- First build and activate the virtual environment as shown in [Jupyter Notebooks](#Jupyter Notebooks)
  ```bash
  foo@bar:~$ ./provisioning/provision_venv.sh
  foo@bar:~$ source ./venv/bin/activate
  ```
- Then go to the terminal and change directory to ./notebooks in the root of the repo
  ```bash
  foo@bar:~$ cd notebooks
  ```
- Launch Jupyter notebook by typing the following command in the terminal:
  ```bash
  foo@bar:~$ jupyter notebook
  ```
- now, a new page in your browser will open where you can create and run Jupyter Notebooks with the module


## Packaging
- First, check that you satisfy the [Dependencies](#dependencies) for this project.
- Check that your `setup.py` is up-to dated.
- Run the following script to generate the wheel :
  ```bash
  foo@bar:~$ ./provisioning/build_package.sh
  ```
- The wheel is then generated under : `./dist/`

## Tests
- Unit tests can be run by issuing the following command (from the base of the module):
  ```bash
  foo@bar:~$ ./run_tests.sh
  ```

## Dependencies
Due to limitation with Python Virtual Environment, some libraries must be installed on the
base machine. This code will then create symbolic link with the virtual environment.
### Python 3
Supported versions: from 3.6 (included) to 3.8 (excluded). (_This is due to an old version of numpy_)

On latest Mac you need to downgrade your python version:
```bash
foo@bar:~$ brew install python@3.7
  ```
  and then follow the instructions to link this version in your path.

### Python virtual environment :

On Ubuntu:
```bash
foo@bar:~$ sudo apt install python3-virtualenv
```
On Mac:
```bash
foo@bar:~$ pip3 install virtualenv
  ```


## Directory structure
```bash
project
│
├───heart_disease ( This project module itself )
│
├───build ( Build outputs for wheel generation)
│
├───notebooks ( a folder where all Jupytor notebooks are kept )
│
├───dist ( Distribution files. For this repo; Python wheels )
│
├───provisioning ( Provisioning scripts to build and package module )
│
├───test ( Test code files: Unit test )
|
├───venv ( Python virtual environment )
│
|   .pre-commit-config.yaml (the pre-commit hooks configuration file)
|   .flake8 (the flake8 configuration file)
|   README.md ( This readme file)
|   requirements.txt ( Python packages required packages in order to run this project )
│   run_tests.sh ( Runs unit tests for the project )
|   setup.py ( Project packaging configuration file )
```