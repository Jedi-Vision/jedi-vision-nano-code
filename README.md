# Jedi Vision Project Code for Jetson Nano

## Contains
* Scene representation conversion
  * Monocular depth model
* Environment representation
  * YOLO model inference
* Object representation
  * Grounded object tracking model
* Audio representation
  * Python bindings to Steam Audio

## Installation

1. First off, make you have [Poetry](https://python-poetry.org/docs/) installed.

2. Then clone the repository.

    via HTTP
    ```bash
    git clone https://github.com/Jedi-Vision/jedi-vision-nano-code.git
    ```

    via SSH
    ```bash
    git clone git@github.com:Jedi-Vision/jedi-vision-nano-code.git
    ```

3. (Optional) If you'd like poetry to automatically create a virtual environment *in the project* run the following.

    ```bash
    poetry config virtualenvs.in-project true
    ```

    Doing this means that the virtual environment won't get installed to a random location, which can make it easier for IDE's like VSCode to find the environment.

4. Navigate to the project folder and install the dependencies

    Then run:
    ```bash
    poetry install
    ```

5. Activate the virtual environment

    Run the following to get the activation command for the Poetry virtual environment.
    ```bash
    poetry env activate
    ```

    If you configured Poetry to install the virtual environment in the project you can just run:
    ```bash
    source .venv/bin/activate
    ```

6. (Optional) If you are running into a *"No module named jv"* error, try installing our `jv` library in editable mode.

    ```bash
    pip install -e .
    ```