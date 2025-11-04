# Jedi Vision Project Code for Jetson Nano

## Contains
* Scene representation conversion
  * Monocular depth model
* Environment representation
  * SegFormer model inference
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

3. Navigate to the project folder and install the dependencies

    ```bash
    poetry install
    ```

4. Activate the virtual environment

    ```bash
    source .venv/bin/activate
    ```