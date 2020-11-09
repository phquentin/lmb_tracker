# LMB Tracker

## Installation

### Dependencies

The required dependencies are listed in the provided `requirements.yml`. This file can be used to directly create an environment with all required dependencies installed (e.g. using conda).

### Package installation

Use the provided `setup.py` file to install the `lmb` package in the currently activated environment. This enables its usage in the example scripts or your own modules.

````
python setup.py install
````

Using the `develop` option instead of `install` creates a symbolic link towards the package and thus enables continuous development without having to reinstall the package after changes.
