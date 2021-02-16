# LMB Tracker
With this repository we ([phquentin](https://github.com/phquentin), [maxschnettler](https://github.com/maxschnettler)) provide a python implementation of the Labeled Multi-Bernoulli Filter for Multiple Target Tracking. The theoretical framework of this implementation is based on the papers [1] [2] and part of the code is based on https://github.com/jonatanolofsson/lmb. The goal of this repository is to apply the tracker on the [MOTChallenge](https://motchallenge.net/) and the [nuScenes tracking challenge](https://www.nuscenes.org/tracking?externalData=all&mapData=all&modalities=Any). By providing a well documented open source code, we aim to make the Labeled Multi-Bernoulli-Filter better accessible for a broader community.

[1] Reuter, A. Danzer, M. Stübler, A. Scheel and K. Granström, "A fast implementation of the Labeled 
    Multi-Bernoulli filter using gibbs sampling," 2017 IEEE Intelligent Vehicles Symposium (IV), Los Angeles, 
    CA, 2017, pp. 765-772, doi: 10.1109/IVS.2017.7995809.
    
[2] Olofsson, J., Veibäck, C., & Hendeby, G. (2017). Sea ice tracking with a spatially indexed labeled 
    multi-Bernoulli filter. In 20th International Conference on Information Fusion (FUSION). Xi’an, China. 

## Installation

### Dependencies

Install the required dependencies of the `requirements.yml`. This file can be used to directly create an environment with all required dependencies installed (e.g. using conda).

### Package installation

Use the provided `setup.py` file to install the `lmb` package in the currently activated environment. This enables its usage in the example scripts or your own modules.

````
python setup.py install
````

Using the `develop` option instead of `install` creates a symbolic link towards the package and thus enables continuous development without having to reinstall the package after changes.

## Example
To run an example of tracking multiple points in a 2D plane, activate the repository environment, `cd` into the directory /examples and run `python3 2D_point_evaluation_example.py`. Find the corresponding evaluation report in the created directory /eval_results. 

The evaluation report contains a page with the overall results, showing a plot with the trajectories of the ground truth tracks (black dots - with a bigger last dot to indicate the movement direction) and tracks estimated by the tracker (colored crosses) as well as common MOT metrics. Exemplary page of the overall results:
![](./doc/Overall_results_in_2D.png)<br/>

Furthermore, the report contains a page for every time step showing a plot of the ground truth and estimated tracks up to that time step, the estimated existence probabilities as well as the corresponding MOT events for that time step. Exemplary page for a time step:
![](./doc/Time_step_2.png)<br/>


## License
This software is released under the GPLv3 license.
