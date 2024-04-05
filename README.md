"# UTrack3D" 

**Environments**: Python3.7 & Matlab2021

**System**: Windows 11

## Install prerequisites

* Python: We test our code using Python3.7. Advanced Python versions work as well. 
            https://www.python.org/downloads/


* Matlab: We test our code using Matlab2021. Advanced Matlab versions work as well. The matlab is only used for performance evaluation in this code. In case you prefer not installing Matlab, several pre-generated examples are provided.

* Python libraries: `pip install -r requirements.txt`

## Running

* Run script `run_offline_analysis.py` for tracking. 

`python run_offline_track.py`

This reads pre-stored CIR data (./raw_data) and generates a file `tracking_results.mat` in ./output which stores the estimated trajectory and ground-truth trajectory. We provide three examples (test1, test2, test3). One can modify the following line to test a specific example.

`file_dir = "./raw_data/test1/"`

* Run script `./matlab_analysis_scripts/evaluate_accuracy_ae.m` to compute error and perform visualization. This script takes `tracking_results.mat` as inputs and generates CDF error plot and trajectory visualization in the current folder.