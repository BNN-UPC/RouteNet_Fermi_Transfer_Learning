# Bridging the Gap Between Simulated and Real Network Data Using Transfer Learning 

**Carlos Güemes Palau, Miquel Ferrior Galmés, Jordi Paillisse Vilanova, Albert López Brescó, Pere Barlet Ros, Albert Cabellos Aparicio**

This repository is the code of the paper *Bridging the Gap Between Simulated and Real Network Data Using Transfer Learning* (publication pending)

Contact us: *[carlos.guemes@upc.edu](mailto:carlos.guemes@upc.edu)*, *[contactus@bnn.upc.edu](mailto:contactus@bnn.upc.edu)*

## Abstract

Currently, most network Machine Learning (ML) models are built from simulated data. This occurs as obtaining captured network data is expensive, and certain scenarios, like network-breaking ones, rarely appear. However, relying solely on simulated data results in less accurate and reliable models when deployed in production environments, due to the differences between the two. For example, the network may use a router whose behavior can only be approximated as its implementation details are private.
In this paper, we propose using transfer learning to build a ML model using simulated and real-world network data, obtaining a more accurate model than just using the latter. We tested this by training a modified RouteNet-Fermi network model, using data generated by the OMNeT++ simulator and our own testbed network. We found that correctly configuring the fine-tuning process reduces the MAPE up to 82% compared to a model trained only with network data.

# Quickstart

1. Please ensure that your OS has installed Python 3 (ideally 3.9)
2. Create the virtual environment and activate the environment:
```bash
virtualenv -p python3 myenv
source myenv/bin/activate
```
3. Then we install the required packages (to avoid issues, make sure to install the specific package versions, especially for TensorFlow):
```bash
pip install tensorflow==2.15.0 numpy==1.26.3 matplotlib==3.8.2 notebook==7.0.7
```

Once those are ready you can:
- Train the baseline/donor model [`train.py`](train.py) and fine-tuned model [`fine_tuning.py`](fine_tuning.py).
- Evaluate the trained models [`evaluation.ipynb`](evaluation.ipynb).

# Repository structure

The repository contains the following structure:
- `ckpt`: Folder containing the checkpoints used in the paper evaluation.
- `data`: Folder containing the datasets used in the paper.
- `normalization`: Folder containing the z-score normalizations used by the trained checkpoints (internal path should match the `ckpt` directory).
- [`train.py`](train.py): script to train a RouteNet-Fermi model normally, without fine-tuning.
- [`fine_tuning.py`](fine_tuning.py): script to fine-tune a RouteNet-Fermi model
- [`evaluation.ipynb`](evaluation.ipynb): notebook folder to evaluate the trained models.
- [`utils.py`](utils.py) contains auxiliary functions common in the previous three files.
- [`models.py`](models.py) contains our modified implementation of RouteNet-Fermi.
- [LICENSE](LICENSE): see the file for the full license.

# Modifying the scripts

The scripts contain the default hyperparameters and configurations used in the paper. Follow the comments in the code to perform your modifications. Here's a quick reference guide for both [`train.py`](train.py) and [`fine_tuning.py`](fine_tuning.py):

## Modifying the `train.py` script:

- Use the `RUN_EAGERLY` variable (line 172) to run TensorFlow in eager mode.
- Use the `RELOAD_WEIGHTS` variable (line 176) to resume training from a specific checkpoint.
- Use the `MAX_STEPS` variable (line 178) to modify the maximum number of steps per epoch.
- Modify the experiment configuration to change aspects such as the dataset used (lines 181-186)
- Change the optimizer (and its hyperparameters) and the loss function on lines 223 and 225.
- Model definition and the remainder of its hyperparameters can be changed on its instantiation (lines 226-240) and the call to fit the model (lines 285-298)

## Modifying the `fine_tuning.py` script:

- Use the `RUN_EAGERLY` variable (line 185) to run TensorFlow in eager mode.
- Use the `RELOAD_WEIGHTS` variable (line 189) to resume training from a specific checkpoint.
- Use the `STORE_SUMMARY` variable (line 192) to print and save to file a summary of the model.
- Use the `MAX_STEPS` variable (line 178) to modify the maximum number of steps per epoch.
- Modify the donor experiment selection at lines 198-203.
- Modify the experiment configuration to change aspects such as the dataset used (lines 215-226)
- Change the optimizer (and its hyperparameters) and the loss function on lines 263-265.
- Model definition and the remainder of its hyperparameters can be changed on its instantiation (lines 266-276) and the call to fit the model (lines 330-343)

# License

See the [file](LICENSE) for the full license:


```
Copyright 2024 Universitat Politècnica de Catalunya

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```