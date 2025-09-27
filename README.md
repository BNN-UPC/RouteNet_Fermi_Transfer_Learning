# Bridging the Gap Between Simulated and Real Network Data Using Transfer Learning 

**Carlos Güemes-Palau, Miquel Ferriol-Galmés, Jordi Paillisse Vilanova, Albert López-Brescó, Pere Barlet-Ros, Albert Cabellos-Aparicio**

This repository is the code of the paper *Bridging the Gap Between Simulated and Real Network Data Using Transfer Learning* (publication pending)

Contact us: *[carlos.guemes@upc.edu](mailto:carlos.guemes@upc.edu)*, *[contactus@bnn.upc.edu](mailto:contactus@bnn.upc.edu)*

## Abstract

 Machine Learning (ML)-based network models provide fast and accurate predictions for complex network behaviors but require substantial training data. Collecting such data from real networks is often costly and limited, especially for critical scenarios like failures. As a result, researchers commonly rely on simulated data, which reduces accuracy when models are deployed in real environments. We propose a hybrid approach leveraging transfer learning to combine simulated and real-world data. Using RouteNet-Fermi, we show that fine-tuning a pre-trained model with a small real dataset significantly improves performance. Our experiments with OMNeT++ and a custom testbed reduce the Mean Absolute Percentage Error (MAPE) in packet delay prediction by up to 88%. With just 10 real scenarios, MAPE drops by 37%, and with 50 scenarios, by 48%.

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
   - **NOTE**: both the mawi simulation and testbed training datasets have been partioned into 4 and 2 segments respectively as to overcome the file size limitations present in GitHub. You can reform each with the following python script:

```python
import tensorflow as tf
ds = tf.data.Dataset.load("data/data_seg_pcaps_simulated/training0", compression="GZIP")
for ii in range(1,4):
   ds = ds.concatenate(tf.data.Dataset.load(f"data/data_seg_pcaps_simulated/training{ii}", compression="GZIP"))
ds.save("data/data_seg_pcaps_simulated/training", compression="GZIP")
```

- `normalization`: Folder containing the z-score normalizations used by the trained checkpoints (internal path should match the `ckpt` directory).
- [`train.py`](train.py): script to train a RouteNet-Fermi model normally, without fine-tuning.
- [`fine_tuning.py`](fine_tuning.py): script to (manually) fine-tune a RouteNet-Fermi model.
- [`fine_tuning_autofreeze.py`](fine_tuning_autofreeze.py): script to fine-tune a RouteNet-Fermi model using AutoFreeze ([Y. Liu, S. Agarwal, and S. Venkataraman](https://arxiv.org/abs/2102.01386)). 
- [`fine_tuning_gtot.py`](fine_tuning_gtot.py): script to fine-tune a RouteNet-Fermi model using GTOT-Tuning ([J. Zhang et al.](https://arxiv.org/abs/2203.10453)). **WARNING:** VERY memory expensive.
- [`fine_tuning_l2sp.py`](fine_tuning_l2sp.py): script to fine-tune a RouteNet-Fermi model using L2-SP ([X. LI, Y. Grandvalet, and F. Davoine](https://proceedings.mlr.press/v80/li18a.html)).
- [`evaluation.ipynb`](evaluation.ipynb): notebook folder to evaluate the trained models.
- [`utils.py`](utils.py) contains auxiliary functions common in the previous three files.
- [`models.py`](models.py) contains our modified implementation of RouteNet-Fermi.
- [LICENSE](LICENSE): see the file for the full license.

# Modifying the scripts

The scripts contain the default hyperparameters and configurations used in the paper. Follow the comments in the code to perform your modifications. Here we present a summary on how to adjust these parameters

## Modifying the `train.py` script:

- Use the `RUN_EAGERLY` variable (line 172) to run TensorFlow in eager mode.
- Use the `RELOAD_WEIGHTS` variable (line 176) to resume training from a specific checkpoint.
- Use the `MAX_STEPS` variable (line 178) to modify the maximum number of steps per epoch.
- Modify the experiment configuration to change aspects such as the dataset used (lines 181-186)
- Change the optimizer (and its hyperparameters) and the loss function on lines 223 and 225.
- Model definition and the remainder of its hyperparameters can be changed on its instantiation (lines 226-240) and the call to fit the model (lines 285-298).

## Modifying the `fine_tuning.py` script:

- Use the `RUN_EAGERLY` variable (line 99) to run TensorFlow in eager mode.
- Use the `RELOAD_WEIGHTS` variable (line 103) to resume training from a specific checkpoint.
- Use the `STORE_SUMMARY` variable (line 106) to print and save to file a summary of the model.
- Use the `MAX_STEPS` variable (line 108) to modify the maximum number of steps per epoch.
- Modify the donor experiment selection at lines 112-117.
- Modify the experiment configuration to change aspects such as the dataset used (lines 129-140)
- Change the optimizer (and its hyperparameters) and the loss function on lines 177-179.
- Model definition and the remainder of its hyperparameters can be changed on its instantiation (lines 180-191) and the call to fit the model (lines 244-257).

## Modifying the `fine_tuning_autofreeze.py` script:

- Use the `RUN_EAGERLY` variable (line 405) to run TensorFlow in eager mode.
- Use the `STORE_SUMMARY` variable (line 409) to print and save to file a summary of the model.
- Use the `MAX_STEPS` variable (line 411) to modify the maximum number of steps per epoch.
- Modify the donor experiment selection at lines 415-420.
- Modify the experiment configuration to change aspects such as the dataset used (lines 432-436)
- Change the optimizer (and its hyperparameters) and the loss function on lines 471-472.
- Model definition and the remainder of its hyperparameters can be changed on its instantiation (lines 473-483). Model's block definitions can also be re-defined (lines 503-523).
- Training and AutoFreeze parameters can be defined in the instantiation of the `AutoFreezeTrainer` class or the call to `AutoFreezeTrainer.fit` (lines 525-541).

## Modifying the `fine_tuning_gtot.py` script:

- Use the `RUN_EAGERLY` variable (line 454) to run TensorFlow in eager mode.
- Use the `STORE_SUMMARY` variable (line 458) to print and save to file a summary of the model.
- Use the `MAX_STEPS` variable (line 460) to modify the maximum number of steps per epoch.
- Modify the donor experiment selection at lines 463-468.
- Modify the experiment configuration to change aspects such as the dataset used (lines 479-483)
- Change the optimizer (and its hyperparameters) and the loss function on lines 518-519.
- Model definition and the remainder of its hyperparameters can be changed on its instantiation (lines 526-528).
- Training and GTOT-Tuning parameters can be defined in the instantiation of the `GTOTTrainer` class or the call to `GTOTTrainer.fit` (lines 552-580).

## Modifying the `fine_tuning_l2sp.py` script:

- Use the `RUN_EAGERLY` variable (line 45) to run TensorFlow in eager mode.
- Use the `RELOAD_WEIGHTS` variable (line 48) to resume training from a specific checkpoint.
- Use the `STORE_SUMMARY` variable (line 52) to print and save to file a summary of the model.
- Use the `MAX_STEPS` variable (line 54) to modify the maximum number of steps per epoch.
- Modify the donor experiment selection at lines 57-62.
- Modify the experiment configuration to change aspects such as the dataset used (lines 74-84)
- Change the optimizer (and its hyperparameters) and the loss function on lines 122-124.
- Model definition and the remainder of its hyperparameters can be changed on its instantiation (lines 142-148) and the call to fit the model (lines 203-216).

# Adjustments over other fine-tuning methods

- AutoFreeze ([Y. Liu, S. Agarwal, and S. Venkataraman](https://arxiv.org/abs/2102.01386)): AutoFreeze was originally designed for deeper neural networks without parameter sharing. To be applicable to RouteNet-Fermi, certains adjustements were made:
   - Threshold is still calculated using the raw network layer's gradients. However, freezing is done by blocks. A block was frozen only when all its layers scored under the threshold.
   - Considered blocks are more granular than those considered in manual configurations. Specifically, the encoding and readout block were subdivided, dividing thier MLP's layers as separate blocks.
   - Because of the fewer amount of layers, the percentile value when determining the threshold was decreased from the default 50th percentile to 40th percentile.
- L2-SP ([X. LI, Y. Grandvalet, and F. Davoine](https://proceedings.mlr.press/v80/li18a.html)): The regularization hyperparameter value was chosen from a small gridsearch of values, chosing the one that maximized validation MAPE. Such value was set at `1e-4`.
- GTOT-Tuning ([J. Zhang et al.](https://arxiv.org/abs/2203.10453)): Hyperparameter values follow those of the original implementation. Implementation of the MWD was following using a sparse adjacency matrix as to minimize memory usage (which still remains high).

# License

See the [file](LICENSE) for the full license:


```
Copyright 2025 Universitat Politècnica de Catalunya

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
