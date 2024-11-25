"""
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
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf


class CustomEarlyStop(tf.keras.callbacks.Callback):
    """Callback that stops training when a low enough learning rate is reached.

    Parameters
    ----------
    min_lr : float
        Minimum learning rate before stopping training.
    """

    def __init__(self, min_lr=1e-6):
        super(CustomEarlyStop, self).__init__()
        self.min_lr = min_lr

    def on_epoch_end(self, epoch, logs=None):
        if logs["lr"] < self.min_lr:
            self.model.stop_training = True


def get_positional_denorm_mape(pos, name):
    """Returns a function to compute de denormalized MAPE at training. Expects the model
    to output a two-dimensional tensor (batch, feature). 'pos' argument specifies which
    feature to use for the metric.

    Parameters
    ----------
    pos : int
        Position of the target variable in the output.
    name : name
        Name of the target variable (to be used in tensorboard).
    """

    def denorm_mape(y_true, y_pred):
        y_true = tf.expand_dims(tf.math.exp(y_true[:, pos]), 1)
        y_pred = tf.expand_dims(tf.math.exp(y_pred[:, pos]), 1)
        return tf.reduce_mean(tf.abs((y_true - y_pred) / y_true)) * 100

    denorm_mape.__name__ = f"denorm_mape_{name}_metric"
    return denorm_mape


def get_experiment_path(
    experiment_name,
    ds_name,
    model_name,
    target,
    fine_tune_options=None,
    variant=None,
    og_ds_name=None,
):
    """Generates a unique experiment path based on the experiment parameters.

    Parameters
    ----------
    experiment_name : str
        Experiment batch name
    ds_name : str
        Dataset name
    model_name : str
        Model name
    target : str
        Perfomance metric to be predicted by the model
    fine_tune_options : str, optional
        Descriptor string indicating fine tune operations, by default None
    variant : str, optional
        Additional experiment discriminating descriptor, by default None
    og_ds_name : str, optional
        If fine tuning, the donor dataset name, by default None

    Returns
    -------
    _type_
        _description_
    """
    experiment_path = f"{experiment_name}/{ds_name}"
    if og_ds_name is not None:
        experiment_path += f"/og_ds_{og_ds_name}"
    experiment_path += f"/{model_name}"
    if fine_tune_options not in [None, ""]:
        experiment_path += f"/{fine_tune_options}"
    if variant not in [None, ""]:
        experiment_path += f"/{variant}"
    experiment_path += f"/{target}"
    return experiment_path


def log_transform(x, y):
    """Apply log transformation to output variable.

    Parameters
    ----------
    x: dict
        Predictor variables
    y: tf.Tensor
        Output variable

    Returns
    -------
    dict
        Predictor variables
    tf.Tensor
        Transformed output variable
    """
    return x, tf.math.log(y)


def prepare_targets_and_mask(targets, mask, output_dim=1):
    """Pre-process the samples by selecting the target variables and applying the mask.
    The mask is used to only selecting valid flows (with generated packets) per window.

    Parameters
    ----------
    targets : List[str]
        List of targest
    mask : str
        Mask feature
    output_dim : int, optional
        Number of output dimensions, by default 1

    Returns
    -------
    Function
        Function to be mapped to the tf.data.Dataset for the processing to take place.
    """
    assert output_dim > 0, "tile_mask must be greater than 0"

    def modified_target_map(x, y):
        reshaped_mask = tf.expand_dims(_seg_to_global_reshape(x[mask], num_dims=2), 1)
        if output_dim > 1:
            reshaped_mask = tf.tile(reshaped_mask, [1, output_dim])

        return x, tf.concat(
            [
                tf.reshape(
                    tf.boolean_mask(_seg_to_global_reshape(x[target]), reshaped_mask),
                    (-1, output_dim),
                )
                for target in targets
            ],
            axis=1,
        )

    return modified_target_map


def _seg_to_global_reshape(tensor, num_dims=3):
    """Function that modifies the shape of the tensor, flattenging the window dimension

    Parameters
    ----------
    tensor : tf.tensor
        Input tensor
    num_dims : int, optional
        Size of last dimension, by default 3

    Returns
    -------
    tf.tensor
        Reshaped tensor
    """
    assert num_dims > 1
    perms = [1, 0] + list(range(2, num_dims))
    total_flows = tf.shape(tensor)[0] * tf.shape(tensor)[1]
    if num_dims == 2:
        new_shape = (total_flows,)
    else:
        new_shape = tf.concat([[total_flows], tf.shape(tensor)[2:]], axis=0)
    return tf.reshape(tf.transpose(tensor, perms), new_shape)
