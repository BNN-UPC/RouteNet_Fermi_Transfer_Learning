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

# When running with RouteNet, it is recommended to disable GPU due to the tf.gather
# operations present being more efficient in CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

from models import RouteNet_temporal_delay

from random import seed
import pickle
import numpy as np
from utils import (
    CustomEarlyStop,
    get_positional_denorm_mape,
    get_experiment_path,
    prepare_targets_and_mask,
    log_transform,
)


def get_temporal_z_scores(
    ds,
    params,
    include_y=None,
    flatten=False,
    summarize=-1,
    only_positive=False,
    store_res_path=None,
    check_existing=False,
):
    """
    Get the mean and the std for different parameters of a dataset. Later used by the
    models for the z-score normalization.

    Parameters
    ----------
    ds: tensorflow.data.Dataset
        Dataset to extract the mean and std from. MUST be training dataset to the model
    params: List[str]
        Parameters to extract the mean and std from.
    include_y, Optional[bool]
        Indicates if to also extract the features of the output variable. Inputs
        indicate the string key used on the return dict. If None, it is not included.
    flatten, Optional[bool]
        If true, mean and std are computed globally for all dimensions in each feature.
        Otherwise, the last dimension is performed. For example, if a feature is of
        shape [a, b, c], the 'a' and 'b' and flattened, and the mean and std are
        computed for 'c'.
    summarize, int
        If > 0, only uses the first n samples to compute the mean and std.
    store_res_path, Optional[bool]
        If not None, the results are stored in the path indicated by the string.
        The dictionary is stored using the pickle library.

    Returns
    -------
    dict
        Dictionary containing the min and the max-min for each parameter.
    """
    # If check_existing is True, check if the file exists and return the dict (if so)
    if store_res_path is not None and check_existing:
        if os.path.exists(store_res_path):
            with open(store_res_path, "rb") as ff:
                return pickle.load(ff)

    def _mean(arr):
        if only_positive:
            valid_pos = (arr > 0).astype(int).sum(axis=0)
            return arr.sum(axis=0) / valid_pos
        return arr.mean(axis=0)

    def _std(arr):
        if only_positive:
            return np.array(
                [np.std(arr[arr[:, ii] > 0, ii]) for ii in range(arr.shape[1])]
            )
        return arr.std(axis=0)

    # Use first sample to get the shape of the tensors
    if flatten:
        params_dims = {param: 1 for param in params}
        if include_y:
            params_dims[include_y] = 1
    else:
        next_sample = next(iter(ds))
        sample, label = next_sample[0], next_sample[1]
        params_dims = {param: sample[param].numpy().shape[-1] for param in params}
        if include_y:
            params_dims[include_y] = label.numpy().shape[-1]

    # Init param list
    params_list = {param: [] for param in params}
    if include_y:
        params_list[include_y] = []

    # Include the rest of the samples
    for ii, (sample, label) in enumerate(iter(ds)):
        if summarize > 0 and ii > summarize:
            break
        for param in params:
            params_list[param].append(
                sample[param].numpy().reshape(-1, params_dims[param])
            )
        if include_y:
            params_lists[include_y].append(
                label.numpy().reshape(-1, params_dims[include_y])
            )

    # Flatten and concatenate
    params_lists = {
        param: np.concatenate(param_list, axis=0)
        for param, param_list in params_list.items()
    }

    scores = dict()
    # If possible (param_size == 1), flatten
    for param, param_list in params_lists.items():
        param_mean = _mean(param_list)
        if param_mean.size == 1:
            param_mean = param_mean[0]

        param_std = _std(param_list)
        # Check if std is 0
        if param_std.size == 1:
            param_std = param_std[0]
            if param_std == 0:
                print(f"Z-score normalization Warning: {param} has a std of 0.")
                param_std = 1
        else:
            if np.any(param_std == 0):
                print(
                    "Z-score normalization Warning:",
                    f"Several values of {param} has a std of 0.",
                )
                param_std[param_std == 0] = 1

        scores[param] = [param_mean, param_std]

    if store_res_path is not None:
        store_res_dir, _ = os.path.split(store_res_path)
        os.makedirs(store_res_dir, exist_ok=True)
        with open(store_res_path, "wb") as ff:
            pickle.dump(scores, ff)

    return scores


# Set all seeds
SEED = 1
seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

# RUN EAGERLY -> True for debugging
RUN_EAGERLY = False
tf.config.run_functions_eagerly(RUN_EAGERLY)
# RELOAD_WEIGHTS -> True to continue training from a checkpoint: use an int to specify
# the epoch to start from.
RELOAD_WEIGHTS = False
# MAX_STEPS -> Maximum number of samples (network scenarios) per epoch
MAX_STEPS = 500

# EXPERIMENT CONFIGURATION
ds_name = "data_seg_poisson_on_off_simulated_0_4_100"
experiment_name = "baselines"
model_class = RouteNet_temporal_delay
variant = f"{MAX_STEPS}_steps"
target = "avg_delay"
mask = f"flow_has_{target.split('_')[1]}"

# Extract unique path for the current experiment
experiment_path = get_experiment_path(
    experiment_name,
    ds_name,
    model_class.__name__,
    target,
    variant=variant,
)

# Dataset selection: ds_name is used to load the dataset. Log transform is applied so
# that the loss is computed over the log-mse. Samples are also shuffled
ds_train = (
    tf.data.Dataset.load(f"data/{ds_name}/training", compression="GZIP")
    .prefetch(tf.data.experimental.AUTOTUNE)
    .map(prepare_targets_and_mask([f"flow_{target}_per_seg"], mask))
    .map(log_transform)
)
ds_train = ds_train.shuffle(len(ds_train), seed=SEED, reshuffle_each_iteration=True)


# If the number of samples in the dataset is bigger than the MAX_STEPS,the repeat()
# function must be applied.
if ds_repeat_activate := len(ds_train) > MAX_STEPS:
    ds_train = ds_train.repeat()

# Validation data: Same steps as above, but without shuffling and calling .repeat()
ds_val = (
    tf.data.Dataset.load(f"data/{ds_name}/validation", compression="GZIP")
    .prefetch(tf.data.experimental.AUTOTUNE)
    .map(prepare_targets_and_mask([f"flow_{target}_per_seg"], mask))
    .map(log_transform)
)

# Training hyperparameters:
# Adam optimizer, lr=0.001, clipnorm=1.0
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
# Loss: Log-MSE (remember that the dataset is log-transformed for this purpose)
loss = tf.keras.losses.MeanSquaredError()
model = model_class(
    output_dim=1,
    mask_field=mask,
    log=True,
    # Z-scores are computed for the training dataset and stored in the normalization
    # directory. We sample the training dataset as to reduce the computation time.
    z_scores=get_temporal_z_scores(
        ds_train,
        model_class.z_scores_fields,
        summarize=1000,
        flatten=True,
        store_res_path=os.path.join("normalization", experiment_path, "z_scores.pkl"),
        check_existing=True,
    ),
)

# get_positional_denorm_mape returns the denormalized MAPE. NOTE: this function does the
# average over network scenario, while the loss function (and model.evaluate() in the
# evaluation notebook) will use the average over the individual flows.
model.compile(
    optimizer=optimizer,
    loss=loss,
    run_eagerly=RUN_EAGERLY,
    metrics=[get_positional_denorm_mape(0, target)],
)

ckpt_dir = f"ckpt/{experiment_path}"
latest = tf.train.latest_checkpoint(ckpt_dir)
if RELOAD_WEIGHTS and latest is not None:
    print("Found a pretrained model, restoring...")
    model.load_weights(latest)
else:
    print("Starting training from scratch...")

filepath = os.path.join(ckpt_dir, "{epoch:02d}-{val_loss:.4f}")
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=filepath,
    verbose=1,
    mode="min",
    save_best_only=False,
    save_weights_only=True,
    save_freq="epoch",
)
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=f"tensorboard/{experiment_path}", histogram_freq=1
)
reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    factor=0.5,
    patience=10,
    verbose=1,
    cooldown=3,
    mode="min",
    monitor="loss",
    # factor=0.5, patience=3, verbose=1, mode="min", monitor="loss"
)
# Early stop that works when min learning rate is surpassed
early_stop_callback = CustomEarlyStop(min_lr=1e-6)

# Fit model
model_fit_kwargs = {
    "x": ds_train,
    "epochs": 10000,
    "validation_data": ds_val,
    "callbacks": [
        cp_callback,
        tensorboard_callback,
        reduce_lr_callback,
        tf.keras.callbacks.TerminateOnNaN(),
        early_stop_callback,
    ],
    "use_multiprocessing": True,
    "initial_epoch": 0 if not RELOAD_WEIGHTS else RELOAD_WEIGHTS,
}
if ds_repeat_activate:
    model.fit(steps_per_epoch=MAX_STEPS, **model_fit_kwargs)
else:
    model.fit(**model_fit_kwargs)
