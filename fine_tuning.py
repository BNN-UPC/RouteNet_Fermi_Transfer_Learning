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
from keras import backend as K
from models import RouteNet_temporal_delay

from random import seed
from enum import Enum
from typing import List
import pickle
import numpy as np
from utils import (
    CustomEarlyStop,
    get_positional_denorm_mape,
    get_experiment_path,
    prepare_targets_and_mask,
    log_transform,
)


class FINETUNE_OPTIONS(Enum):
    """Enum class to define the fine tunning options."""
    FREEZE = 0
    FINETUNE = 1
    RETRAIN = 2


def get_layer_options_RouteNet_temporal_delay(
    encoding_option: FINETUNE_OPTIONS,
    mp_option: FINETUNE_OPTIONS,
    window_option: FINETUNE_OPTIONS,
    readout_option: FINETUNE_OPTIONS,
) -> List[FINETUNE_OPTIONS]:
    """Obtain the fine tunning options for each layer of the RouteNet_temporal model.
    Also returns the identifier string for the fine tunning options.

    Parameters
    ----------
    encoding_option : FINETUNE_OPTIONS
        Action to take for the encoding layers
    mp_option : FINETUNE_OPTIONS
        Action to take for the message passing layers
    window_option : FINETUNE_OPTIONS
        Action to take for the window update layers
    readout_option : FINETUNE_OPTIONS
        Action to take for the readout layer

    Returns
    -------
    List[FINETUNE_OPTIONS], str
        List of fine tunning options for each layer, identifier string
    """
    options = [
        mp_option,
        mp_option,
        mp_option,
        window_option,
        encoding_option,
        encoding_option,
        encoding_option,
        readout_option,
    ]

    # List of lists, format of [freeze_options, finetune_options, retrain_options]
    options_list = [[], [], []]
    options_list[encoding_option.value].append("encoding")
    options_list[mp_option.value].append("mp")
    options_list[window_option.value].append("window")
    options_list[readout_option.value].append("readout")
    final_string = [
        f"{option_name}_{'_'.join(options)}"
        for option_name, options in zip(["freeze", "finetune", "retrain"], options_list)
        if len(options)
    ]

    return options, "/".join(final_string)


def load_model_with_ckpt(
    model: tf.keras.Model, ckpt_path: str, layer_options: List[FINETUNE_OPTIONS]
) -> None:
    """Loads a model with donor weights according to the fine tuning options.

    Parameters
    ----------
    model : tf.keras.Model
        Reciever model
    ckpt_path : str
        Path to donor checkpoint
    layer_options : List[FINETUNE_OPTIONS]
        Fine tuning options per layer
    """
    # Save randomly initialized weights for retrain scenarios
    model_random_weights = [layer.get_weights() for layer in model.layers]
    # Load weights from checkpoint
    model.load_weights(ckpt_path)
    # Set layers
    for layer, option, layer_rng_init in zip(
        model.layers, layer_options, model_random_weights
    ):
        if option == FINETUNE_OPTIONS.FREEZE:
            layer.trainable = False
        elif option == FINETUNE_OPTIONS.FINETUNE:
            layer.trainable = True
        elif option == FINETUNE_OPTIONS.RETRAIN:
            layer.trainable = True
            layer.set_weights(layer_rng_init)


def load_and_copy_z_scores(
    params,
    donor_res_path,
    new_res_path,
    check_existing=False,
):
    """
    Get the mean and the std for different parameters of a dataset. Works by copying the
    z-scores from another experiment. Meant for transfer learning.

    Parameters
    ----------
    params: List[str]
        Input features to be normalized
    donor_res_path: str
        Path to normalization results of the donor experiment
    new_res_path: str
        Path to store the normalization results of the receiver experiment
    check_existing: bool
        If True, check if the new_res_path exists and return the dict if so.

    Returns
    -------
    dict
        Dictionary containing the min and the max-min for each parameter.
    """
    # If check_existing is True, check if the file exists and return the dict (if so)
    if check_existing and os.path.exists(new_res_path):
        with open(new_res_path, "rb") as ff:
            return pickle.load(ff)

    # Load the donor dict
    with open(donor_res_path, "rb") as ff:
        donor_dict = pickle.load(ff)

    # Check the dict
    assert all(
        kk in donor_dict for kk in params
    ), "Some parameters are missing in the donor dict."

    # Store the dict
    store_res_dir, _ = os.path.split(new_res_path)
    os.makedirs(store_res_dir, exist_ok=True)
    with open(new_res_path, "wb") as ff:
        pickle.dump(donor_dict, ff)

    return donor_dict


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
# STORE_SUMMARY -> True to store the model summary. Not recommended always, but useful
# for debugging
STORE_SUMMARY = False
# MAX_STEPS -> Maximum number of samples (network scenarios) per epoch
MAX_STEPS = 500

# SELECT DONOR EXPERIMENT SELECTION -> make sure the values are the same as those used
# in the donor experiment (train.py script)
donor_ds_name = "data_seg_poisson_on_off_simulated_0_4_100"
donor_experiment_name = "baselines"
model_class = RouteNet_temporal_delay
donor_variant = "500_steps"
donor_target = "avg_delay"
donor_weights = "120-0.0132"
assert donor_weights != "", "Donor weights must be provided"

donor_experiment_path = get_experiment_path(
    donor_experiment_name,
    donor_ds_name,
    model_class.__name__,
    donor_target,
    variant=donor_variant,
)

# FINE TUNING EXPERIMENT CONFIGURATION
new_ds_name = "data_seg_on_off_0_4_100_v2/topo_5_10_2_SP_k_4"
new_experiment_name = "fine_tuning"
new_variant = "all_samples"
# ENCODING OPTIONS: SELECT DECISION PER BLOCK
encoding_option = FINETUNE_OPTIONS.FREEZE
mp_option = window_option = FINETUNE_OPTIONS.FREEZE
readout_option = FINETUNE_OPTIONS.FINETUNE
finetune_options, finetune_options_str = get_layer_options_RouteNet_temporal_delay(
    encoding_option, mp_option, window_option, readout_option
)
new_target = "avg_delay"
mask = f"flow_has_{new_target.split('_')[0]}"

new_experiment_path = get_experiment_path(
    new_experiment_name,
    new_ds_name,
    model_class.__name__,
    new_target,
    finetune_options_str,
    new_variant,
    donor_ds_name,
)

# Dataset selection: ds_name is used to load the dataset. Log transform is applied so
# that the loss is computed over the log-mse. Samples are also shuffled
ds_train = (
    tf.data.Dataset.load(f"data/{new_ds_name}/training", compression="GZIP")
    .prefetch(tf.data.experimental.AUTOTUNE)
    .map(prepare_targets_and_mask([f"flow_{new_target}_per_seg"], mask))
    .map(log_transform)
)
ds_train = ds_train.shuffle(len(ds_train), seed=SEED, reshuffle_each_iteration=True)
# If the number of samples in the dataset is bigger than the MAX_STEPS,the repeat()
# function must be applied.
if ds_repeat_activate := len(ds_train) > MAX_STEPS:
    ds_train = ds_train.repeat()

# Validation data: Same steps as above, but without shuffling and calling .repeat()
ds_val = (
    tf.data.Dataset.load(f"data/{new_ds_name}/validation", compression="GZIP")
    .prefetch(tf.data.experimental.AUTOTUNE)
    .map(prepare_targets_and_mask([f"flow_{new_target}_per_seg"], mask))
    .map(log_transform)
)

# Training hyperparameters:
# Adam optimizer, lr=0.0001, clipnorm=1.0 (have to later update with keras.set_value
# to reset that occurs when loading the donor checkpoint)
lr = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
loss = tf.keras.losses.MeanSquaredError()
model = model_class(
    output_dim=1,
    mask_field=mask,
    log=True,
    # We copy the z-scores from the donor model
    z_scores=load_and_copy_z_scores(
        model_class.z_scores_fields,
        os.path.join("normalization", donor_experiment_path, "z_scores.pkl"),
        os.path.join("normalization", new_experiment_path, "z_scores.pkl"),
        check_existing=True,
    ),
)
# Store in normalization a note with the donor experiment path
with open(
    os.path.join("normalization", new_experiment_path, "donor_experiment_path.txt"), "w"
) as ff:
    ff.write(os.path.join(donor_experiment_path, donor_weights))
# Load donor checkpoint
load_model_with_ckpt(
    model, f"ckpt/{donor_experiment_path}/{donor_weights}", finetune_options
)
# get_positional_denorm_mape returns the denormalized MAPE. NOTE: this function does the
# average over network scenario, while the loss function (and model.evaluate() in the
# evaluation notebook) will use the average over the individual flows.
model.compile(
    optimizer=optimizer,
    loss=loss,
    run_eagerly=RUN_EAGERLY,
    metrics=[get_positional_denorm_mape(0, new_target)],
)
# Set true learning rate
K.set_value(model.optimizer.learning_rate, lr)

ckpt_dir = f"ckpt/{new_experiment_path}"
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
    log_dir=f"tensorboard/{new_experiment_path}", histogram_freq=1
)
reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    factor=0.5,
    patience=10,
    verbose=1,
    cooldown=3,
    mode="min",
    monitor="loss",
)
# Early stop that works when min learning rate is surpassed
early_stop_callback = CustomEarlyStop(min_lr=1e-6)

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

# Store model summary, if requested
if STORE_SUMMARY:
    with open(
        os.path.join("normalization", new_experiment_path, "model_summary.txt"), "w"
    ) as ff:
        model.summary(print_fn=lambda x: ff.write(x + "\n"))
