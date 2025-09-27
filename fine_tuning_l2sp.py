"""
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
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from keras import backend as K
from models import RouteNet_temporal_delay_l2sp, RouteNet_temporal_delay

from random import seed
import numpy as np
from utils import (
    CustomEarlyStop,
    get_positional_denorm_mape,
    get_experiment_path,
    prepare_targets_and_mask,
    log_transform,
    load_and_copy_z_scores,
    FINETUNE_OPTIONS,
    load_model_with_ckpt,
)


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

# SELECT DONOR MODEL
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

# SELECT TARGET MODEL
new_ds_name = "data_seg_poisson_0_4_100_v2/topo_5_10_2_SP_k_4"
new_tp_rt_list = None
# ALTERNATIVE: TO BE USED TO JOIN PARTITIONS IN IDEALIZED TESTBED SCENARIO
# new_tp_rt_list, new_ds_name = join_and_filter_topologies(
#     new_ds_name, "topo_5_10_2", exclude=False
# )
new_experiment_name = "advanced_fine_tuning/l2sp"
new_model_class = RouteNet_temporal_delay_l2sp
new_variant = "all_samples"
new_target = "avg_delay"
mask = f"flow_has_{new_target.split('_')[0]}"

new_experiment_path = get_experiment_path(
    new_experiment_name,
    new_ds_name,
    model_class.__name__,
    new_target,
    None,
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
z_scores = load_and_copy_z_scores(
    model_class.z_scores_fields,
    os.path.join("normalization", donor_experiment_path, "z_scores.pkl"),
    os.path.join("normalization", new_experiment_path, "z_scores.pkl"),
    check_existing=True,
)
# Load donor checkpoint
donor_model = model_class(output_dim=1, mask_field=mask, log=True, z_scores=z_scores)
donor_model.compile(optimizer=optimizer, loss=loss, run_eagerly=RUN_EAGERLY)
# To build the model
donor_model.predict(ds_val.take(1), verbose=0)
load_model_with_ckpt(
    donor_model,
    f"ckpt/{donor_experiment_path}/{donor_weights}",
    [FINETUNE_OPTIONS.FREEZE] * len(donor_model.layers),
)
# Prepare new model with L2-SP regularization
model = new_model_class(
    output_dim=1,
    mask_field=mask,
    log=True,
    z_scores=z_scores,
    regularization_weight=0.0001,
    donor_model=donor_model,
)
load_model_with_ckpt(
    model,
    f"ckpt/{donor_experiment_path}/{donor_weights}",
    [FINETUNE_OPTIONS.FINETUNE] * len(model.layers),
)
# Store in normalization a note with the donor experiment path
with open(
    os.path.join("normalization", new_experiment_path, "donor_experiment_path.txt"), "w"
) as ff:
    ff.write(os.path.join(donor_experiment_path, donor_weights))
# get_positional_denorm_mape returns the denormalized MAPE. NOTE: this function does the
# average over network scenario, while the loss function (and model.evaluate() in the
# evaluation notebook) will use the average over the individual flows.
model.compile(
    optimizer=optimizer,
    loss=loss,
    run_eagerly=RUN_EAGERLY,
    metrics=["mse", get_positional_denorm_mape(0, new_target)],
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
