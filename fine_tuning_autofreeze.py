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

import tensorflow as tf
from keras import backend
from models import RouteNet_temporal_delay
import os
from random import seed
import numpy as np
from utils import (
    get_positional_denorm_mape,
    get_experiment_path,
    prepare_targets_and_mask,
    log_transform,
    FINETUNE_OPTIONS,
    load_model_with_ckpt,
    DatasetIteratorWrapper,
    AvgAccumulator,
    load_and_copy_z_scores,
    FINETUNE_OPTIONS,
    load_model_with_ckpt,
)


class AutoFreezeTrainer:
    """
    Class that implements a custom training loop implementing AutoFreeze.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        loss_fn: callable,
        optimizer: tf.keras.optimizers.Optimizer,
        block_to_layers: dict,
        pct: int = 50,
        freeze_order: tuple = ("encoding", "mpa", "readout"),
        additional_metrics: list = [],
        reduce_lr_on_plateau: bool = False,
        reduce_lr_factor: float = 0.5,
        reduce_lr_patience: int = 10,
        reduce_lr_cooldown: int = 3,
        reduce_lr_monitor: str = "loss",
        early_stopping: bool = False,
        early_stopping_min_lr: float = 1e-6,
        save_weights_path: str = None,
        save_weights_monitor: str = "val_loss",
        logs_path: str = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.opt = optimizer
        self.block_to_layers = block_to_layers
        self.previous_layer_grads = None
        self.pct = pct
        self.freeze_order = list(freeze_order)
        self.additional_metrics = additional_metrics

        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        if self.reduce_lr_on_plateau:
            self.reduce_lr_factor = reduce_lr_factor
            self.reduce_lr_patience = reduce_lr_patience
            self.reduce_lr_cooldown = reduce_lr_cooldown
            self.reduce_lr_monitor = reduce_lr_monitor
        else:
            self.reduce_lr_factor = None
            self.reduce_lr_patience = None
            self.reduce_lr_cooldown = None
            self.reduce_lr_monitor = None

        self.early_stopping = early_stopping
        if self.early_stopping:
            self.early_stopping_min_lr = early_stopping_min_lr
        else:
            self.early_stopping_min_lr = None

        self.save_weights_path = save_weights_path
        if save_weights_path is not None:
            os.makedirs(save_weights_path, exist_ok=True)
            self.save_weights_monitor = save_weights_monitor
        else:
            self.save_weights_monitor = None

        if logs_path is not None:
            os.makedirs(logs_path, exist_ok=True)
            self.logs_path = os.path.join(logs_path, "training_log.csv")
        else:
            self.logs_path = None
        self.logs_num_headers = None

    def _vars_for_block(self, b):
        vs = []
        for lyr in self.block_to_layers[b]:
            if lyr.trainable:
                vs.extend(lyr.trainable_variables)
        return vs

    def _all_trainable_vars(self):
        res_list = []
        res_named_list = []
        for name, weights in self._get_sublayers_recursive(self.model):
            res_named_list.append((name, weights))
            if weights is not None:
                res_list.extend(weights)
        return res_list, res_named_list

    def _get_sublayers_recursive(self, layer):
        if hasattr(layer, "layers"):
            res = []
            for sublayer in layer.layers:
                res.extend(self._get_sublayers_recursive(sublayer))
            return res
        return [(layer.name, layer.trainable_variables)]

    def _get_all_layers_names(self):
        return [name for name, var in self._get_sublayers_recursive(self.model)]

    def _update_intervals_and_maybe_freeze(self, grads_by_layer):
        # 1) Extract current gradients from active layers
        current_layers = {}
        for name, weights in self._get_sublayers_recursive(self.model):
            if weights:
                wnorm = tf.linalg.global_norm(weights)
                wnorm = tf.cast(wnorm, tf.float32)
            else:
                wnorm = None
            current_layers[name] = wnorm
        if len(current_layers) == 0:
            return

        # 2) Compare with previous (if there is none (first epoch), just continue)
        # Also skip if not checking gradients this epoch
        if self.previous_layer_grads is None:
            self.previous_layer_grads = current_layers
            return
        differences_layer = {}
        for name, curr_vals in current_layers.items():
            if curr_vals is None:
                continue
            if curr_vals == 0:
                diff = 0.0 if self.previous_layer_grads[name] == 0 else 1.0
            else:
                diff = (
                    tf.abs(curr_vals - self.previous_layer_grads[name])
                    / self.previous_layer_grads[name]
                )
            differences_layer[name] = diff
        self.previous_layer_grads = current_layers

        # 3) Compute threshold (based on all layers, not blocks)
        threshold = np.percentile(list(differences_layer.values()), self.pct)
        # 4) freeze strictly left→right; only consider the *first* active block that qualifies
        active_blocks = [
            bb
            for bb in self.freeze_order
            if any(ll.trainable for ll in self.block_to_layers[bb])
        ]
        for bb in active_blocks:
            # Freeze block if all layers below threshold
            if any(
                differences_layer[lyr.name] > threshold
                for lyr in self.block_to_layers[bb]
            ):
                break
            # Freeze the block
            for lyr in self.block_to_layers[bb]:
                lyr.trainable = False
            print(f"[AutoFreeze] Froze block: {bb}")

    @tf.function
    def _train_step(self, x, y):
        # compute grads on *currently* trainable vars
        with tf.GradientTape() as tape:
            yhat = self.model(x, training=True)
            # ensure per-batch scalar loss; your loss_fn can be with reduction=SUM_OVER_BATCH_SIZE
            loss = self.loss_fn(y, yhat)
        trainable_vars, layers_to_vars = self._all_trainable_vars()
        grads = tape.gradient(loss, trainable_vars)
        # apply optimizer step
        self.opt.apply_gradients(zip(grads, trainable_vars))
        # collect grads by block for norms
        grads_by_layer = {}
        cursor = 0
        for name, ll_vars in layers_to_vars:
            n = len(ll_vars)
            if n:
                grads_by_layer[name] = grads[cursor : cursor + n]
            else:
                grads_by_layer[name] = []
            cursor += n
        # Compute additional metrics before the return
        metrics = [metric(y, yhat) for metric in self.additional_metrics]
        return (
            loss,
            grads_by_layer,
            metrics,
        )

    @tf.function
    def _validation_step(self, x, y):
        yhat = self.model(x, training=False)
        loss = self.loss_fn(y, yhat)
        metrics = [metric(y, yhat) for metric in self.additional_metrics]
        return loss, metrics

    def _set_headers(self, headers):
        with open(self.logs_path, "w") as ff:
            ff.write("Epoch, " + ", ".join(headers) + ", Frozen Blocks\n")
            self.logs_num_headers = len(headers)

    def _log_metrics(self, values, epoch):
        assert self.logs_num_headers > 0, "Headers still not defined"
        frozen_blocks = [
            bb
            for bb in self.freeze_order
            if not any(l.trainable for l in self.block_to_layers[bb])
        ]
        if len(frozen_blocks) == 0:
            frozen_blocks = ["None"]
        with open(self.logs_path, "a") as ff:
            assert (
                len(values) == self.logs_num_headers
            ), "Mismatch in number of logged metrics"
            ff.write(f"{epoch}, " + ", ".join(f"{m:.7}" for m in values) + ", ")
            ff.write(" ".join(frozen_blocks) + "\n")

    def fit(
        self, ds_tr, steps_per_epoch=None, epochs=1, checks_per_epoch=1, ds_val=None
    ):
        ds_tr_iter = DatasetIteratorWrapper(ds_tr, steps_per_epoch)
        max_steps = steps_per_epoch if steps_per_epoch is not None else len(ds_tr)
        autofreeze_intervals = max(1, max_steps // checks_per_epoch)
        self.previous_layer_grads = None
        no_trainable_blocks = False
        best_ckpt_monitor = np.inf
        reduce_lr_epoch_cooldown = 0
        reduce_lr_patience = self.reduce_lr_patience
        best_reduce_lr_monitor = np.inf
        best_reduce_lr_weights = self.model.get_weights()

        # Prepare logs (if applicable)
        if self.logs_path is not None:
            headers = ["loss"] + [mm.__name__ for mm in self.additional_metrics]
            if ds_val is not None:
                headers += ["val_loss"] + [
                    f"val_{mm.__name__}" for mm in self.additional_metrics
                ]
            self._set_headers(headers)

        for epoch in range(1, epochs + 1):
            # Training loop
            tr_loss = AvgAccumulator()
            tr_metrics = [AvgAccumulator() for _ in self.additional_metrics]
            interval_grads_per_layer = {bb: [] for bb in self._get_all_layers_names()}
            print(f"Epoch {epoch}, Step {0}", end="\r")
            for step, (x, y) in enumerate(ds_tr_iter.get_epoch_samples(), start=1):
                loss, grads_by_layer, metrics_vals = self._train_step(x, y)
                # Terminate on NaN
                if tf.math.is_nan(loss):
                    print()
                    raise ValueError("NaN loss encountered, terminating training")
                # Accumulate gradients for interval
                for ll, vals in grads_by_layer.items():
                    interval_grads_per_layer[ll].extend(vals)
                # Update loss and metrics
                tr_loss.update(tf.reduce_mean(loss))
                for metric, value in zip(tr_metrics, metrics_vals):
                    metric.update(tf.reduce_mean(value))
                print(
                    f"Epoch {epoch}, Step {step}/{max_steps} - loss {tr_loss.value:.6}",
                    *{
                        f"{mm_name.__name__}: {mm_value.value:.6}"
                        for mm_name, mm_value in zip(
                            self.additional_metrics, tr_metrics
                        )
                    },
                    end="\r",
                )
                if step % autofreeze_intervals == 0:
                    self._update_intervals_and_maybe_freeze(interval_grads_per_layer)
                    interval_grads_per_layer = {
                        bb: [] for bb in self._get_all_layers_names()
                    }
                    # for bb, layers in self.block_to_layers.items():
                    #     print(
                    #         f"Block {bb}, trainable:", *[ll.trainable for ll in layers]
                    #     )
                # Check if all layers are frozen
                no_trainable_blocks = all(
                    not ll.trainable
                    for bb in self.freeze_order
                    for ll in self.block_to_layers[bb]
                )
                if no_trainable_blocks:
                    break

            # Validation loop
            if ds_val is not None:
                val_loss = AvgAccumulator()
                val_metrics = [AvgAccumulator() for _ in self.additional_metrics]
                for x, y in ds_val:
                    loss, metrics_vals = self._validation_step(x, y)
                    val_loss.update(loss)
                    for metric, value in zip(val_metrics, metrics_vals):
                        metric.update(tf.reduce_mean(value))

            epoch_metrics = [tr_loss.value] + [mm.value for mm in tr_metrics]
            epoch_metrics_dict = {
                mm.__name__: mm_value.value
                for mm, mm_value in zip(self.additional_metrics, tr_metrics)
            }
            epoch_metrics_dict["loss"] = tr_loss.value
            final_print_metrics_name = ["loss"] + [
                mm.__name__ for mm in self.additional_metrics
            ]
            if ds_val is not None:
                epoch_metrics += [val_loss.value] + [mm.value for mm in val_metrics]
                epoch_metrics_dict.update(
                    {
                        f"val_{mm.__name__}": mm_value.value
                        for mm, mm_value in zip(self.additional_metrics, val_metrics)
                    }
                )
                epoch_metrics_dict["val_loss"] = val_loss.value
                final_print_metrics_name += ["val_loss"] + [
                    f"val_{mm.__name__}" for mm in self.additional_metrics
                ]

            print(
                f"Epoch {epoch}, Step {step}/{max_steps} -",
                *[
                    f"{name}, {val:.6}"
                    for name, val in zip(final_print_metrics_name, epoch_metrics)
                ],
                f"lr {backend.get_value(self.opt.lr):.4}",
            )

            # Logging results
            if self.logs_path is not None:
                self._log_metrics(epoch_metrics, epoch)

            # Checkpointing
            if self.save_weights_path is not None:
                if epoch_metrics_dict[self.save_weights_monitor] < best_ckpt_monitor:
                    best_ckpt_monitor = epoch_metrics_dict[self.save_weights_monitor]
                    best_ckpt_name = f"{epoch}-{best_ckpt_monitor:.4f}"
                    ckpt_path = os.path.join(self.save_weights_path, best_ckpt_name)
                    self.model.save_weights(ckpt_path)
                    print(f"[Checkpoint] Weights saved at {ckpt_path}")

            # Early stopping
            # Activate early stopping if all layers are frozen
            if no_trainable_blocks:
                print("[AutoFreeze] All layers frozen, stopping training")
                return
            if self.early_stopping:
                if self.early_stopping_min_lr > self.opt.learning_rate:
                    print(
                        f"[EarlyStopping] Learning rate below threshold, stopping training"
                    )
                    return

            # Reduce LR on plateau
            if self.reduce_lr_on_plateau:
                # Then check if monitored metric improved
                if epoch_metrics_dict[self.reduce_lr_monitor] < best_reduce_lr_monitor:
                    # Yes: annotate new best and epoch, patience reset
                    best_reduce_lr_monitor = epoch_metrics_dict[self.reduce_lr_monitor]
                    best_reduce_lr_weights = self.model.get_weights()
                    reduce_lr_patience = self.reduce_lr_patience
                else:
                    # No: reduce patience
                    reduce_lr_patience -= 1
                # Finally, if patience and cooldown is exhausted, reduce LR
                if reduce_lr_patience <= 0:
                    reduce_lr_patience = self.reduce_lr_cooldown
                    self.model.set_weights(best_reduce_lr_weights)
                    old_lr = backend.get_value(self.opt.lr)
                    backend.set_value(self.opt.lr, old_lr * self.reduce_lr_factor)
                    print(
                        f"[ReduceLROnPlateau] Reduced learning rate to {backend.get_value(self.opt.lr):.6}"
                    )


# Set all seeds
SEED = 1
seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

# RUN EAGERLY -> True for debugging
RUN_EAGERLY = False
tf.config.run_functions_eagerly(RUN_EAGERLY)
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

# SELECT TARGET MODEL
new_ds_name = "data_seg_on_off_0_4_100_v2/topo_5_10_2_SP_k_4"
new_experiment_name = "advanced_fine_tuning/autofreeze"
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

# Prepare model with donor weights
loss_fn = tf.keras.losses.MeanSquaredError()
opt = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
model = model_class(
    output_dim=1,
    mask_field=mask,
    log=True,
    z_scores=load_and_copy_z_scores(
        model_class.z_scores_fields,
        os.path.join("normalization", donor_experiment_path, "z_scores.pkl"),
        os.path.join("normalization", new_experiment_path, "z_scores.pkl"),
        check_existing=True,
    ),
)
model.compile(optimizer=opt, loss=loss_fn, run_eagerly=RUN_EAGERLY)
# Build the model by running a prediction
model.predict(ds_val.take(1), verbose=0)
# Load donor checkpoint
load_model_with_ckpt(
    model,
    f"ckpt/{donor_experiment_path}/{donor_weights}",
    [FINETUNE_OPTIONS.FINETUNE] * len(model.layers),
)
# Set true learning rate
backend.set_value(model.optimizer.learning_rate, 1e-4)
# Store in normalization a note with the donor experiment path
with open(
    os.path.join("normalization", new_experiment_path, "donor_experiment_path.txt"), "w"
) as ff:
    ff.write(os.path.join(donor_experiment_path, donor_weights))


# Define model blocks. NOTE: ensure all trainable layers are included in a block
block_to_layers = {
    "encoding_1": [
        model.flow_embedding.layers[0],
        model.link_embedding.layers[0],
        model.queue_embedding.layers[0],
    ],
    "encoding_2": [
        model.flow_embedding.layers[1],
        model.link_embedding.layers[1],
        model.queue_embedding.layers[1],
    ],
    "mpa": [
        model.flow_update,
        model.link_update,
        model.queue_update,
        model.queue_window_update,
    ],
    "readout_1": [model.readout_path.layers[0]],
    "readout_2": [model.readout_path.layers[1]],
    "readout_3": [model.readout_path.layers[2]],
}
# Train
trainer = AutoFreezeTrainer(
    model,
    loss_fn,
    opt,
    block_to_layers,
    pct=40,
    freeze_order=(
        "encoding_1",
        "encoding_2",
        "mpa",
        "readout_1",
        "readout_2",
        "readout_3",
    ),
    additional_metrics=[get_positional_denorm_mape(0, new_target)],
    reduce_lr_on_plateau=True,
    early_stopping=True,
    save_weights_path=f"ckpt/{new_experiment_path}",
    logs_path=f"tensorboard/{new_experiment_path}",
    save_weights_monitor=f"val_denorm_mape_{new_target}_metric",
)
trainer.fit(
    ds_train,
    steps_per_epoch=min(MAX_STEPS, len(ds_train)) if MAX_STEPS > 0 else None,
    epochs=10000,
    checks_per_epoch=5,
    ds_val=ds_val,
)
# Store model summary, if requested
if STORE_SUMMARY:
    with open(
        os.path.join("normalization", new_experiment_path, "model_summary.txt"), "w"
    ) as ff:
        model.summary(print_fn=lambda x: ff.write(x + "\n"))
