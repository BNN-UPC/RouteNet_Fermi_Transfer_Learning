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
from models import RouteNet_temporal_delay_with_encodings
from random import seed
import numpy as np
import os
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


@tf.function
def get_cosine_dissimilarity_matrix(source, target):
    """
    Compute the cosine dissimilarity matrix between two sets of vectors.
    Args:
        source: A tensor of shape (N, D) representing N vectors of dimension D.
        target: A tensor of shape (M, D) representing M vectors of dimension D.
    Returns:
        A tensor of shape (N, M) representing the cosine dissimilarity between each pair of vectors.
    """
    # Normalize the source and target vectors
    source_normalized = tf.nn.l2_normalize(source, axis=0)
    target_normalized = tf.nn.l2_normalize(target, axis=0)

    # Compute the cosine similarity matrix
    cosine_similarity = tf.matmul(
        source_normalized, target_normalized, transpose_b=True
    )

    # Convert cosine similarity to cosine dissimilarity
    cosine_dissimilarity = 1 / 2 * (1 - cosine_similarity)

    return cosine_dissimilarity


def mse_error(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


class GTOTTrainer:
    """
    Class that implements a custom training loop implementing GTOT-Tuning.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        donor_model: tf.keras.Model,
        loss_fn: tf.keras.losses.Loss,
        optimizer: tf.keras.optimizers.Optimizer,
        mwd_entropy_regularization: float = 0.1,
        mwd_threshold: float = 0.1,
        mwd_max_iter: int = 100,
        mwd_weight: float = 0.1,
        mwd_adjacency_field: str = "adjacency_matrix",
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

        self.donor_model = donor_model
        # Freeze donor model weights
        for layer in self.donor_model.layers:
            layer.trainable = False
        self.mwd_entropy_regularization = mwd_entropy_regularization
        self.mwd_threshold = mwd_threshold
        self.mwd_max_iter = mwd_max_iter
        self.mwd_weight = mwd_weight
        self.mwd_adjacency_field = mwd_adjacency_field

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

    @tf.function
    def _mwd_sparse(self, cost_matrix, adj_matrix, num_vertices):
        marginal = tf.ones((num_vertices, 1), dtype=tf.float32)
        marginal = marginal / tf.reduce_sum(marginal)
        marginal_log = tf.math.log(marginal)
        u = tf.zeros((num_vertices, 1), dtype=tf.float32)
        v_t = tf.zeros((1, num_vertices), dtype=tf.float32)
        neg_cost_matrix = -1 * cost_matrix

        for _ in tf.range(self.mwd_max_iter):
            old_u = u
            u_block = tf.tile(u, [1, num_vertices])
            v_block = tf.tile(v_t, [num_vertices, 1])
            u_v_block = u_block + v_block  # (num_vertices, num_vertices)

            c_u_v = tf.math.exp(neg_cost_matrix + u_v_block)
            c_u_v_masked = adj_matrix * c_u_v  # (num_vertices, num_vertices)

            # u_update = tf.sparse.sparse_dense_matmul(
            #     c_u_v_masked, tf.ones((num_vertices, 1))
            # )  # (num_vertices, 1)
            u_update = tf.sparse.reduce_sum(
                c_u_v_masked, axis=1, keepdims=True, output_is_sparse=False
            )  # (num_vertices, 1)
            u_update = tf.ensure_shape(u_update, (None, 1))
            u = (
                self.mwd_entropy_regularization * (marginal_log - tf.math.log(u_update))
                + u
            )  # (num_vertices, 1)

            v_update = tf.sparse.reduce_sum(
                tf.sparse.transpose(c_u_v_masked),
                axis=1,
                keepdims=True,
                output_is_sparse=False,
            )  # (num_vertices, 1)
            v_update = tf.ensure_shape(v_update, (None, 1))
            v_t = (
                tf.transpose(
                    self.mwd_entropy_regularization
                    * (marginal_log - tf.math.log(v_update))
                )
                + v_t
            )  # (1, num_vertices)

            # Check convergence
            if tf.reduce_sum(tf.abs(u - old_u)) < self.mwd_threshold:
                break

        # Final computation of the transport plan
        u_block = tf.tile(u, [1, num_vertices])
        v_block = tf.tile(v_t, [num_vertices, 1])
        u_v_block = u_block + v_block
        p = adj_matrix * tf.math.exp(
            tf.sparse.add(adj_matrix * neg_cost_matrix, u_v_block)
        )
        mwd = tf.sparse.reduce_sum(p * cost_matrix)
        # tf.print("MWD:", mwd)
        return mwd

    @tf.function
    def _mwd_dense(self, cost_matrix, adj_matrix, num_vertices):
        marginal = tf.ones((num_vertices,), dtype=tf.float32)
        marginal = marginal / tf.reduce_sum(marginal)
        marginal_log = tf.math.log(marginal)
        u = tf.zeros((num_vertices, 1), dtype=tf.float32)
        v_t = tf.zeros((1, num_vertices), dtype=tf.float32)
        neg_cost_matrix = -1 * cost_matrix

        for _ in tf.range(self.mwd_max_iter):
            old_u = u
            u_block = tf.tile(u, [1, num_vertices])
            v_block = tf.tile(v_t, [num_vertices, 1])
            u_v_block = u_block + v_block  # (num_vertices, num_vertices)

            c_u_v = tf.math.exp(neg_cost_matrix + u_v_block)
            c_u_v_masked = adj_matrix * c_u_v  # (num_vertices, num_vertices)

            u_update = tf.math.matmul(
                c_u_v_masked, tf.ones((num_vertices, 1))
            )  # (num_vertices, 1)
            u = (
                self.mwd_entropy_regularization * (marginal_log - tf.math.log(u_update))
                + u
            )  # (num_vertices, 1)
            v_update = tf.math.matmul(
                tf.transpose(c_u_v_masked), tf.ones((num_vertices, 1))
            )  # (num_vertices, 1)
            v_t = (
                tf.transpose(
                    self.mwd_entropy_regularization
                    * (marginal_log - tf.math.log(v_update))
                )
                + v_t
            )  # (1, num_vertices)

            # Check convergence
            if tf.reduce_sum(tf.abs(u - old_u)) < self.mwd_threshold:
                break

        # Final computation of the transport plan
        u_block = tf.tile(u, [1, num_vertices])
        v_block = tf.tile(v_t, [num_vertices, 1])
        u_v_block = u_block + v_block
        p = adj_matrix * tf.math.exp((adj_matrix * neg_cost_matrix) + u_v_block)
        mwd = tf.reduce_sum(p * cost_matrix)
        return mwd

    @tf.function
    def _gtot_regularization(self, target_encodings, donor_encodings, sample):
        """
        Follows implementation in:
        https://www.ijcai.org/proceedings/2022/0518.pdf
        https://github.com/youjibiying/GTOT-Tuning
        """
        # 1. Get cost function: cosine dissimilarity matrix:
        cost_matrix = get_cosine_dissimilarity_matrix(target_encodings, donor_encodings)
        # 2. Get elements from sample
        adj_matrix = sample[self.mwd_adjacency_field]
        # Num elements flows + links + queues, times number of segments
        # Queues == links, hence 2 * links
        num_vertices = (
            sample["num_flows"] + (2 * (sample["num_r_links"] + sample["num_s_links"]))
        ) * sample["seg_num"]

        # 3. Compute MWD (use sparse or dense version depending on the adjacency matrix)
        if isinstance(adj_matrix, tf.SparseTensor):
            mwd = self._mwd_sparse(cost_matrix, adj_matrix, num_vertices)
        else:
            mwd = self._mwd_dense(cost_matrix, adj_matrix, num_vertices)
        return mwd

    def _all_trainable_vars(self):
        return [var for var in self.model.trainable_variables]

    @tf.function
    def _train_step(self, x, y):
        # compute grads on *currently* trainable vars
        with tf.GradientTape() as tape:
            yhat, target_encodings = self.model(x, training=True)
            _, donor_encodings = self.donor_model(x, training=False)
            # Compute loss
            loss = self.loss_fn(y, yhat)
            # Compute MWD Regularization loss
            gtot_regularization = self._gtot_regularization(
                target_encodings,
                donor_encodings,
                x,
            )
            loss = loss + (self.mwd_weight * gtot_regularization)
        trainable_vars = self._all_trainable_vars()
        grads = tape.gradient(loss, trainable_vars)
        # apply optimizer step
        self.opt.apply_gradients(zip(grads, trainable_vars))
        # Compute additional metrics before the return
        metrics = [metric(y, yhat) for metric in self.additional_metrics]
        return (loss, metrics)

    @tf.function
    def _validation_step(self, x, y):
        yhat, _ = self.model(x, training=False)
        loss = self.loss_fn(y, yhat)
        metrics = [metric(y, yhat) for metric in self.additional_metrics]
        return loss, metrics

    def _set_headers(self, headers):
        with open(self.logs_path, "w") as ff:
            ff.write("Epoch, " + ", ".join(headers) + "\n")
            self.logs_num_headers = len(headers)

    def _log_metrics(self, values, epoch):
        assert self.logs_num_headers > 0, "Headers still not defined"
        with open(self.logs_path, "a") as ff:
            assert (
                len(values) == self.logs_num_headers
            ), "Mismatch in number of logged metrics"
            ff.write(f"{epoch}, " + ", ".join(f"{m:.7}" for m in values) + "\n")

    def fit(self, ds_tr, steps_per_epoch=None, epochs=1, ds_val=None):
        ds_tr_iter = DatasetIteratorWrapper(ds_tr, steps_per_epoch)
        max_steps = steps_per_epoch if steps_per_epoch is not None else len(ds_tr)
        best_ckpt_monitor = np.inf
        reduce_lr_patience = max(self.reduce_lr_patience, self.reduce_lr_cooldown)
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
            print(f"Epoch {epoch}, Step {0}", end="\r")
            for step, (x, y) in enumerate(ds_tr_iter.get_epoch_samples(), start=1):
                loss, metrics_vals = self._train_step(x, y)
                # Terminate on NaN
                if tf.math.is_nan(loss):
                    print()
                    raise ValueError("NaN loss encountered, terminating training")
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
                    ckpt_path = os.path.join(
                        self.save_weights_path, f"{epoch}-{best_ckpt_monitor:.4f}"
                    )
                    self.model.save_weights(ckpt_path)
                    print(f"[Checkpoint] Weights saved at {ckpt_path}")

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

# SELECT DONOR MODEL
donor_ds_name = "data_seg_poisson_on_off_simulated_0_4_100"
donor_experiment_name = "baselines"
donor_variant = "500_steps"
donor_target = "avg_delay"
donor_weights = "120-0.0132"
assert donor_weights != "", "Donor weights must be provided"

donor_experiment_path = get_experiment_path(
    donor_experiment_name,
    donor_ds_name,
    RouteNet_temporal_delay_with_encodings.name,
    donor_target,
    variant=donor_variant,
)

# SELECT TARGET MODEL
new_ds_name = "data_seg_on_off_0_4_100_v2/topo_5_10_2_SP_k_4"
new_experiment_name = "advanced_fine_tuning/gtot"
new_variant = "all_samples"
new_target = "avg_delay"
mask = f"flow_has_{new_target.split('_')[0]}"

new_experiment_path = get_experiment_path(
    new_experiment_name,
    new_ds_name,
    RouteNet_temporal_delay_with_encodings.__name__,
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
z_scores = load_and_copy_z_scores(
    RouteNet_temporal_delay_with_encodings.z_scores_fields,
    os.path.join("normalization", donor_experiment_path, "z_scores.pkl"),
    os.path.join("normalization", new_experiment_path, "z_scores.pkl"),
    check_existing=True,
)
model = RouteNet_temporal_delay_with_encodings(
    output_dim=1, mask_field=mask, log=True, z_scores=z_scores
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

# Prepare donor model for GTOT regularization
donor_model = RouteNet_temporal_delay_with_encodings(
    output_dim=1, mask_field=mask, log=True, z_scores=z_scores
)
# Parameters not important, weights will be frozen
donor_model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
# Build the model by running a prediction
donor_model.predict(ds_val.take(1), verbose=0)
# Load donor checkpoint
load_model_with_ckpt(
    donor_model,
    f"ckpt/{donor_experiment_path}/{donor_weights}",
    [FINETUNE_OPTIONS.FREEZE] * len(donor_model.layers),
)

# Train
trainer = GTOTTrainer(
    model,
    donor_model,
    loss_fn,
    opt,
    mwd_weight=0.001,
    additional_metrics=[mse_error, get_positional_denorm_mape(0, new_target)],
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
    ds_val=ds_val,
)
# Store model summary, if requested
if STORE_SUMMARY:
    with open(
        os.path.join("normalization", new_experiment_path, "model_summary.txt"), "w"
    ) as ff:
        model.summary(print_fn=lambda x: ff.write(x + "\n"))
