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
import tensorflow as tf

class RouteNet_temporal_delay(tf.keras.Model):
    """
    RouteNet_temporal_delay model: a ST-GNN model for flow perfomance prediction. Based
    on RouteNet-Fermi, but with the following changes:
    - Added temporal dimension 
    - Removed distribution paramters from input: due to it being windowed
    - Added dependency between windows: self.queue_window_update is used to model
    the dependency of queues between windows

    Note we adapted the DELAY model. Readout structure may be different when predicting
    other perfomance metric. Please, refer to the original paper RouteNet-Fermi model
    for more details: https://github.com/BNN-UPC/RouteNet-Fermi

    Parameters
    ----------
    z_scores : Dict[str, Tuple[float, float]]
        Dictionary with the z-scores for the input fields, according to the
        self.z_scores_fields set
    log : bool, optional
        If true, the model returns the log of the output, by default False
    iterations : int, optional
        Number of iterations in the messaging passing phase, by default 8
    flow_state_dim : int, optional
        Dimension of flow state, by default 32
    link_state_dim : int, optional
        Dimension of link state, by default 32
    queue_state_dim : int, optional
        Dimension of queue state, by default 32
    output_dim : int, optional
        Output dimensions, equal to the number of perfomance metrics to predict,
        by default 1
    mask_field : str, optional
        Name of field that masks window outputs, by default None
    inference_mode : bool, optional
        If true, negative values will be rectified using a RELU. This does NOT work well
        during training, so it is recommended to disable it then. By default False

    Attributes
    ----------
    name : str
        Name of the model
    z_scores_fields : set
        Set of input fields that are z-scored
    """

    z_scores_fields = {
        "flow_traffic_per_seg",
        "flow_packets_per_seg",
    }

    def __init__(
        self,
        z_scores,
        log=False,
        iterations=8,
        flow_state_dim=32,
        link_state_dim=32,
        queue_state_dim=32,
        output_dim=1,
        mask_field=None,
        inference_mode=False,
    ):
        super(RouteNet_temporal_delay, self).__init__()

        # Configuration dictionary. It contains the needed Hyperparameters for the model.
        # All the Hyperparameters can be found in the config.ini file

        self.max_buffer_types = 2
        self.iterations = iterations
        self.flow_state_dim = flow_state_dim
        self.link_state_dim = link_state_dim
        self.queue_state_dim = queue_state_dim
        self.output_dim = output_dim

        assert mask_field is not None, "mask_field must be specified"
        self.mask_field = mask_field

        self.set_z_scores(z_scores)
        self.log = log
        self.inference_mode = inference_mode

        # GRU Cells used in the Message Passing step
        self.flow_update = tf.keras.layers.GRUCell(
            self.flow_state_dim, name="FlowUpdate"
        )
        self.link_update = tf.keras.layers.GRUCell(
            self.link_state_dim, name="LinkUpdate"
        )
        self.queue_update = tf.keras.layers.GRUCell(
            self.queue_state_dim, name="QueueUpdate"
        )

        # GRU Cells used to track the dependencies between windows
        self.queue_window_update = tf.keras.layers.GRUCell(
            self.queue_state_dim, name="QueueWindowUpdate"
        )

        # Embedding functions
        self.flow_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(None, 3)),
                tf.keras.layers.Dense(
                    self.flow_state_dim, activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(
                    self.flow_state_dim, activation=tf.keras.activations.relu
                ),
            ],
            name="FlowEmbedding",
        )
        self.queue_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=self.max_buffer_types),
                tf.keras.layers.Dense(
                    self.queue_state_dim, activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(
                    self.queue_state_dim, activation=tf.keras.activations.relu
                ),
            ],
            name="QueueEmbedding",
        )
        self.link_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(None, 1)),
                tf.keras.layers.Dense(
                    self.link_state_dim, activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(
                    self.link_state_dim, activation=tf.keras.activations.relu
                ),
            ],
            name="LinkEmbedding",
        )

        self.readout_path = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(None, self.flow_state_dim)),
                tf.keras.layers.Dense(
                    int(self.link_state_dim / 2), activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(
                    int(self.flow_state_dim / 2), activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(self.output_dim),
            ],
            name="FlowReadout",
        )

    def set_z_scores(self, z_scores):
        assert (
            type(z_scores) == dict
            and all(kk in z_scores for kk in self.z_scores_fields)
            and all(len(val) == 2 for val in z_scores.values())
        ), "overriden z_scores dict is not valid!"
        self.z_scores = z_scores

    @tf.function
    def call(self, inputs):
        # Initialize result matrix
        total_results = tf.zeros((0, self.output_dim))

        seg_num = inputs["seg_num"]
        link_to_path = inputs["link_to_path"]
        path_to_link = path_to_queue = inputs["path_to_link"]

        queue_to_link = inputs["queue_to_link"]
        link_to_path = queue_to_path = inputs["link_to_path"]

        # Initial embeddings
        traffic = inputs["flow_traffic_per_seg"]
        pkt_rate = inputs["flow_packets_per_seg"]
        # Doing the transpose will optimize memory access later on
        avg_pkt_size = tf.transpose(
            inputs["flow_avg_packet_size_per_seg"], perm=[1, 0, 2]
        )
        length = tf.squeeze(inputs["flow_length"], 1)
        flow_has_traffic = inputs["flow_has_traffic"]
        # Doing the transpose will optimize memory access later on
        output_mask = tf.transpose(inputs[self.mask_field], perm=[1, 0])
        # We apply the transpose so the first dimension are the segments, the second the
        # flows. This will optimize memory access later on
        initial_flow_state = tf.transpose(
            self.flow_embedding(
                tf.concat(
                    [
                        (traffic - self.z_scores["flow_traffic_per_seg"][0])
                        / self.z_scores["flow_traffic_per_seg"][1],
                        (pkt_rate - self.z_scores["flow_packets_per_seg"][0])
                        / self.z_scores["flow_packets_per_seg"][1],
                        tf.expand_dims(tf.cast(flow_has_traffic, tf.float32), 2),
                    ],
                    axis=2,
                ),
            ),
            perm=[1, 0, 2],
        )

        # Calculate load per link per window, including packet size correction due to
        # l1 and l2 headers size
        capacity = inputs["link_capacity"] * 1e9
        expanded_capacity = tf.tile(tf.expand_dims(capacity, 1), [1, seg_num, 1])
        pkt_size_correction = inputs["link_pkt_header_size"]
        pkt_size_correction = tf.tile(
            tf.expand_dims(pkt_size_correction, 1), [1, seg_num, 1]
        )
        flow_gather_traffic = tf.gather(traffic, path_to_link[:, :, 0])
        flow_traffic = tf.math.reduce_sum(flow_gather_traffic, axis=1)
        flow_gather_pkt_rate = tf.gather(pkt_rate, path_to_link[:, :, 0])
        flow_pkt_rate = tf.math.reduce_sum(flow_gather_pkt_rate, axis=1)
        load = (flow_traffic + flow_pkt_rate * pkt_size_correction) / expanded_capacity
        # We apply the transpose so the first dimension are the segments, the second the
        # links
        initial_link_state = tf.transpose(self.link_embedding(load), [1, 0, 2])

        # Queue_state and node states are related to memory buffers, these are the
        # states that are kept between windows
        buffer_type = inputs["buffer_type"]
        initial_queue_state = previous_queue_state = queue_state = self.queue_embedding(
            tf.squeeze(tf.one_hot(buffer_type, self.max_buffer_types), 1)
        )

        # Variables for tf.autograd
        flow_state_sequence = tf.RaggedTensor.from_row_lengths(
            tf.zeros((tf.reduce_sum(length), self.flow_state_dim)), length
        ).with_row_splits_dtype(tf.int64)

        for curr_seg in range(inputs["seg_num"]):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (total_results, tf.TensorShape([None, self.output_dim])),
                    (
                        flow_state_sequence,
                        tf.TensorShape([None, None, self.flow_state_dim]),
                    ),
                ],
            )

            # Initialize segment states for flows and links
            flow_state = initial_flow_state[curr_seg]
            link_state = initial_link_state[curr_seg]
            previous_queue_state = queue_state

            # Iterate t times doing the message passing
            for it in range(self.iterations):
                ###################
                #  LINK AND QUEUE #
                #     TO PATH     #
                ###################
                queue_gather = tf.gather(queue_state, queue_to_path)
                link_gather = tf.gather(link_state, link_to_path, name="LinkToFlow")
                flow_update_rnn = tf.keras.layers.RNN(
                    self.flow_update, return_sequences=True, return_state=True
                )
                previous_flow_state = flow_state

                # flow_state -> state of flow after processing sequence
                # flow_state_sequence -> sequence of intermediate states of the apth
                # when elements within the sequence are processed
                flow_state_sequence, flow_state = flow_update_rnn(
                    tf.concat([queue_gather, link_gather], axis=2),
                    initial_state=flow_state,
                )
                # We select the element in flow_state_sequence so that it corresponds to
                # the state before the link was considered
                flow_state_sequence = tf.concat(
                    [tf.expand_dims(previous_flow_state, 1), flow_state_sequence],
                    axis=1,
                )

                ###################
                #  PATH TO QUEUE  #
                ###################
                flow_gather = tf.gather_nd(flow_state_sequence, path_to_queue)
                flow_sum = tf.math.reduce_sum(flow_gather, axis=1)
                queue_state, _ = self.queue_update(flow_sum, [queue_state])

                ###################
                #  QUEUE TO LINK  #
                ###################
                queue_gather = tf.gather(queue_state, queue_to_link)

                link_gru_rnn = tf.keras.layers.RNN(
                    self.link_update, return_sequences=False
                )
                link_state = link_gru_rnn(queue_gather, initial_state=link_state)


            ###################
            # MESSAGE PASSING #
            #       END       #
            ###################
            # Queue and node state update
            queue_state, _ = self.queue_window_update(
                tf.concat([initial_queue_state, previous_queue_state], axis=1),
                [queue_state],
            )

            ###################
            #     READOUT     #
            ###################
            # Delay and jitter is calcuated from the final state of each link in
            # path, then normalized by capacity
            capacity_gather = tf.gather(capacity, link_to_path)
            input_tensor = flow_state_sequence[:, 1:].to_tensor()
            length = tf.ensure_shape(length, [None])

            links_gather = self.readout_path(input_tensor)
            links_gather = tf.RaggedTensor.from_tensor(links_gather, lengths=length)
            # If jitter, this is the final result
            # If delay, we only computed the queueing delay
            result = tf.math.reduce_sum(links_gather / capacity_gather, axis=1)
            if self.inference_mode:
                result = tf.keras.activations.relu(result)

            # Need to compute and add transmission delay
            trans_delay = avg_pkt_size[curr_seg] * tf.math.reduce_sum(
                1 / capacity_gather, axis=1
            )
            result = result + trans_delay

            total_results = tf.concat(
                [total_results, tf.boolean_mask(result, output_mask[curr_seg])], axis=0
            )

        ##################
        #     WINDOW     #
        # PROCESSING END #
        ##################
        if self.log:
            return tf.math.log(total_results)
        return total_results

