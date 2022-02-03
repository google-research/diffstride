# coding=utf-8
# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training library."""

import os
from typing import Optional, Type
from absl import logging

import gin
import tensorflow as tf


@gin.configurable
def train(load_data_fn,
          model_cls: Type[tf.keras.Model],
          optimizer_cls: Type[tf.keras.optimizers.Optimizer],
          num_epochs: int = 200,
          workdir: Optional[str] = '/tmp/diffstride/') -> tf.keras.Model:
  """Runs the training using keras .fit way."""
  train_ds, test_ds, num_classes = load_data_fn()

  strategy = tf.distribute.MirroredStrategy()
  logging.info('Number of devices: %d', strategy.num_replicas_in_sync)

  # Decides to run channels first on GPU and channels last otherwise.
  with strategy.scope():
    model = model_cls(
        num_output_classes=num_classes,
        channels_first=bool(tf.config.list_physical_devices('GPU')))
    model.compile(optimizer=optimizer_cls(),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])

  callbacks = []
  if workdir is not None:
    callbacks.extend([
        tf.keras.callbacks.TensorBoard(
            log_dir=workdir, write_steps_per_second=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(workdir, 'ckpts')),
        tf.keras.callbacks.experimental.BackupAndRestore(
            backup_dir=os.path.join(workdir, 'backup'))
    ])
  model.fit(train_ds, validation_data=test_ds,
            epochs=num_epochs, callbacks=callbacks)
  return model
