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
from typing import Callable, Optional, Sequence, Tuple, Type

import gin
import tensorflow as tf



@gin.configurable
def lr_schedule(
    epoch: int,
    lr: float,
    divisions: Sequence[Tuple[int, float]] = ((100, 2.0), (200, 2.0))):
  """Return the value of the learning rate at the epoch."""
  factor = dict(divisions).get(epoch, 1.0)
  return lr / factor


@gin.configurable
def train(load_data_fn,
          model_cls: Type[tf.keras.Model],
          optimizer_cls: Type[tf.keras.optimizers.Optimizer],
          num_epochs: int = 200,
          scheduler_fn: Callable[[int, float, ...], float] = lr_schedule,
          workdir: Optional[str] = '/tmp/diffstride/') -> tf.keras.Model:
  """Runs the training using keras .fit way."""
  train_ds, test_ds, info = load_data_fn()
  _, label_key = info.supervised_keys

  # Decides to run channels first on GPU and channels last otherwise.
  model = model_cls(
      num_output_classes=info.features[label_key].num_classes,
      channels_first=bool(tf.config.list_physical_devices('GPU')))
  model.compile(optimizer=optimizer_cls(),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

  callbacks = [tf.keras.callbacks.LearningRateScheduler(scheduler_fn)]
  if workdir is not None:
    callbacks.extend([
        tf.keras.callbacks.TensorBoard(
            log_dir=workdir, write_steps_per_second=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(workdir, 'ckpts'))
    ])
  model.fit(train_ds, validation_data=test_ds,
            epochs=num_epochs, callbacks=callbacks)
  return model
