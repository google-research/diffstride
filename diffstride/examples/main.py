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

r"""Launches the training loop given a configuration.

For instance to train a resnet18 and cifar10, with diffstride and saves the
training results locally to be displayed with tensorboard:

python3 -m diffstride.example.main \
  --gin_config=cifar10 \
  --gin_bindings="train.workdir=/tmp/exp/diffstride/resnet18/"
"""

import os
from typing import Sequence

from absl import app
from absl import flags
from diffstride.examples import train
import gin
import tensorflow as tf

flags.DEFINE_multi_string(
    'gin_config', [], 'List of paths to the config files.')
flags.DEFINE_multi_string(
    'gin_bindings', [], 'Newline separated list of Gin parameter bindings.')
flags.DEFINE_string(
    'workdir', None, 'Sets the directory where to save tfevents.')
flags.DEFINE_integer('seed', 1, 'Used for replication.')
flags.DEFINE_string('configs_folder',
                    'third_party/py/diffstride/examples',
                    'Where to find the gin config files.')
FLAGS = flags.FLAGS


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  tf.random.set_seed(FLAGS.seed)
  gin_files = [os.path.join(FLAGS.configs_folder, x) for x in FLAGS.gin_config]
  gin.parse_config_files_and_bindings(gin_files, FLAGS.gin_bindings)
  train.train(workdir=FLAGS.workdir)


if __name__ == '__main__':
  app.run(main)
