# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A mock-up showing a ResNet50 network with training on synthetic data.

This file uses the stax neural network definition library and the optimizers
optimization library.
"""

import numpy.random as npr

import jax.numpy as jnp
from jax import jit, grad, random
from jax.example_libraries import optimizers
from jax.example_libraries import stax
from jax.example_libraries.stax import (AvgPool, BatchNorm, Conv, Dense,
                                        FanInSum, FanOut, Flatten, GeneralConv,
                                        Identity, MaxPool, Relu, LogSoftmax)
import tensorflow_datasets as tfds
import time
from geesibling.adapters.jax import parallelize, device_config, DeviceType
from geesibling.adapters.jax import api
from geesibling.adapters.jax.pipeline.devicecontext import init_global_cluster,get_global_virtual_physical_mesh,get_sliced_virtual_submeshes,shutdown_global_cluster
import ray.util.collective as col
from ray.util.collective.collective_group.nccl_collective_group import NCCLGroup
# ResNet blocks compose other layers

def ConvBlock(kernel_size, filters, strides=(2, 2)):
  ks = kernel_size
  filters1, filters2, filters3 = filters
  Main = stax.serial(
      Conv(filters1, (1, 1), strides), BatchNorm(), Relu,
      Conv(filters2, (ks, ks), padding='SAME'), BatchNorm(), Relu,
      Conv(filters3, (1, 1)), BatchNorm())
  Shortcut = stax.serial(Conv(filters3, (1, 1), strides), BatchNorm())
  return stax.serial(FanOut(2), stax.parallel(Main, Shortcut), FanInSum, Relu)


def IdentityBlock(kernel_size, filters):
  ks = kernel_size
  filters1, filters2 = filters
  def make_main(input_shape):
    # the number of output channels depends on the number of input channels
    return stax.serial(
        Conv(filters1, (1, 1)), BatchNorm(), Relu,
        Conv(filters2, (ks, ks), padding='SAME'), BatchNorm(), Relu,
        Conv(input_shape[3], (1, 1)), BatchNorm())
  Main = stax.shape_dependent(make_main)
  return stax.serial(FanOut(2), stax.parallel(Main, Identity), FanInSum, Relu)


# ResNet architectures compose layers and ResNet blocks

def ResNet50(num_classes):
  return stax.serial(
      GeneralConv(('HWCN', 'OIHW', 'NHWC'), 64, (7, 7), (2, 2), 'SAME'),
      BatchNorm(), Relu, MaxPool((3, 3), strides=(2, 2)),
      ConvBlock(3, [64, 64, 256], strides=(1, 1)),
      IdentityBlock(3, [64, 64]),
      IdentityBlock(3, [64, 64]),
      ConvBlock(3, [128, 128, 512]),
      IdentityBlock(3, [128, 128]),
      IdentityBlock(3, [128, 128]),
      IdentityBlock(3, [128, 128]),
      ConvBlock(3, [256, 256, 1024]),
      IdentityBlock(3, [256, 256]),
      IdentityBlock(3, [256, 256]),
      IdentityBlock(3, [256, 256]),
      IdentityBlock(3, [256, 256]),
      IdentityBlock(3, [256, 256]),
      ConvBlock(3, [512, 512, 2048]),
      IdentityBlock(3, [512, 512]),
      IdentityBlock(3, [512, 512]),
      AvgPool((7, 7)), Flatten, Dense(num_classes), LogSoftmax)


if __name__ == "__main__":
  rng_key = random.PRNGKey(0)

  batch_size = 8
  num_classes = 1001
  input_shape = (224, 224, 3, batch_size)
  step_size = 0.1
  num_steps = 5

  init_fun, predict_fun = ResNet50(num_classes)
  _, init_params = init_fun(rng_key, input_shape)

  def loss(params, batch):
    inputs, targets = batch
    logits = predict_fun(params, inputs)
    return -jnp.sum(logits * targets)

  def accuracy(params, batch):
    inputs, targets = batch
    target_class = jnp.argmax(targets, axis=-1)
    predicted_class = jnp.argmax(predict_fun(params, inputs), axis=-1)
    return jnp.mean(predicted_class == target_class)

  def synth_batches():
    rng = npr.RandomState(0)
    while True:
      images = rng.rand(*input_shape).astype('float32')
      labels = rng.randint(num_classes, size=(batch_size, 1))
      onehot_labels = labels == jnp.arange(num_classes)
      yield images, onehot_labels

  opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=0.9)
  batches = synth_batches()
  
  # Training loop 
  init_global_cluster("ray")
  @parallelize(
        devices=device_config(
            {
                "gpu:0": {
                    "type": DeviceType.gpu,
                    "memory": 3 * 1024 * 1024* 1024 ,
                    "free_memory":3 * 1024 * 1024* 1024 ,
                    "execute_time": 0,
                },
            }
        ),
        policy="sgp",
        method="PipeshardParallel"
    )
  def update(i, opt_state, batch):
    params = get_params(opt_state)
    return opt_update(i, api.grad(loss)(params, batch), opt_state)

  opt_state = opt_init(init_params)
  all_time = 0
  for i in range(num_steps):
    start_time = time.time()
    opt_state = update(i, opt_state, next(batches))
    dcu_time = time.time() - start_time
    print(f"Epoch {i+1} used {dcu_time:0.2f} sec")
    all_time += dcu_time
  trained_params = get_params(opt_state)
  print(f"One DCU: The average time each epoch used is {all_time/num_steps:0.2f} sec")
  

