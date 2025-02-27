import jax
import jax.numpy as jnp                # JAX NumPy
from jax import nn as jnn              # JAX nn
from flax import jax_utils
from flax import linen as nn           # The Linen API
from flax.training import train_state  # Useful dataclass to keep train state
import functools
import time
from absl import app
from functools import partial
from absl import flags
import numpy as np                     # Ordinary NumPy
import optax                           # Optimizers
#import tensorflow_datasets as tfds     # TFDS for MNIST
from keras.datasets import mnist
from typing import (Any, List)
#from custom_optimizer import *
import custom_optimizer as custom_optimizer
import shampoo_optax as shampoo_optax


flags.DEFINE_float('beta1', 0.9, help='Beta1')
flags.DEFINE_float('beta2', 0.999, help='Beta2')
flags.DEFINE_float('lr', 0.0001, help='Learning rate')
flags.DEFINE_float('eps', 1e-8, help='eps')
flags.DEFINE_integer('batch_size',
                     1000, help='Batch size.')
flags.DEFINE_integer('model_size_multiplier',
                     1, help='Multiply model size by a constant')
flags.DEFINE_integer('model_depth_multiplier',
                     1, help='Multiply model depth by a constant')
flags.DEFINE_integer('warmup_epochs', 5, help='Warmup epochs')
flags.DEFINE_integer('epochs', 100, help='#Epochs')
flags.DEFINE_integer('t', 20, help='preconditioner computation frequency')
flags.DEFINE_enum('dtype', 'float32', ['float32', 'bfloat16'], help='dtype')
flags.DEFINE_enum('optimizer', 'shampoo', ['sgd', 'momentum', 'nesterov', 'adagrad',
  'rmsprop', 'tds', 'shampoo', 'diag_sonew'], help='optimizer')
FLAGS = flags.FLAGS


class Autoencoder(nn.Module):
  enc_hidden_states: List[int]
  dec_hidden_states: List[int]
  dtype: Any
  param_dtype: Any

  @nn.compact
  def __call__(self, x):
    for i in range(len(self.enc_hidden_states)):
      x = nn.Dense(features = self.enc_hidden_states[i],
                   kernel_init=jnn.initializers.glorot_uniform(),
                   dtype=self.dtype, param_dtype=self.param_dtype)(x)
      if i<len(self.enc_hidden_states)-1:
        x = nn.tanh(x)
    for i in range(len(self.dec_hidden_states)):
      x = nn.Dense(features = self.dec_hidden_states[i],
                   kernel_init=jnn.initializers.glorot_uniform(),
                   dtype=self.dtype, param_dtype=self.param_dtype)(x)
      x = nn.tanh(x)
    x = nn.Dense(features = 784,
                 kernel_init=jnn.initializers.glorot_uniform(),
                 dtype=self.dtype, param_dtype=self.param_dtype)(x)
    return x

  def __hash__(self):
    return id(self)


def get_optimizer(opt, learning_rate):
  print("using optimizer:", opt)
  if opt=="sgd":
    return optax.sgd(learning_rate)
  elif opt=="momentum":
    return optax.sgd(learning_rate=learning_rate, momentum=FLAGS.beta1)
  elif opt=="nesterov":
    return optax.sgd(learning_rate=learning_rate, momentum=FLAGS.beta1, nesterov=True)
  elif opt=="adam":
    return optax.adam(b1=FLAGS.beta1, b2=FLAGS.beta2, eps=FLAGS.eps, learning_rate=learning_rate)
  elif opt=="adagrad":
    return optax.adagrad(learning_rate=learning_rate, eps=FLAGS.eps)
  elif opt=="rmsprop":
    return optax.rmsprop(learning_rate=learning_rate, decay=FLAGS.beta2, momentum=FLAGS.beta1, eps=FLAGS.eps)
  elif opt=="diag_sonew":
    return custom_optimizer.diag_sonew(learning_rate, b1=FLAGS.beta1, b2=FLAGS.beta2, eps=FLAGS.eps, adam_grafting=False)
  elif opt=="tds":
    return custom_optimizer.tds(learning_rate, b1=FLAGS.beta1, b2=FLAGS.beta2, eps=FLAGS.eps, transpose=True, adam_grafting=False)
  elif opt=="shampoo":
    print("FLAGS.t:", FLAGS.t)
    print("FLAGS.eps:", FLAGS.eps)
    return shampoo_optax.distributed_shampoo(
      learning_rate=learning_rate,
      block_size=2048,
      beta1=FLAGS.beta1,
      beta2=FLAGS.beta2,
      diagonal_epsilon=1e-12,
      matrix_epsilon=FLAGS.eps,
      weight_decay=0.0,
      start_preconditioning_step=25,
      preconditioning_compute_steps=FLAGS.t,
      statistics_compute_steps=1,
      best_effort_shape_interpretation=True,
      graft_type=4,
      nesterov=False,
      best_effort_memory_usage_reduction=False,
      inverse_failure_threshold=0.1,
      moving_average_for_momentum=True,
      skip_preconditioning_dim_size_gt=4096,
      clip_by_scaled_gradient_norm=None,
      batch_axis_name='batch')
  else:
      raise NotImplementedError

def create_train_state(params, model, opt, learning_rate):
  """Creates initial `TrainState`."""
  tx = get_optimizer(opt, learning_rate)
  return train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=tx)

# Training epoch
def train_step(state, x, model):
  def loss_fn(params):
    logits = model.apply(params, x)
    loss = optax.sigmoid_binary_cross_entropy(logits, x).mean(0).sum()
    return loss
  grad_fn = jax.value_and_grad(loss_fn)
  loss, grads = grad_fn(state.params)
  grads = jax.lax.pmean(grads, axis_name='batch')
  state = state.apply_gradients(grads=grads)
  return state, loss

#Evaluation. This is what gets optimized by hparam search
@partial(jax.jit, static_argnums=0)
def eval_step(model, state, x):
  logits = model.apply(state.params, x)
  loss = optax.sigmoid_binary_cross_entropy(logits, x)
  return loss.astype(jnp.float32).mean(0).sum()

#Training
def train(state, model, train_inputs, rng, lrVec):
  time_array = []
  if FLAGS.batch_size %  jax.device_count() != 0:
    raise ValueError('Batch size must be divisible by the number of devices')

  train_ds_size = len(train_inputs)
  steps_per_epoch = train_ds_size // FLAGS.batch_size

  #Make copies of state
  state = jax_utils.replicate(state)

  #pmap train, eval, and update_lr functions
  #p_update_lr = jax.pmap(update_lr, in_axes=(0,None))
  p_train_step = jax.pmap(functools.partial(train_step, model=model), axis_name='batch')
  p_eval_step = jax.pmap(functools.partial(eval_step, model=model))

  for epoch in range(FLAGS.epochs):
    start_time = time.time()
    print("epoch:", epoch,"and lr going to be used:", lrVec[epoch])
    rng, key = jax.random.split(rng)
    perms = jax.random.permutation(key, train_ds_size)
    perms = perms[:steps_per_epoch * FLAGS.batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, FLAGS.batch_size))
    for perm in perms:
      train_x = train_inputs[perm].reshape((jax.device_count(), -1, 784))
      state, loss = p_train_step(state=state, x=train_x)
      print("loss:", loss, loss.dtype)
    train_loss_val = p_eval_step(state=state, x=train_inputs.reshape((jax.device_count(),-1,784))).mean()
    print("train_loss_val:", train_loss_val, train_loss_val.dtype)
    time_array.append(time.time()-start_time)
    print("TIME TAKEN FOR THIS EPOCH:", time_array[-1])
  state = jax_utils.unreplicate(state)
  return state, time_array


def main(argv):
  #Get random keys
  rng = jax.random.PRNGKey(0)
  rng, key1 = jax.random.split(rng)
  rng, key2 = jax.random.split(rng)
  rng, key3 = jax.random.split(rng)

  #Get dtype
  if FLAGS.dtype=="float32":
    dtype = jnp.float32
  elif FLAGS.dtype=="bfloat16":
    dtype = jnp.bfloat16
  else:
      raise NotImplementedError

  #Generate data
  (train_inputs, _), (test_inputs, test_labels) = mnist.load_data()
  train_inputs = train_inputs.astype(jnp.float32)
  test_inputs = test_inputs.astype(jnp.float32)

  # Rescale input images to [0, 1]
  train_inputs = jnp.reshape(train_inputs, [-1, 784]) / 255.0
  test_inputs = jnp.reshape(test_inputs, [-1, 784]) / 255.0

  train_inputs = train_inputs.astype(dtype)
  test_inputs = test_inputs.astype(dtype)

  num_train_examples = train_inputs.shape[0]
  num_test_examples = test_inputs.shape[0]
  print('MNIST dataset:')
  print('Num train examples: ' + str(num_train_examples))
  print('Num test examples: ' + str(num_test_examples))

  batch_size = FLAGS.batch_size

  encoder_sizes = [1000] +  [500] * FLAGS.model_depth_multiplier + [250, 30]
  decoder_sizes = [250] +  [500] * FLAGS.model_depth_multiplier + [1000]

  encoder_sizes = [FLAGS.model_size_multiplier * e for e in encoder_sizes]
  decoder_sizes = [FLAGS.model_size_multiplier * e for e in decoder_sizes]

  input_image_batch = np.random.normal(size=(batch_size, 784))
  input_image_batch = input_image_batch.astype(dtype)

  #Set learning rate schedule array
  num_epochs = FLAGS.epochs
  warmup_epochs = FLAGS.warmup_epochs
  lr = FLAGS.lr
  lrVec = np.concatenate([np.linspace(0,lr,warmup_epochs),
                          np.linspace(lr, 0, num_epochs-warmup_epochs+2)[1:-1]],
                         axis=0)

  lrVec = jnp.array(lrVec).astype(dtype)

  model = Autoencoder(encoder_sizes, decoder_sizes, dtype=dtype,
                      param_dtype=dtype)
  params = model.init(key3, input_image_batch)
  def autoencoder_schedule(lrVec):
    def schedule(count):
      bucket = count//60
      return lrVec[bucket]
    return schedule
  state = create_train_state(params, model, FLAGS.optimizer, autoencoder_schedule(lrVec))

  state, time_array = train(state=state, model=model, train_inputs=train_inputs, rng=rng, lrVec=lrVec)
  return time_array

#start = time.time()
#time_array = main()
#print("TOTAL TIME TAKEN:", time.time()-start)
#print("TIME ARRAY:", time_array)


if __name__ == '__main__':
  start = time.time()
  app.run(main)
  print("TOTAL TIME TAKEN:", time.time()-start)
