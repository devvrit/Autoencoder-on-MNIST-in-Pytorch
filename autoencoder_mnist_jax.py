import jax
import jax.numpy as jnp                # JAX NumPy
from jax import nn as jnn              # JAX nn

from flax import linen as nn           # The Linen API
from flax.training import train_state  # Useful dataclass to keep train state

from absl import app
from functools import partial
from absl import flags
import numpy as np                     # Ordinary NumPy
import optax                           # Optimizers
import tensorflow_datasets as tfds     # TFDS for MNIST
from keras.datasets import mnist
from typing import List
from custom_optimizer import *


flags.DEFINE_float('beta1', 0.9, help='Beta1 for Adam')
flags.DEFINE_float('beta2', 0.999, help='Beta2 for Adam')
flags.DEFINE_float('lr', 0.0001, help='Learning rate')
flags.DEFINE_float('eps', 1e-8, help='eps')
flags.DEFINE_integer('batch_size',
                     1000, help='Batch size.')
flags.DEFINE_integer('model_size_multiplier',
                     1, help='Multiply model size by a constant')
flags.DEFINE_integer('model_depth_multiplier',
                     1, help='Multiply model depth by a constant')
flags.DEFINE_integer('warmup_epochs', 5, help='Warmup epochs')
flags.DEFINE_integer('epochs', 10, help='#Epochs')
FLAGS = flags.FLAGS


class Autoencoder(nn.Module):
  enc_hidden_states: List[int]
  dec_hidden_states: List[int]

  @nn.compact
  def __call__(self, x):
    for i in range(len(self.enc_hidden_states)):
      x = nn.Dense(features = self.enc_hidden_states[i],
                   kernel_init=jnn.initializers.glorot_uniform())(x)
      if i<len(self.enc_hidden_states)-1:
        x = nn.tanh(x)
    for i in range(len(self.dec_hidden_states)):
      x = nn.Dense(features = self.dec_hidden_states[i],
                   kernel_init=jnn.initializers.glorot_uniform())(x)
      x = nn.tanh(x)
    x = nn.Dense(features = 784,
                 kernel_init=jnn.initializers.glorot_uniform())(x)
    return x

  def __hash__(self):
    return id(self)


def create_train_state(params, model, learning_rate):
  """Creates initial `TrainState`."""
#   tx = optax.inject_hyperparams(optax.sgd)(learning_rate)
  tx = optax.inject_hyperparams(tds)(learning_rate, b1=FLAGS.beta1, b2=FLAGS.beta2, eps=FLAGS.eps)
  # tx = optax.sgd(learning_rate)
  return train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=tx)

# Training epoch
# @partial(jax.jit, static_argnums=0)
def train_step(model, state, x):
  def loss_fn(params):
    logits = model.apply(params, x)
    loss = optax.sigmoid_binary_cross_entropy(logits, x).mean(0).sum()
    return loss
  grad_fn = jax.value_and_grad(loss_fn)
  loss, grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  return state, loss

# @partial(jax.jit, static_argnums=0)
def eval_step(model, state, x):
  logits = model.apply(state.params, x)
  loss = optax.sigmoid_binary_cross_entropy(logits, x).mean(0).sum()
  return loss

def train_epoch(state, model, train_ds, batch_size, epoch, rng, lrVec, train_loss_val=None):
  train_ds_size = len(train_ds)
  steps_per_epoch = train_ds_size // batch_size
  print("epoch:", epoch,"and lr going to be used:", lrVec[epoch])
  state.opt_state.hyperparams['learning_rate']=lrVec[epoch]

  perms = jax.random.permutation(rng, train_ds_size)
  perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))
  batch_metrics = []
  for perm in perms:
    train_x = train_ds[perm]
    state, loss = train_step(model, state, train_x)
    batch_metrics.append(loss)
  
  batch_metrics_np = jax.device_get(batch_metrics)
  epoch_metrics_np = np.mean(batch_metrics_np)

  print('train epoch: %d, loss: %.4f' % (epoch, epoch_metrics_np))
  #train_loss_val.create_measurement(objective_value=epoch_metrics_np, step=epoch)

  return state


def main(argv):
  #Get random keys
  rng = jax.random.PRNGKey(0)
  rng, key1 = jax.random.split(rng)
  rng, key2 = jax.random.split(rng)
  rng, key3 = jax.random.split(rng)

  #Generate data
  (train_inputs, _), (test_inputs, test_labels) = mnist.load_data()
  train_inputs = train_inputs.astype(jnp.float32)
  test_inputs = test_inputs.astype(jnp.float32)

  # Rescale input images to [0, 1]
  train_inputs = jnp.reshape(train_inputs, [-1, 784]) / 255.0
  test_inputs = jnp.reshape(test_inputs, [-1, 784]) / 255.0

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
  encoder_decoder_sizes = encoder_sizes, decoder_sizes

  input_image_batch = np.random.normal(size=(batch_size,784))

  #Set learning rate schedule array
  num_epochs = FLAGS.epochs
  warmup_epochs = FLAGS.warmup_epochs
  lr = FLAGS.lr
  lrVec = np.concatenate([np.linspace(0,lr,warmup_epochs),
                          np.linspace(lr,0,num_epochs-warmup_epochs+2)[1:-1]],
                         axis=0)

  train_loss_val_=[]
  model = Autoencoder(encoder_sizes, decoder_sizes)
  params = model.init(key3, input_image_batch)
  state = create_train_state(params, model, FLAGS.lr)

  for i in range(num_epochs):
    rng, key = jax.random.split(rng)
    state = train_epoch(state, model, train_inputs, FLAGS.batch_size, i, key, lrVec, train_loss_val=None)
    train_loss_val = eval_step(model, state, train_inputs)
    train_loss_val_.append(train_loss_val)
    print("epoch: " + str(i) +", train_loss_val: " + str(train_loss_val))
    print("")


if __name__ == '__main__':
  app.run(main)
