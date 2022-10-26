import time
import argparse

import jax
import jax.numpy as jnp
from flax import jax_utils
from flax.training import checkpoints
import tensorflow as tf
import tensorflow_datasets as tfds
import wandb

from utils import init_model_state, get_first_device, ProgressMeter
from model import ResNet
CKPT_DIR = '/ckpts'

def main():
    global model
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    train_loader = load_dataset(config, train=True)

    batch = next(train_loader)
    batch = get_first_device(batch)

    model = ResNet()
    state = init_model_state(init_rng, model, batch, config)
    state = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, target=state)

    train(model, state, train_loader)


def train_step(batch, state):
    def loss_fn(params, batch):
        out = state.apply_fn(
            {'params': params},
            image=batch['image'],
            label=batch['label']
        )
        return out['loss'], out

    aux, grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(state.params, batch)
    out = aux[1]
    
    grads = jax.lax.pmean(grads, axis_name='device')
    new_state = state.apply_gradients(
        grads=grads,
    )

    return new_state, out


def train(model, state, train_loader):
    progress = ProgressMeter(
        config.total_steps,
        ['time', 'data'] + model.metrics
    )

    p_train_step = jax.pmap(train_step, axis_name='device')
    start_itr = state.step
    state = jax_utils.replicate(state)

    end = time.time()
    for itr in range(start_itr, config.total_steps):
        batch = next(train_loader)
        batch_size = batch['image'].shape[1]
        progress.update(data=time.time() - end)

        state, return_dict = p_train_step(batch=batch, state=state)

        metrics = {k: return_dict[k].mean() for k in model.metrics}
        metrics = {k: v.astype(jnp.float32) for k, v in metrics.items()}
        progress.update(n=batch_size, **{k: v for k, v in metrics.items()})
        progress.update(time=time.time() - end)
        end = time.time()
        if itr % config.log_interval == 0:
            progress.display(itr)
            if jax.process_index() == 0:
                wandb.log(metrics)
                checkpoints.save_checkpoint(ckpt_dir=CKPT_DIR, target=jax_utils.unreplicate(state), step=itr)


def load_dataset(config, train):
    split = 'train' if train else 'test'
    split = tfds.split_for_jax_process(
        split, 
        process_index=jax.process_index(), 
        process_count=jax.process_count()
    )
    dataset = tfds.load('cifar10', split=split)

    def process(features):
        image = tf.cast(features['image'], tf.float32)
        image = 2 * (image / 255.) - 1
        label = features['label']
        return dict(image=image, label=label)

    dataset = dataset.map(process)
    dataset = dataset.cache()

    batch_size = config.batch_size // jax.process_count()
    dataset = dataset.shuffle(batch_size * 64)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    def prepare_tf_data(xs):
        def _prepare(x):
            x = x._numpy()
            return x.reshape(jax.local_device_count(), -1, *x.shape[1:])
        xs = jax.tree_map(_prepare, xs)
        return xs
    
    iterator = map(prepare_tf_data, dataset)
    iterator = jax_utils.prefetch_to_device(iterator, 2)
    
    return iterator


if __name__ == '__main__':
    print(f'JAX process: {jax.process_index()} / {jax.process_count()}')
    print(f'JAX total devices: {jax.device_count()}')
    print(f'JAX local devices: {jax.local_device_count()}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--total_steps', type=int, default=1000000)
    parser.add_argument('--log_interval', type=int, default=100)
    config = parser.parse_args()

    wandb.login()
    wandb.init(project="tpu_cifar", id='spot-run-'+config.run_id, resume=True)

    is_master_process = jax.process_index() == 0
    main()
