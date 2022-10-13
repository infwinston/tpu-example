from typing import Tuple
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn


class ResBlock(nn.Module):
    channels: int
    stride: int = 1
    expansion: int = 4

    @nn.compact
    def __call__(self, x):
        out = nn.Sequential([
            nn.Conv(self.channels, [1, 1]),
            nn.GroupNorm(),
            nn.relu,
            nn.Conv(self.channels, [3, 3], strides=[self.stride, self.stride]),
            nn.GroupNorm(),
            nn.relu,
            nn.Conv(self.expansion * self.channels, [1, 1]),
            nn.GroupNorm()
        ])(x)

        if self.stride != 1 or x.shape[-1] != out.shape[-1]:
            x = nn.Sequential([
                nn.Conv(self.expansion * self.channels, [1, 1], strides=[self.stride, self.stride]),
                nn.GroupNorm()
            ])(x)
        
        return nn.relu(out + x)


class ResNet(nn.Module):
    num_blocks: Tuple[int] = (3, 4, 6, 3)
    num_classes: int = 10

    @property
    def metrics(self):
        return ['loss', 'acc']

    @nn.compact
    def __call__(self, image, label):
        x = nn.Sequential([
            nn.Conv(64, [3, 3]),
            nn.GroupNorm(),
            nn.relu
        ])(image)

        x = self._make_layer(x, 64, self.num_blocks[0], stride=1)
        x = self._make_layer(x, 128, self.num_blocks[1], stride=2)
        x = self._make_layer(x, 256, self.num_blocks[2], stride=2)
        x = self._make_layer(x, 512, self.num_blocks[3], stride=2)
        x = nn.avg_pool(x, (4, 4)).reshape(x.shape[0], -1)
        logits = nn.Dense(self.num_classes)(x)

        pred = jnp.argmax(logits, axis=-1)
        acc = (pred == label).mean() * 100.

        label = jax.nn.one_hot(label, self.num_classes)
        loss = optax.softmax_cross_entropy(logits, label).mean()

        return dict(loss=loss, acc=acc)
        

    def _make_layer(self, x, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        for stride in strides:
            x = ResBlock(channels, stride)(x)
        return x
