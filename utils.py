from collections import OrderedDict
import optax
import jax
from flax.training.train_state import TrainState


def get_first_device(x):
    x = jax.tree_util.tree_map(lambda a: a[0], x)
    return jax.device_get(x)

 
def init_model_state(rng, model, batch, config):
    params = model.init(
        rngs={'params': rng},
        image=batch['image'],
        label=batch['label']
    ).unfreeze()['params']

    learning_rate_fn = optax.warmup_cosine_decay_schedule(
        init_value=0.,
        peak_value=config.lr,
        warmup_steps=config.warmup_steps,
        decay_steps=config.total_steps- config.warmup_steps
    )

    tx = optax.adam(learning_rate=learning_rate_fn)

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, total_iters, meter_names, prefix=""):
        self.iter_fmtstr = self._get_iter_fmtstr(total_iters)
        self.meters = OrderedDict({mn: AverageMeter(mn, ':6.3f')
                                   for mn in meter_names})
        self.prefix = prefix

    def update(self, n=1, **kwargs):
        for k, v in kwargs.items():
            self.meters[k].update(v, n=n)

    def display(self, iteration):
        entries = [self.prefix + self.iter_fmtstr.format(iteration)]
        entries += [str(meter) for meter in self.meters.values()]
        print('\t'.join(entries))

    def _get_iter_fmtstr(self, total_iters):
        num_digits = len(str(total_iters // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(total_iters) + ']'
 
