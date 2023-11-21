import torch
from argparse import ArgumentParser
from sampler_dutifulpond10 import setup_sampler
from jaxtyping import Float
from torch import Tensor
import pickle

def single_sample(sampler, return_every: int = 10):
    samples, intermediates = sampler.sample(2, return_every)
    samples = samples[0]
    intermediates = {t:samples[0] for (t,samples) in intermediates.values()}
    return samples, intermediates

def sample_from_t(start_sample: Tensor, t: int, num_samples: int):
    start_sample = start_sample.unsqueeze(0).repeat(num_samples, 1, 1, 1)
    for i in reversed(range(1, t)):
        t = i * torch.ones((num_samples), dtype=torch.long, device=start_sample.device)
        x = sampler.model.denoise_singlestep(x, t)
    return x

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c","--checkpoint", type=str)
    parser.add_argument("-n", "--num_samples", type=int)
    args = parser.parse_args()

    device = torch.device("cuda")
    sampler = setup_sampler(args.checkpoint).to(device)
    sample, interm = single_sample(sampler)
    outs = {}
    for (t,sample) in interm.values():
        outs[t] = sample_from_t(sample, t, args.num_samples)
    with open('sample_variances.obj', 'wb') as f:
        pickle.dump(outs, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('sample_variances.obj', 'rb') as f:
        outs = pickle.load(f)