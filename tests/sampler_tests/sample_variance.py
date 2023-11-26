import torch
from argparse import ArgumentParser
from sampler_dutifulpond10 import setup_sampler
from jaxtyping import Float
from torch import Tensor
import pickle

def single_sample(sampler, return_every: int = 10):
    beta = sampler.model.fwd_diff.betas[-1].view(-1,1,1,1)
    x = sampler.model.init_noise(2) * torch.sqrt(beta)
    intermediates = {}
    for i in reversed(range(1, sampler.model.fwd_diff.timesteps)):
        t = i * torch.ones((2), dtype=torch.long, device=torch.device("cuda"))
        x = sampler.model.denoise_singlestep(x, t)
        if (i%return_every)==0:
            intermediates[i] = x
    return x[0], {t:elem[0] for (t,elem) in intermediates.items()}

def sample_from_t(start_sample: Tensor, t: int, num_samples: int):
    x = start_sample.unsqueeze(0).repeat(num_samples, 1, 1, 1)
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
    sample, interm = single_sample(sampler, return_every=10)
    outs = {}
    for (t,sample) in interm.items():
        outs[t] = sample_from_t(sample, t, args.num_samples)
    with open('/itet-stor/peerli/net_scratch/sample_variance.obj', 'wb') as f:
        pickle.dump(outs, f, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('sample_variances.obj', 'rb') as f:
    #     outs = pickle.load(f)