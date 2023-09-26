# Idea Corner
## Public Datasets
- [fastMRI - NYU and Facebook AI Research](https://fastmri.med.nyu.edu/)
## Training Strategies
- [Mixed Precision Training](https://arxiv.org/pdf/1710.03740.pdf)
## Bayesian Perspective on Generative AI
- [Jake Tae - From ELBO to DDPM](https://jaketae.github.io/study/elbo/)
- [Lilian Weng - What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Lilian Weng - Self-Supervised Representation Learning](https://lilianweng.github.io/posts/2019-11-10-self-supervised/)
- [Lilian Weng - Flow-based Deep Generative Models](https://lilianweng.github.io/posts/2018-10-13-flow-models/)
- [Lilian Weng - From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/)
- [Lilian Weng - From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/)
- [Valentin De Bortoli - Generative Modeling](https://vdeborto.github.io/project/generative_modeling/)
- [Yang Song - Scored-Based Generative Modeling](https://yang-song.net/blog/2021/score/)

## Normalization Papers
- [Group Normalization](https://arxiv.org/pdf/1803.08494.pdf)
- [Batch Normalization](https://arxiv.org/pdf/1502.03167.pdf)
- [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)

## U-Net Variants
- [Recurrent Residual Convolutional Neural Network based on U-Net for Medical Image Segmentation](https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf)
- [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/pdf/1804.03999.pdf)
- [Collection of U-Net Structure](https://medium.com/aiguys/attention-u-net-resunet-many-more-65709b90ac8b)

## Diffusion Models with Transformers
- [Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf)
- [Scalable Diffusion Models with Transformers](https://arxiv.org/pdf/2212.09748.pdf)

## Relative Positional Encodings
- [Self-Attention with Relative Position Representations](https://arxiv.org/pdf/1803.02155.pdf)

## Sophisticated VAE/DDPMs
- [Vector Quantized Diffusion Model for Text-to-Image Synthesis](https://arxiv.org/pdf/2111.14822.pdf)
- [Stable Diffusion](https://arxiv.org/pdf/2112.10752.pdf)

## BatchNorm vs. LayerNorm vs. InstanceNorm vs. GroupNorm
- [BatchNorm 2015](https://arxiv.org/pdf/1502.03167.pdf)
- [LayerNorm 2016](https://arxiv.org/pdf/1607.06450.pdf)
- [InstanceNorm 2016](https://arxiv.org/pdf/1607.08022.pdf)
- [GroupNorm 2018](https://arxiv.org/pdf/1803.08494.pdf)
### BatchNorm
We normalize each feature, so for images we calculate means and variances for each channel and normalize each channel with its respective values. In addition **we learn new variances and means for each channel** and scale the samples again by these values.

With images `(100, 16, 64, 64) = (batch channel height width)``:
1. Normalize: 16 means and 16 variances, e.g. `x - torch.mean(x, dim=(0,2,3)).view(1,-1,1,1)`
2. Scaling: 16 means and variances

**Idea**: One could try to use `BatchNorm2d(track_running_stats=False)`, this should not shift things around during training if batches are small.

### LayerNorm
In BatchNorm we essentially took the mean over dimensions `dim=(0,2,3)`, now we can choose, but we do it over the last few `D` dimensions, e.g. over `dim=(1,2,3) = (channel height width)` or only over the last 2 dimensions `dim=(2,3)`. Seems a little bit counterintuitive to calculate the mean also over the channel dimension in my opinion, intuitively channels are the feature dimension and should be left independent.

With images `(100, 16, 64,64) = (batch channel height width)` and LayerNorm over 3 dimensions:
1. Normalize: 100 means and variances, e.g. `x = x - torch.mean(x, dim=(1,2,3)).view(-1,1,1,1)`
2. Scaling: once (16,64,64) means and once (16,64,64) variances.
**Meaning**: Normalizing is globally per instance, scaling is per pixel but the same for all instances.

### InstanceNorm
Operates on single sample as the name indicates. In BatchNorm we calculated a mean and variance for each channels over all samples in the mini-batch. **Now we ignore the mini-batch** and calculate the mean and variance for each sample independently a set of means and variances for all channels in that sample.

With images `(100, 16, 64,64) = (batch channel height width)`:
1. Normalize: 100 times 16 means and 16 variances, e.g. `x = x - torch.mean(x, dim=(2,3)).view(x.shape[0], x.shape[1], 1, 1)`
2. Scaling: Usually we don't apply any scaling, but if `affine=True` then we get the same scaling for every sample (same as in BatchNorm)

### GroupNorm
pass

###

## Image Inpainting
### RePaint
[RePaint 2022](https://arxiv.org/pdf/2201.09865.pdf)
- pretrained unconditional DDPM as generative prior
- only the initial sampling is conditioned on the image (in the regions where we have no mask)
- it goes forward and backward in diffusion time during inference (allows resampling and better semantic matching of reconstructed region). starting at complete noise; fixed amount of steps in reverse process; fixed amount of steps in forward process.
- sampling schedule proposed that greatly improves image quality
- they use human evaluation an LPIPS


## Image-Conditioned Diffusion Models
### ILVR
[ILVR 2021](https://arxiv.org/pdf/2108.02938.pdf)
- low-frequency information is used from conditional image - undersampling masks usually have a lot of low frequency information

### SEdit
[SDEdit 2022](https://arxiv.org/pdf/2108.01073.pdf)
- conditional image goes through forward process (but not completely) and is then denoised in the reverse process
- it is repeated several times and somehow merged for better results