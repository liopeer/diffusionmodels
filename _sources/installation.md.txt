# Installation
If you don't intend to make changes to the models I would use one of the [General Usage](#general-usage),
else I recommend going with one of the [Development](#development) options.

(general-usage)=
## General Usage
### Conda
```bash
git clone https://github.com/liopeer/diffusionmodels.git
cd diffusionmodels
conda env create -f environment.yml
conda activate liotorch
pip install .
```
### Docker

(development)=
## Development
### Conda
```bash
git clone https://github.com/liopeer/diffusionmodels.git
cd diffusionmodels
conda env create -f environment.yml
```
### Docker