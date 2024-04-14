import os
import sys
sys.path.append(os.path.abspath("../.."))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'diffusionmodels'
copyright = '2024, Lionel Peer'
author = 'Lionel Peer'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints"
]

autosummary_mock_imports = [
    "torch",
    "jaxtyping",
    "numpy",
    "wandb",
    "torchvision",
    "h5py",
    "tqdm"
    "time",
    "typing",
    "math"
]

templates_path = ['_templates']
exclude_patterns = []
source_suffix = [".rst", ".md"]

autosummary_generate = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']