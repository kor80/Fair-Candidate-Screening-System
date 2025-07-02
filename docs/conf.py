# Configuration file for the Sphinx documentation builder.

project = 'Fair-Candidate-Screening-System'
author = 'CAT'
release = '0.1'

extensions = [
    'nbsphinx',
    'sphinx.ext.mathjax',  # optional, for math support
]

# Tell nbsphinx to allow execution or not (optional)
nbsphinx_execute = 'never'  # or 'auto' if you want to run notebooks on build

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'alabaster'
html_static_path = ['_static']
