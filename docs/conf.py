import os
import sys

# Add project root to path so autodoc can import modules
sys.path.insert(0, os.path.abspath('..'))

project = 'CapturingUsersPredictionModels'
author = 'Onyberg'
release = '1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',   # NumPy/Google-style docstrings
    'sphinx.ext.viewcode',   # Links to source code
]

html_theme = 'sphinx_rtd_theme'

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
