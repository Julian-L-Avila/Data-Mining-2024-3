# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 1.- Intro e Instalación de Python
1. Start virtual environment
    source .venv/venv/bin/active
2. Copy .ipynb into a .sync.ipynb file
3. Convert .sync.ipynb to py:percent
    jupytext --to py:percent file.sync.ipynb
4. Make pair .py and .ipynb
    python -m jupyter_ascending.scripts.make_pair --base file
5. Start notebook
    python -m jupyter notebook file.sync.ipynb &

# %%
!pip3 install pandas

# %%
import pandas as pd
import numpy as np
