
# coding: utf-8

# In[1]:


import simpy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import datetime as dt
from itertools import product

from tqdm import tqdm_notebook


# In[2]:


import importlib
try:
    importlib.reload(sm)
except:
    import sim_model as sm


# Simpy documentation - https://simpy.readthedocs.io/en/latest/contents.html 

# # Data preparation

# In[3]:


try:
    simulation_ds = pd.read_csv('simulation_ds.csv', index_col=0)
except:
    simulation_ds = pd.DataFrame(columns=[
    'n_lines', 'n_vip_lines', 'cost',
    'gold_wait', 'silver_wait', 'regular_wait',
    'vip_no_lines', 'regular_no_lines',
    'gold_7', 'gold_8', 'gold_9', 'gold_10', 'gold_11',
    'silver_7', 'silver_8', 'silver_9', 'silver_10', 'silver_11',
    'regular_7', 'regular_8', 'regular_9', 'regular_10', 'regular_11',
    ])


# # Set simulation and run

# In[4]:


n_lines = 55
n_vip_lines = 3


# In[5]:


op_time_ds = pd.DataFrame(index=range(7,12))
op_time_ds['gold'] = [3, 4, 0, 0, 3]
op_time_ds['silver'] = [5, 6, 0, 5, 4]
op_time_ds['regular'] = [6, 10, 10, 7,6]
op_time_ds


# In[6]:


client_ds, op_ds, sim_data = sm.run_simulation(op_time_ds, n_lines, n_vip_lines, verb=True)


# In[7]:


simulation_ds = simulation_ds.append(sim_data, ignore_index=True)
simulation_ds.to_csv('simulation_ds.csv')


# In[8]:


sm.plot_clients_no_lines(client_ds)


# In[9]:


sm.plot_clients_waitings(client_ds)


# In[10]:


sm.plot_clients_success(client_ds)


# # Find best parameters

# In[11]:


cds = simulation_ds.copy()


# In[12]:


cds = cds[(cds['gold_wait']>=0.98)&(cds['silver_wait']>=0.95)&(cds['regular_wait']>=0.85)
   &(cds['regular_no_lines']<=0.2)&(cds['vip_no_lines']<=0.02)]


# In[13]:


cds.shape


# In[14]:


cds.sort_values('cost').iloc[0]

