
# coding: utf-8

# In[1]:


import simpy as sp
import numpy as np
import pandas as pd

import datetime as dt

from tqdm import tqdm_notebook


# Simpy documentation - https://simpy.readthedocs.io/en/latest/contents.html 

# # Data preparation

# In[2]:


cl_statuses = ['generated', 'ask_for_line', 'get_line', 'no_lines', 'blocked',
               'unblocked', 'drop_on_unblock', 'in_queue', 'drop_from_queue', 'connected',
               'drop_success']
map_cl_status_code = {s:idx for idx,s in enumerate(cl_statuses)}
map_code_cl_status = {v:k for k,v in map_cl_status_code.items()}


# In[3]:


cl_columns = ['id','priority','call_start_time','call_end_time','max_waiting_time','status']
cl_columns_map = {k:idx for idx,k in enumerate(cl_columns)}


# ## Useful functions

# In[4]:


def add_client_to_matrix(matrix, priority, call_start_time):
    data = np.array([-1]*matrix.shape[1])
    id_ = len(matrix)
    data[cl_columns_map['id']] = id_
    data[cl_columns_map['priority']] = priority
    data[cl_columns_map['call_start_time']] = call_start_time
    data[cl_columns_map['max_waiting_time']] = 5*60
    data[cl_columns_map['status']] = map_cl_status_code['generated']
    return id_, np.append(matrix, [data], axis=0)


# In[5]:


def get_client_ds(matrix, columns):
    client_ds = pd.DataFrame(matrix, columns=columns, dtype=np.int)
    client_ds['status_code'] = client_ds['status']
    client_ds['status'] = client_ds['status'].transform(lambda x: map_code_cl_status[x])
    client_ds['type'] = client_ds['priority'].transform(lambda x: {1:'gold',2:'silver',3:'regular'}[x])
    client_ds['call_start_time_dt'] = client_ds['call_start_time'].transform(lambda x: dt.timedelta(seconds=x))
    client_ds['call_start_time_dt'] = client_ds['call_start_time_dt'] + dt.datetime(2018,1,1,7)
    client_ds['call_end_time_dt'] = client_ds['call_end_time'].transform(lambda x: dt.timedelta(seconds=x) if x>=0 else None)
    client_ds['call_end_time_dt'] = client_ds['call_end_time_dt'] + dt.datetime(2018,1,1,7)
    client_ds['max_waiting_time_dt'] = client_ds['max_waiting_time'].transform(lambda x: dt.timedelta(seconds=x))
    client_ds = client_ds.reindex(columns=['id','priority','type', 'status_code','status',
                           'call_start_time','call_start_time_dt','call_end_time', 'call_end_time_dt',
                           'max_waiting_time', 'max_waiting_time_dt'])
    return client_ds


# # Testing model

# In[6]:


class Queue(sp.PriorityStore):
    
    get = sp.core.BoundClass(sp.resources.store.FilterStoreGet)
    """Request a to get an *item*, for which *filter* returns ``True``, out of
    the store."""

    def _do_get(self, event):
        for item in self.items:
            if event.filter(item):
                self.items.remove(item)
                event.succeed(item)
                break
        return True


# In[7]:


class CallCenter(object):
    def __init__(self, env, n_lines, n_vip_lines):
        self.env = env
        self.n_lines = n_lines
        self.n_vip_lines = n_vip_lines
        self.lines = sp.Resource(env, capacity=self.n_lines)
        self.queue = Queue(env)

    def request_line(self, cl_id):
        cl_priority = self.env.client_mx[cl_id, cl_columns_map['priority']]
        n_lines_to_give = self.n_lines-self.n_vip_lines if cl_priority==3 else self.n_lines
        Client.set_status_by_id(self.env, cl_id, 'ask_for_line')
        if self.lines.count<n_lines_to_give:
            req = self.lines.request()
            req.cl_id = cl_id
            yield req
            Client.set_status_by_id(self.env, cl_id, 'get_line')
        else:
            Client.set_status_by_id(self.env, cl_id, 'no_lines')
            raise self.NoLinesAvailable()
        Client.set_status_by_id(self.env, cl_id, 'blocked')
        yield self.env.timeout(8)
        Client.set_status_by_id(self.env, cl_id, 'unblocked')
        yield self.queue.put(sp.PriorityItem(cl_priority, cl_id))
        Client.set_status_by_id(self.env, cl_id, 'in_queue')
        return req
        
    def release_line(self, req):
        if req.cl_id in [i.item for i in self.queue.items]:
            yield self.queue.get(lambda x: x.item==req.cl_id)
        yield self.lines.release(req)
        Client.set_call_end_time_by_id(self.env, req.cl_id)
    
    class NoLinesAvailable(sp.exceptions.SimPyException):
        pass


# In[8]:


class Client(object):
    def __init__(self, env, id_):
        self.env = env
        self.id_ = id_
        self.action = env.process(self.run())
    
    def run(self):
        cc = self.env.call_center
        try:
            req = yield self.env.process(cc.request_line(self.id_))
        except cc.NoLinesAvailable as e:
            return
        yield self.env.timeout(self.env.client_mx[self.id_,cl_columns_map['max_waiting_time']])
        yield self.env.process(cc.release_line(req))
        self.set_status('drop_from_queue')
        
    @staticmethod
    def set_status_by_id(env, id_, status):
        env.client_mx[id_, cl_columns_map['status']] = map_cl_status_code[status]
    
    def set_status(self, status):
        self.set_status_by_id(self.env, self.id_, status)
        
    @staticmethod
    def set_call_end_time_by_id(env, id_):
        env.client_mx[id_, cl_columns_map['call_end_time']] = env.now
    
    def set_call_end_time(self):
        self.set_call_end_time_by_id(self.env, self.id_)


# In[ ]:


class Operator(object):
    def __init__(self, env, id_):
        self.env = env
        self.id_ = id_
        self.action = env.process(self.run())
        
    def run(self):
        pass


# In[9]:


def client_generator(env):
    while True:
        if np.random.rand()<0.5:
            id_, env.client_mx = add_client_to_matrix(env.client_mx, 3, env.now)
            client = Client(env, id_)
        yield env.timeout(1)


# In[ ]:


def 


# In[10]:


def init_env():
    env = sp.Environment()
    env.client_mx = np.empty([0,len(cl_columns)])
    env.call_center = CallCenter(env, 2,0)
    env.client_generator = env.process(client_generator(env))
    return env


# In[11]:


env = init_env()
for i in tqdm_notebook(range(6*60)):
    env.run(until=i+1)


# In[12]:


client_ds = get_client_ds(env.client_mx, cl_columns)
print(client_ds.shape)
client_ds.head()


# In[13]:


client_ds['status'].value_counts()

