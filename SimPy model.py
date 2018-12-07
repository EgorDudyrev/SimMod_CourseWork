
# coding: utf-8

# In[1]:


import simpy as sp
import numpy as np
import pandas as pd

import datetime as dt

from tqdm import tqdm_notebook


# Simpy documentation - https://simpy.readthedocs.io/en/latest/contents.html 

# ## Useful functions

# In[2]:


def add_client_to_matrix(matrix, priority, call_start_time):
    data = np.array([-1]*matrix.shape[1])
    id_ = len(matrix)
    data[cl_columns_map['id']] = id_
    data[cl_columns_map['priority']] = priority
    data[cl_columns_map['call_start_time']] = call_start_time
    data[cl_columns_map['max_waiting_time']] = 9*60
    data[cl_columns_map['status']] = map_cl_status_code['generated']
    return id_, np.append(matrix, [data], axis=0)


# In[3]:


def add_operator_to_matrix(matrix, priority, start_work_time, work_duration=10*60):
    data = np.array([-1]*matrix.shape[1])
    id_ = len(matrix)
    data[op_columns_map['id']] = id_
    data[op_columns_map['priority']] = priority
    data[op_columns_map['start_work_time']] = start_work_time
    data[op_columns_map['work_duration']] = work_duration
    return id_, np.append(matrix, [data], axis=0)


# In[4]:


def get_client_ds(matrix, columns):
    client_ds = pd.DataFrame(matrix, columns=columns, dtype=np.int)
    client_ds = client_ds.replace(-1,np.nan)
    client_ds['status'] = client_ds['status'].transform(lambda x: map_code_cl_status[x])
    client_ds['type'] = client_ds['priority'].transform(lambda x: {1:'gold',2:'silver',3:'regular'}[x])
    client_ds = client_ds.drop('priority',axis=1)
    client_ds['max_waiting_time_dt'] = client_ds['max_waiting_time'].transform(lambda x: dt.timedelta(seconds=x))
    client_ds = client_ds.drop('max_waiting_time',axis=1)
    
    for i in ['call','block','queue','connect']:
        client_ds[f'{i}_duration_time'] = client_ds[f'{i}_end_time']-client_ds[f'{i}_start_time']
    
    for i in ['call', 'block', 'queue', 'connect']:
        for j in ['start','end','duration']:
            client_ds[f'{i}_{j}_time_dt'] = client_ds[f'{i}_{j}_time'].transform(
                lambda x: dt.timedelta(seconds=x) if x>=0 else None)
            if j!='duration':
                client_ds[f'{i}_{j}_time_dt'] = client_ds[f'{i}_{j}_time_dt']+dt.datetime(2018,1,1,7)
            client_ds = client_ds.drop(f'{i}_{j}_time', axis=1)
    
    return client_ds


# In[5]:


def get_operator_ds(matrix, columns):
    operator_ds = pd.DataFrame(matrix, columns=columns, dtype=np.int)
    operator_ds = operator_ds.replace(-1, np.nan)
    operator_ds['type'] = operator_ds['priority'].transform(lambda x: {1:'gold',2:'silver',3:'regular'}[x])
    operator_ds['end_work_time'] = operator_ds['start_work_time']+operator_ds['work_duration']
    for i in ['start_work_time', 'end_work_time', 'work_duration']:
            operator_ds[f'{i}_dt'] = operator_ds[i].transform(
                lambda x: dt.timedelta(seconds=x) if x>=0 else None)
            if 'duration' not in i:
                operator_ds[f'{i}_dt'] = operator_ds[f'{i}_dt']+dt.datetime(2018,1,1,7)
    
    operator_ds = operator_ds.drop(['priority', 'start_work_time', 'work_duration', 'end_work_time'], axis=1)
    return operator_ds


# # Data preparation

# In[6]:


cl_statuses = ['generated', 'ask_for_line', 'get_line', 'no_lines', 'blocked',
               'unblocked', 'drop_on_unblock', 'in_queue', 'drop_from_queue', 'connected',
               'drop_success']
map_cl_status_code = {s:idx for idx,s in enumerate(cl_statuses)}
map_code_cl_status = {v:k for k,v in map_cl_status_code.items()}


# In[7]:


cl_columns = ['id','priority','call_start_time','call_end_time','max_waiting_time','status',
             'block_start_time', 'block_end_time',
             'queue_start_time', 'queue_end_time',
             'connect_start_time', 'connect_end_time', 'operator_id']
cl_columns_map = {k:idx for idx,k in enumerate(cl_columns)}


# In[8]:


op_columns = ['id', 'priority', 'start_work_time', 'work_duration']
op_columns_map = {k:idx for idx,k in enumerate(op_columns)}
op_mx = np.empty([0,len(op_columns)], dtype=np.int)
for p, swt in [(3, 0),
               (3, 10*60)]:
    id_, op_mx = add_operator_to_matrix(op_mx, p, swt)


# In[9]:


VERY_LONG_TIME = 12*60


# # Testing model

# In[10]:


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


# In[11]:


class CallCenter(object):
    def __init__(self, env, n_lines, n_vip_lines):
        self.env = env
        self.n_lines = n_lines
        self.n_vip_lines = n_vip_lines
        self.lines = sp.Resource(env, capacity=self.n_lines)
        self.queue = Queue(env)

    def request_line(self, client):
        cl_id = client.id_
        cl_priority =  client.get_mx_field('priority')
        n_lines_to_give = self.n_lines-self.n_vip_lines if cl_priority==3 else self.n_lines
        if self.lines.count<n_lines_to_give:
            req = self.lines.request()
            req.cl_id = cl_id
            client.req = req
            yield req
            client.set_status('get_line')
        else:
            raise self.NoLinesAvailable()
        yield self.env.process(self.put_to_queue(client))
    
    def release_line(self, client):
        if client.id_ in [i.item.id_ for i in self.queue.items]:
            yield self.queue.get(lambda x: x.item.id_==client.id_)
        yield self.lines.release(client.req)
    
    def put_to_queue(self, client):
        cl_id = client.id_
        cl_priority = client.get_mx_field('priority')
        if cl_priority == 3:
            yield self.queue.put(sp.PriorityItem(cl_priority, client))
            client.put_in_queue.succeed()
            yield self.env.timeout(1)
            # Если клиент всё ещё в очереди, то оцениваем время его ожидания
            if client in [x.item for x in self.queue.items]:
                yield self.queue.get(lambda x: x.item.id_==client.id_)
                client.waiting_queue.interrupt()
                self.env.process(self.block_client(client))
        else:
            self.env.process(self.block_client(client))
            
    def request_client(self, operator):
        op_id = operator.id_
        op_priority = self.env.op_mx[operator.id_, op_columns_map['priority']]
        client = yield self.queue.get(lambda x: x.priority>=op_priority)
        client = client.item
        return client
    
    def block_client(self, client):
        client.block.succeed()
        cl_priority = client.get_mx_field('priority')
        block_time = 7 if cl_priority == 3 else 10 
        yield self.env.timeout(block_time)
        ttw = self.estimate_wait_time(client)
        dropped = yield self.env.process(client.decide_to_drop_unblock(time_to_wait=ttw))
        if not dropped:
            yield self.queue.put(sp.PriorityItem(cl_priority, client))
            client.put_in_queue.succeed()
    
    def estimate_wait_time(self, client):
        return 0  # For testing purposes only
    
    class NoLinesAvailable(sp.exceptions.SimPyException):
        pass


# In[12]:


class Client(object):
    def __init__(self, env, id_):
        self.env = env
        self.id_ = id_
        
        self.put_in_queue, self.get_from_queue = env.event(), env.event()
        self.block, self.unblock = env.event(), env.event()
        self.connect, self.disconnect = env.event(), env.event()
                
        self.waiting_block = env.process(self.wait_in_block())
        self.waiting_queue = env.process(self.wait_in_queue())
        self.talking_proc = env.process(self.talk_with_operator())
        self.action = env.process(self.call())
    
    def call(self):
        cc = self.env.call_center
        try:
            self.set_status('ask_for_line')
            yield self.env.process(cc.request_line(self))
        except cc.NoLinesAvailable as e:
            self.set_status('no_lines')
            return
    
    def drop_call(self, status):
        yield self.env.process(self.env.call_center.release_line(self))
        self.set_mx_field('call_end_time', self.env.now)
        self.set_status(status)
    
    def wait_in_queue(self):
        while True:
            yield self.put_in_queue
            self.set_status('in_queue')
            self.set_mx_field('queue_start_time', self.env.now)
            try:
                yield self.env.timeout(VERY_LONG_TIME)
                yield self.env.process(self.drop_call('drop_from_queue'))
            except sp.Interrupt: 
                pass
            self.set_mx_field('queue_end_time', self.env.now)
            self.put_in_queue = self.env.event()
        
    def wait_in_block(self):
        yield self.block
        self.set_status('blocked')
        self.set_mx_field('block_start_time', self.env.now)
        
    def decide_to_drop_unblock(self, time_to_wait):
        mwt = self.get_mx_field('max_waiting_time')
        self.set_mx_field('block_end_time', self.env.now)
        if time_to_wait > mwt:
            yield self.env.process(self.drop_call('drop_on_unblock'))
            return True
        return False
        
    def talk_with_operator(self):
        yield self.connect
        self.waiting_queue.interrupt()
        self.set_status('connected')
        self.set_mx_field('connect_start_time', self.env.now)
        yield self.disconnect
        self.set_mx_field('connect_end_time', self.env.now)
        yield self.env.process(self.drop_call('drop_success'))
    
    def set_mx_field(self, field, value):
        self.env.client_mx[self.id_, cl_columns_map[field]] = value
    
    def set_status(self, status):
        self.set_mx_field('status', map_cl_status_code[status])
        
    def set_call_end_time(self):
        self.set_mx_field('call_end_time', self.env.now)
    
    def get_mx_field(self, field):
        return self.env.client_mx[self.id_, cl_columns_map[field]]


# In[13]:


class Operator(object):
    def __init__(self, env, id_):
        self.env = env
        self.id_ = id_
        
        self.action = env.process(self.start_working())
        
    def start_working(self):
        swt = self.env.op_mx[self.id_, op_columns_map['start_work_time']]
        yield self.env.timeout(swt)
        yield self.env.process(self.working())
        
    def working(self):
        swt = self.env.op_mx[self.id_, op_columns_map['start_work_time']]
        wd = self.env.op_mx[self.id_, op_columns_map['work_duration']]
        while self.env.now<swt+wd:         
            client = yield self.env.process(self.env.call_center.request_client(self))
            client.set_mx_field('operator_id', self.id_)
            client.connect.succeed()
            yield self.env.timeout(1*60)
            client.disconnect.succeed()


# In[14]:


def client_generator(env):
    while True:
        if np.random.rand()<0.5:
            id_, env.client_mx = add_client_to_matrix(env.client_mx, 3, env.now)
            client = Client(env, id_)
        yield env.timeout(1)


# In[15]:


def init_env():
    env = sp.Environment()
    env.client_mx = np.empty([0,len(cl_columns)], dtype=np.int)
    env.op_mx = op_mx
    env.call_center = CallCenter(env, 2,0)
    env.client_generator = env.process(client_generator(env))
    env.operators = [Operator(env, id_) for id_ in op_mx[:,op_columns_map['id']]]
    return env


# In[16]:


env = init_env()
for i in tqdm_notebook(range(20*60)):
    env.run(until=i+1)


# In[17]:


client_ds = get_client_ds(env.client_mx, cl_columns)
print(client_ds.shape)
client_ds.head()


# In[18]:


client_ds['status'].value_counts()


# In[19]:


op_ds = get_operator_ds(env.op_mx, op_columns)
print(op_ds.shape)
op_ds.head()

