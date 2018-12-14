
# coding: utf-8

# In[1]:


import simpy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime as dt

from tqdm import tqdm_notebook


# Simpy documentation - https://simpy.readthedocs.io/en/latest/contents.html 

# ## Useful functions

# In[2]:


def add_client_to_matrix(matrix, priority, call_start_time):
    """
    Calling at client_generator
    """
    data = np.array([-1]*matrix.shape[1])
    id_ = len(matrix)
    data[cl_columns_map['id']] = id_
    data[cl_columns_map['priority']] = priority
    data[cl_columns_map['call_start_time']] = call_start_time
    data[cl_columns_map['max_waiting_time']] = dt.timedelta(minutes=9).seconds #CHANGE TO RANDOM VALUE
    data[cl_columns_map['status']] = map_cl_status_code['generated']
    data[cl_columns_map['call_type']] = np.random.choice(calls_type_distr.index, p=calls_type_distr['p'])
    return id_, np.append(matrix, [data], axis=0)


# In[3]:


def add_operator_to_matrix(matrix, priority, start_work_time, work_duration=10*60):
    """
    Call manually when configuring operators shedule
    """
    data = np.array([-1]*matrix.shape[1])
    id_ = len(matrix)
    data[op_columns_map['id']] = id_
    data[op_columns_map['priority']] = priority
    data[op_columns_map['start_work_time']] = start_work_time
    data[op_columns_map['work_duration']] = work_duration
    return id_, np.append(matrix, [data], axis=0)


# In[4]:


def get_client_ds(env):
    """
    Get pretty DataFrame with all the clients information
    """
    matrix = env.client_mx
    client_ds = pd.DataFrame(matrix, columns=cl_columns, dtype=np.int)
    client_ds = client_ds.replace(-1,np.nan)
    client_ds['status'] = client_ds['status'].transform(lambda x: map_code_cl_status[x])
    client_ds['type'] = client_ds['priority'].transform(lambda x: {1:'gold',2:'silver',3:'regular'}[x])
    client_ds['call_type'] = client_ds['call_type'].transform(lambda x: calls_type_distr.at[x, 'type'])
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
    
    client_ds['hour'] = [x.hour for x in client_ds['call_start_time_dt']]
    return client_ds


# In[5]:


def get_operator_ds(env):
    matrix = env.op_mx
    operator_ds = pd.DataFrame(matrix, columns=op_columns, dtype=np.int)
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


# In[6]:


def get_client_calls_distribution(client_ds):
    call_numbers = client_ds.groupby(['type', 'hour'])['id'].count().to_frame()
    call_numbers['type'] = [x[0] for x in call_numbers.index.values]
    call_numbers['hour'] = [x[1] for x in call_numbers.index.values]
    call_numbers.index = range(len(call_numbers))
    call_numbers = call_numbers.pivot_table(columns=['type'], index=['hour'], values='id')
    call_numbers = call_numbers.reindex(columns=['gold','silver','regular'])
    return call_numbers


# In[7]:


def plot_client_calls_distribution(calls_distribution, figsize=(15,5), colors=['gold','silver','black']):
    plt.figure(figsize=figsize)
    for idx, f in enumerate(calls_distribution.columns):
        label = {0:'gold',1:'silver',2:'ordinary'}[idx]+' empirical'
        plt.plot(calls_distribution[f]/3600, '-', label=label, color=colors[idx])
    for idx, f in enumerate(call_frequency_ds.columns):
        label = {0:'gold',1:'silver',2:'ordinary'}[idx]+' theoretical'
        plt.plot(call_frequency_ds[f]/3600, '--', label=label, color=colors[idx])
    plt.legend()
    plt.title('Client calls distribution')
    plt.ylabel('probability')
    plt.xlabel('hour')
    plt.show()


# In[8]:


def plot_call_types_distribution(calls_type_distr, figsize=(8,4)):
    plt.figure(figsize=figsize)
    plt.bar(calls_type_distr.index-0.2, client_ds.groupby('call_type')['id'].count()/len(client_ds), label='empirical', width=0.4)
    plt.bar(calls_type_distr.index+0.2, calls_type_distr['p'], label='theoretical', width=0.4)
    plt.title('Call types distribution')
    plt.xticks(calls_type_distr.index, calls_type_distr['type'])
    plt.ylim(0,1)
    plt.ylabel('probability')
    plt.legend()
    plt.show()


# # Data preparation

# In[9]:


cl_statuses = ['generated', 'ask_for_line', 'get_line', 'no_lines', 'blocked',
               'unblocked', 'drop_on_unblock', 'in_queue', 'drop_from_queue', 'connected',
               'drop_success']
map_cl_status_code = {s:idx for idx,s in enumerate(cl_statuses)}
map_code_cl_status = {v:k for k,v in map_cl_status_code.items()}


# In[10]:


cl_columns = ['id','priority','call_start_time','call_end_time','max_waiting_time','status', 'call_type',
             'block_start_time', 'block_end_time',
             'queue_start_time', 'queue_end_time',
             'connect_start_time', 'connect_end_time', 'operator_id']
cl_columns_map = {k:idx for idx,k in enumerate(cl_columns)}


# In[11]:


op_columns = ['id', 'priority', 'start_work_time', 'work_duration']
op_columns_map = {k:idx for idx,k in enumerate(op_columns)}
op_mx = np.empty([0,len(op_columns)], dtype=np.int)
for p, swt in [(3, dt.timedelta(seconds=0).seconds),
               (3, dt.timedelta(minutes=10).seconds)]:
    id_, op_mx = add_operator_to_matrix(op_mx, p, swt)


# In[12]:


call_frequency_ds = pd.DataFrame()
call_frequency_ds['time_range'] = range(7,19)
call_frequency_ds['regular_clients'] = [87, 165, 236, 323, 277, 440, 269, 342, 175, 273, 115,  56]
call_frequency_ds['vip_clients'] = [89, 243, 221, 180, 301, 490, 394, 347, 240, 269, 145,  69]
call_frequency_ds['silver_clients'] = 0.68*call_frequency_ds['vip_clients']
call_frequency_ds['gold_clients'] = call_frequency_ds['vip_clients']-call_frequency_ds['silver_clients']
call_frequency_ds.index = call_frequency_ds['time_range']
call_frequency_ds = call_frequency_ds.reindex(columns=['gold_clients', 'silver_clients', 'regular_clients'])
call_frequency_ds


# In[13]:


calls_type_distr = pd.DataFrame([['ask', 'Вопрос', 0.16], ['book', 'Бронь', 0.76], ['rebook', 'Перебронь', 0.08]],
                                columns=['type', 'type_rus', 'p'])
calls_type_distr


# # Testing model

# In[14]:


class Queue(sp.PriorityStore):
    """
    Queue where clients are waiting till operators catch them.
    Combination of simpy PriorityStore and FilterStore.
    When calling get(filter) it return item (client) with least priority (the most important) who matches filter
    """
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


# In[15]:


class CallCenter(object):
    def __init__(self, env, n_lines, n_vip_lines):
        self.env = env
        self.n_lines = n_lines
        self.n_vip_lines = n_vip_lines
        self.lines = sp.Resource(env, capacity=self.n_lines)
        self.queue = Queue(env)

    def request_line(self, client):
        """
        Execute when client starts calling
        """
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
        """
        Execute when client dropped the phone
        """
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
            # If client is still in queue, estimate its waiting time
            if client in [x.item for x in self.queue.items]:
                yield self.queue.get(lambda x: x.item.id_==client.id_)
                client.waiting_queue.interrupt()
                self.env.process(self.block_client(client))
        else:
            self.env.process(self.block_client(client))
            
    def request_client(self, operator):
        """
        Execute when operator is freed and ready to take another client
        """
        op_id = operator.id_
        op_priority = operator.get_mx_field('priority')
        client = yield self.queue.get(lambda cl: cl.priority<=op_priority)
        client = client.item
        return client
    
    def block_client(self, client):
        """
        Execute when client should be blocked.
        For regular clients: "blocking" is when system "estimates" time for client to wait in queue
        For vip clients: "blocking" is "entering" vip card ids
        """
        client.block.succeed()
        cl_priority = client.get_mx_field('priority')
        if cl_priority == 3:
            block_time = 7
        else:
            block_time = 10  #CHANGE TO RANDOM VALUE
        yield self.env.timeout(block_time)
        ttw = self.estimate_wait_time(client)
        dropped = yield self.env.process(client.decide_to_drop_unblock(time_to_wait=ttw))
        if not dropped:
            yield self.queue.put(sp.PriorityItem(cl_priority, client))
            client.put_in_queue.succeed()
    
    def estimate_wait_time(self, client):
        """
        Function to estimate how much time client will wait in queue
        """
        return 0  # For testing purposes only
    
    class NoLinesAvailable(sp.exceptions.SimPyException):
        """
        Special exception saying there no free phone lines that client cat occupy
        """
        pass


# In[16]:


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
        """
        When client is just generated and calling to system
        """
        cc = self.env.call_center
        try:
            self.set_status('ask_for_line')
            yield self.env.process(cc.request_line(self))
        except cc.NoLinesAvailable as e:
            self.set_status('no_lines')
            return
    
    def drop_call(self, status):
        """
        When client wants to drop the phone by some reason given in 'status'
        """
        yield self.env.process(self.env.call_center.release_line(self))
        self.set_mx_field('call_end_time', self.env.now)
        self.set_status(status)
    
    def wait_in_queue(self):
        """
        When client waiting in queue till he/she decide to drop the phone
        """
        while True:
            yield self.put_in_queue
            self.set_status('in_queue')
            self.set_mx_field('queue_start_time', self.env.now)
            try:
                max_queue_wait_time = dt.timedelta(minutes=10).seconds #CHANGE TO TASK GIVEN LIMITATIONS
                yield self.env.timeout(max_queue_wait_time)
                yield self.env.process(self.drop_call('drop_from_queue'))
            except sp.Interrupt: 
                pass
            self.set_mx_field('queue_end_time', self.env.now)
            self.put_in_queue = self.env.event()
        
    def wait_in_block(self):
        """
        When client is blocked and "do" something mentioned in CallCenter.block_client()
        """
        yield self.block
        self.set_status('blocked')
        self.set_mx_field('block_start_time', self.env.now)
        
    def decide_to_drop_unblock(self, time_to_wait):
        """
        When client is told how much time he/she will wait in queue
        """
        mwt = self.get_mx_field('max_waiting_time')
        self.set_mx_field('block_end_time', self.env.now)
        if time_to_wait > mwt:
            yield self.env.process(self.drop_call('drop_on_unblock'))
            return True
        return False
        
    def talk_with_operator(self):
        """
        When client is connected with operator and "talks" with him/her
        """
        yield self.connect
        self.waiting_queue.interrupt()
        self.set_status('connected')
        self.set_mx_field('connect_start_time', self.env.now)
        yield self.disconnect
        self.set_mx_field('connect_end_time', self.env.now)
        yield self.env.process(self.drop_call('drop_success'))
    
    def set_mx_field(self, field, value):
        """
        Function for code to be more readable
        """
        self.env.client_mx[self.id_, cl_columns_map[field]] = value 
    
    def set_status(self, status):
        """
        Function for code to be more readable
        """
        self.set_mx_field('status', map_cl_status_code[status])
        
    def set_call_end_time(self):
        """
        Function for code to be more readable
        """
        self.set_mx_field('call_end_time', self.env.now)
    
    def get_mx_field(self, field):
        """
        Function for code to be more readable
        """
        return self.env.client_mx[self.id_, cl_columns_map[field]]


# In[17]:


class Operator(object):
    def __init__(self, env, id_):
        self.env = env
        self.id_ = id_
        
        self.action = env.process(self.start_working())
        
    def start_working(self):
        swt = self.get_mx_field('start_work_time')
        yield self.env.timeout(swt)
        yield self.env.process(self.working())
        
    def working(self):
        swt, wd = [self.get_mx_field(f) for f in ['start_work_time', 'work_duration']]
        while self.env.now<swt+wd:         
            client = yield self.env.process(self.env.call_center.request_client(self))
            client.set_mx_field('operator_id', self.id_)
            client.connect.succeed()
            yield self.env.timeout(dt.timedelta(minutes=1).seconds) #CHANGE TO RANDOM VALUES
            client.disconnect.succeed()
            
    def get_mx_field(self, field):
        """
        Function for code to be more readable
        """
        return self.env.op_mx[self.id_, op_columns_map[field]]


# In[18]:


def client_generator(env):
    while True:
        for cl_priority in range(1,4):
            p = 0.5  #CHANGE_TO_TASK_VALUE
            p = call_frequency_ds.iat[env.now//3600, cl_priority-1]/3600
            if np.random.rand()<p:
                id_, env.client_mx = add_client_to_matrix(env.client_mx, cl_priority, env.now)
                client = Client(env, id_)
        yield env.timeout(1)


# In[19]:


def init_env():
    env = sp.Environment()
    env.client_mx = np.empty([0,len(cl_columns)], dtype=np.int) #matrix to write clients data
    env.op_mx = op_mx #matrix with operators data
    env.call_center = CallCenter(env, n_lines=2, n_vip_lines=0) 
    env.client_generator = env.process(client_generator(env))
    env.operators = [Operator(env, id_) for id_ in op_mx[:,op_columns_map['id']]]
    return env


# In[20]:


env = init_env()
for i in tqdm_notebook(range(dt.timedelta(hours=12, minutes=0, seconds=0).seconds)):
    env.run(until=i+1)


# In[21]:


client_ds = get_client_ds(env)
print(client_ds.shape)
client_ds.head()


# In[22]:


op_ds = get_operator_ds(env)
print(op_ds.shape)
op_ds.head()


# In[23]:


plot_call_types_distribution(calls_type_distr)

