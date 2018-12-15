
# coding: utf-8

# In[ ]:


import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import seaborn as sns

from tqdm import tqdm_notebook


# In[ ]:


import importlib
try:
    importlib.reload(sm)
except:
    import sim_model as sm


# In[ ]:


from IPython.display import clear_output


# In[ ]:


print(nx.__version__)
print(np.__version__)


# # Define functions

# In[ ]:


MAX_OPERATORS_START_ON_HOUR = 10


# In[ ]:


def generate_gene():
    return np.random.randint(0,MAX_OPERATORS_START_ON_HOUR,15)

def get_cost(stat):
    cost = (np.array([stat['gold_wait']<0.98, stat['silver_wait']<0.95, stat['regular_wait']<0.85,
                      stat['regular_no_lines']>0.2, stat['vip_no_lines']>0.02
                     ])*1e9
           ).sum()+stat['cost']
    return cost

def get_fitness(stat):
    return 1e6/get_cost(stat)

def select_pairs(chromos):
    return np.array([(chromos[i], chromos[j]) for i,j in np.random.randint(0, len(chromos), size=(N//2, 2))])

def crossover(a, b):
    return np.array([i if np.random.rand()<0.5 else j for i,j in zip(a,b)])

def mutation(chromos):
    mutated = np.array([i if np.random.rand()<1-MUTATION_P else np.random.randint(MAX_OPERATORS_START_ON_HOUR)
                        for i in chromos])
    return mutated

def reduction(old_chromos, children, fits=None, is_manual=False):
    if is_manual:
        new_chromos = np.vstack([old_chromos[:1], children[:len(old_chromos)-1]])
    else:
        old_chromos_ids = list(range(len(old_chromos)))
        old_chromos_fits_ar = [(i, f) for i,f in zip(old_chromos_ids, fits)]
        old_chromos_ids = [i for i,f in sorted(old_chromos_fits_ar, key=lambda x: -x[1])]
        old_chromos = old_chromos[old_chromos_ids]
        new_chromos = np.vstack([old_chromos[:1],children[:len(old_chromos)-1]])
    return new_chromos

def select_manually(chromos):
    to_print = [(i, get_path_len(chromo), chromo,) for i,chromo in enumerate(chromos)]
    to_print = sorted(to_print, key = lambda line: line[1])
    ids_map = {line[0]:idx for idx, line in enumerate(to_print)}
    ids_map_r = {v:k for k,v in ids_map.items()}
    if CLEAR_OUTPUT: clear_output()
    print('Select best chromosomes')
    for line in to_print:
        print("{id_}: {l}, (len={length})".format(id_=ids_map[line[0]], l=line[2], length=line[1]))
    print('Enter best chromosomes ids:')
    ids = input().strip()
    if ids=='stop':
        raise StopIteration
    if len(ids)>0:
        ids = list(map(int, ids.split(' ')))
        ids = [ids_map_r[i] for i in ids]
        not_ids = [ids_map_r[i] for i in ids_map_r if i not in ids]
        parents = chromos[ids]
        chromos = np.vstack([chromos[ids], chromos[not_ids]])
    else:
        parents = chromos
    print('-----------------\n')
    return parents, chromos
# In[ ]:


def select_auto(chromos, fits):
    probs = np.exp(fits)/np.exp(fits).sum() #Softmax function
    parents = np.array([chromos[i] for i in np.random.choice(range(len(chromos)), p=probs, size=len(chromos))])
    return parents


# In[ ]:


def ga_step(old_chromos, is_manual=False, n_lines=55, n_vip_lines=5):
    chromos = old_chromos.copy()
    #if len(chromos)%2==1: raise Exception('chromos number should be odd')
    # step2: calculate fitness
    stats_sim = np.array([sm.run_simulation(c, n_lines=55, n_vip_lines=5, verb=False, only_stat=True) for c in chromos])
    fits = np.array([get_fitness(stat) for stat in stats_sim])
    fits = fits/(fits.mean()+1e-10)
    # step3: select parents vectors
    if is_manual:
        parents, chromos = select_manually(chromos)
    else:
        parents = select_auto(chromos, fits)
    
    # step4: set pairs. apply crossover and mutation
    pairs = select_pairs(parents)
    children = np.array([crossover(pair[0],pair[1]) for pair in np.vstack([pairs,pairs])]) #Each pair gives 2 children
    mutated = np.array([mutation(child) for child in children])
    new_chromos = reduction(chromos, mutated, fits, is_manual)
    
    stats = np.array([old_chromos, parents, children, mutated, new_chromos])
    return new_chromos, stats, stats_sim


# # Create graph

# # Manual optimization
N=10
chromo_len = 10
is_manual = True
CLEAR_OUTPUT = False
MUTATION_P = 0.3chromos = np.array([generate_path(chromo_len, start_node, end_node, N_NODES) for i in range(N)])
all_chromos = np.array([chromos])
min_path_lens = np.array([min([get_path_len(i) for i in chromos])])
stats = []for i in tqdm_notebook(range(1000)):
    try:
        chromos, stat = ga_step(A, chromos, is_manual=is_manual)
    except StopIteration:
        print("Stopped at",i)
        break
    stats.append(stat)
    
    min_path_lens = np.append(min_path_lens, min([get_path_len(i) for i in chromos]))
    all_chromos = np.append(all_chromos, [chromos], axis=0)
    
    min_len = min_path_lens[-1]
    if not is_manual and i>=early_stopping_steps \
        and min_len!=np.inf and all(min_len==min_path_lens[- early_stopping_steps:]):
        print('early stops at',i)
        print('result', min_len)
        break
stats = np.array(stats)plt.figure(figsize=(15,10))
plt.subplot(231)
sns.heatmap(stats[0][0], cmap=colors, cbar=False, annot=True)
plt.ylabel('Хромосомы')
plt.xlabel('Гены')
plt.title('Исходные хромосомы')

plt.subplot(232)
sns.heatmap(stats[0][1], cmap=colors, cbar=False, annot=True)
plt.ylabel('Хромосомы')
plt.xlabel('Гены')
plt.title('Селекционированные хромосомы')

plt.subplot(233)
sns.heatmap(stats[0][2], cmap=colors, cbar=False, annot=True)
plt.ylabel('Хромосомы')
plt.xlabel('Гены')
plt.title('Скрещенные хромосомы')

plt.subplot(234)
sns.heatmap(stats[0][3], cmap=colors, cbar=False, annot=True)
plt.ylabel('Хромосомы')
plt.xlabel('Гены')
plt.title('Мутировавшие хромосомы')

plt.subplot(235)
sns.heatmap(stats[0][4], cmap=colors, cbar=False, annot=True)
plt.ylabel('Хромосомы')
plt.xlabel('Гены')
plt.title('Редуцированные хромосомы')
plt.suptitle('Развитие хромосом на первой итерации')
plt.show()plot_data = np.array(min_path_lens)
#plot_data[np.isinf(plot_data)]=plot_data[~np.isinf(plot_data)].max()*1.2 #Replace inf with finite numbers
plot_data[np.isinf(plot_data)]=3#plot_data[~np.isinf(plot_data)].max()*1.2 #Replace inf with finite numbers

plt.plot(plot_data)
plt.title('Минимальный найденный путь')
plt.ylabel('Минимальное расстояние')
plt.xlabel('Итерация')
#plt.yscale('log')
plt.show()best_chromo = chromos[np.argmin([get_path_len(i) for i in chromos])]
best_chromo = best_chromo[np.append(best_chromo[:-1]!=best_chromo[1:], True)]
best_edges = [(best_chromo[i], best_chromo[i+1]) for i in range(len(best_chromo[:-1]))]

plot_graph(G, best_edges=best_edges, title='Найденный путь', colors=colors)
# # Automatic optimization

# In[ ]:


N=10


# In[ ]:


early_stopping_steps = 30
is_manual = False
MUTATION_P = 0.3
n_lines = 50
n_vip_lines = 5


# In[ ]:


chromos = np.array([generate_gene() for i in range(N)])
all_chromos = np.array([chromos])
stats_sim = np.array([sm.run_simulation(c, n_lines=n_lines, n_vip_lines=n_vip_lines, verb=False, only_stat=True) for c in chromos])
min_costs = np.array([min([get_cost(stat) for stat in stats_sim])])
stats = []


# In[ ]:


for i in tqdm_notebook(range(100)):
    try:
        chromos, stat, stats_sim = ga_step(chromos, is_manual=is_manual, n_lines=n_lines, n_vip_lines=n_vip_lines)
    except StopIteration:
        print("Stopped at",i)
        break
    stats.append(stat)
    
    min_costs = np.append(min_costs, min([get_cost(s) for s in stats_sim]))
    all_chromos = np.append(all_chromos, [chromos], axis=0)
    
    min_cost = min_costs[-1]
    if not is_manual and i>=early_stopping_steps         and all(min_cost==min_costs[- early_stopping_steps:]):
        print('early stops at',i)
        print('result', min_cost)
        break
stats = np.array(stats)


# In[ ]:


plot_data = np.array(min_costs)
plot_data[np.isinf(plot_data)]=plot_data[~np.isinf(plot_data)].max()*1.2 #Replace inf with finite numbers

plt.plot(plot_data)
plt.title('Минимальная найденная стоимость')
plt.ylabel('Минимальное расстояние')
plt.xlabel('Итерация')
#plt.yscale('log')
plt.show()


# In[ ]:


best_chromo_idx = np.argmin([get_cost(s) for s in stats_sim])
best_chromo = chromos[best_chromo_idx]
best_stat = stats_sim[best_chromo_idx]


# In[ ]:


client_ds, op_ds, st = sm.run_simulation(best_chromo, n_lines=n_lines, n_vip_lines=n_vip_lines, verb=True, only_stat=False)


# In[ ]:


op_time_ds = pd.DataFrame(best_chromo.reshape(5,3), index=range(7,12), columns=['gold','silver','regular'])
op_time_ds


# In[ ]:


sm.plot_clients_no_lines(client_ds)


# In[ ]:


sm.plot_clients_waitings(client_ds)


# In[ ]:


sm.plot_clients_success(client_ds)


# # Эксперимент 3

# In[ ]:


early_stopping_steps = 30
is_manual = False
MUTATION_P = 0.3
n_lines = 50
n_vip_lines = 0


# In[ ]:


chromos = np.array([generate_gene() for i in range(N)])
all_chromos = np.array([chromos])
stats_sim = np.array([sm.run_simulation(c, n_lines=n_lines, n_vip_lines=n_vip_lines, verb=False, only_stat=True) for c in chromos])
min_costs = np.array([min([get_cost(stat) for stat in stats_sim])])
stats = []


# In[ ]:


for i in tqdm_notebook(range(100)):
    try:
        chromos, stat, stats_sim = ga_step(chromos, is_manual=is_manual, n_lines=n_lines, n_vip_lines=n_vip_lines)
    except StopIteration:
        print("Stopped at",i)
        break
    stats.append(stat)
    
    min_costs = np.append(min_costs, min([get_cost(s) for s in stats_sim]))
    all_chromos = np.append(all_chromos, [chromos], axis=0)
    
    min_cost = min_costs[-1]
    if not is_manual and i>=early_stopping_steps         and all(min_cost==min_costs[- early_stopping_steps:]):
        print('early stops at',i)
        print('result', min_cost)
        break
stats = np.array(stats)


# In[ ]:


plot_data = np.array(min_costs)
plot_data[np.isinf(plot_data)]=plot_data[~np.isinf(plot_data)].max()*1.2 #Replace inf with finite numbers

plt.plot(plot_data)
plt.title('Минимальная найденная стоимость')
plt.ylabel('Минимальное расстояние')
plt.xlabel('Итерация')
#plt.yscale('log')
plt.show()


# In[ ]:


best_chromo_idx = np.argmin([get_cost(s) for s in stats_sim])
best_chromo = chromos[best_chromo_idx]
best_stat = stats_sim[best_chromo_idx]


# In[ ]:


client_ds, op_ds, st = sm.run_simulation(best_chromo, n_lines=n_lines, n_vip_lines=n_vip_lines, verb=True, only_stat=False)


# In[ ]:


op_time_ds = pd.DataFrame(best_chromo.reshape(5,3), index=range(7,12), columns=['gold','silver','regular'])
op_time_ds


# In[ ]:


sm.plot_clients_no_lines(client_ds)


# In[ ]:


sm.plot_clients_waitings(client_ds)


# In[ ]:


sm.plot_clients_success(client_ds)


# # Эксперимент 4

# In[ ]:


early_stopping_steps = 30
is_manual = False
MUTATION_P = 0.3
n_lines = 55
n_vip_lines = 0


# In[ ]:


chromos = np.array([generate_gene() for i in range(N)])
all_chromos = np.array([chromos])
stats_sim = np.array([sm.run_simulation(c, n_lines=n_lines, n_vip_lines=n_vip_lines, verb=False, only_stat=True) for c in chromos])
min_costs = np.array([min([get_cost(stat) for stat in stats_sim])])
stats = []


# In[ ]:


for i in tqdm_notebook(range(100)):
    try:
        chromos, stat, stats_sim = ga_step(chromos, is_manual=is_manual, n_lines=n_lines, n_vip_lines=n_vip_lines)
    except StopIteration:
        print("Stopped at",i)
        break
    stats.append(stat)
    
    min_costs = np.append(min_costs, min([get_cost(s) for s in stats_sim]))
    all_chromos = np.append(all_chromos, [chromos], axis=0)
    
    min_cost = min_costs[-1]
    if not is_manual and i>=early_stopping_steps         and all(min_cost==min_costs[- early_stopping_steps:]):
        print('early stops at',i)
        print('result', min_cost)
        break
stats = np.array(stats)


# In[ ]:


plot_data = np.array(min_costs)
plot_data[np.isinf(plot_data)]=plot_data[~np.isinf(plot_data)].max()*1.2 #Replace inf with finite numbers

plt.plot(plot_data)
plt.title('Минимальная найденная стоимость')
plt.ylabel('Минимальное расстояние')
plt.xlabel('Итерация')
#plt.yscale('log')
plt.show()


# In[ ]:


best_chromo_idx = np.argmin([get_cost(s) for s in stats_sim])
best_chromo = chromos[best_chromo_idx]
best_stat = stats_sim[best_chromo_idx]


# In[ ]:


client_ds, op_ds, st = sm.run_simulation(best_chromo, n_lines=n_lines, n_vip_lines=n_vip_lines, verb=True, only_stat=False)


# In[ ]:


op_time_ds = pd.DataFrame(best_chromo.reshape(5,3), index=range(7,12), columns=['gold','silver','regular'])
op_time_ds


# In[ ]:


sm.plot_clients_no_lines(client_ds)


# In[ ]:


sm.plot_clients_waitings(client_ds)


# In[ ]:


sm.plot_clients_success(client_ds)


# # Эксперимент 5

# In[ ]:


early_stopping_steps = 30
is_manual = False
MUTATION_P = 0.3
n_lines = 55
n_vip_lines = 5


# In[ ]:


chromos = np.array([generate_gene() for i in range(N)])
all_chromos = np.array([chromos])
stats_sim = np.array([sm.run_simulation(c, n_lines=n_lines, n_vip_lines=n_vip_lines, verb=False, only_stat=True) for c in chromos])
min_costs = np.array([min([get_cost(stat) for stat in stats_sim])])
stats = []


# In[ ]:


for i in tqdm_notebook(range(100)):
    try:
        chromos, stat, stats_sim = ga_step(chromos, is_manual=is_manual, n_lines=n_lines, n_vip_lines=n_vip_lines)
    except StopIteration:
        print("Stopped at",i)
        break
    stats.append(stat)
    
    min_costs = np.append(min_costs, min([get_cost(s) for s in stats_sim]))
    all_chromos = np.append(all_chromos, [chromos], axis=0)
    
    min_cost = min_costs[-1]
    if not is_manual and i>=early_stopping_steps         and all(min_cost==min_costs[- early_stopping_steps:]):
        print('early stops at',i)
        print('result', min_cost)
        break
stats = np.array(stats)


# In[ ]:


plot_data = np.array(min_costs)
plot_data[np.isinf(plot_data)]=plot_data[~np.isinf(plot_data)].max()*1.2 #Replace inf with finite numbers

plt.plot(plot_data)
plt.title('Минимальная найденная стоимость')
plt.ylabel('Минимальное расстояние')
plt.xlabel('Итерация')
#plt.yscale('log')
plt.show()


# In[ ]:


best_chromo_idx = np.argmin([get_cost(s) for s in stats_sim])
best_chromo = chromos[best_chromo_idx]
best_stat = stats_sim[best_chromo_idx]


# In[ ]:


client_ds, op_ds, st = sm.run_simulation(best_chromo, n_lines=n_lines, n_vip_lines=n_vip_lines, verb=True, only_stat=False)


# In[ ]:


op_time_ds = pd.DataFrame(best_chromo.reshape(5,3), index=range(7,12), columns=['gold','silver','regular'])
op_time_ds


# In[ ]:


sm.plot_clients_no_lines(client_ds)


# In[ ]:


sm.plot_clients_waitings(client_ds)


# In[ ]:


sm.plot_clients_success(client_ds)


# # Эксперимент 6

# In[ ]:


early_stopping_steps = 30
is_manual = False
MUTATION_P = 0.3
n_lines = 50
n_vip_lines = 10


# In[ ]:


chromos = np.array([generate_gene() for i in range(N)])
all_chromos = np.array([chromos])
stats_sim = np.array([sm.run_simulation(c, n_lines=n_lines, n_vip_lines=n_vip_lines, verb=False, only_stat=True) for c in chromos])
min_costs = np.array([min([get_cost(stat) for stat in stats_sim])])
stats = []


# In[ ]:


for i in tqdm_notebook(range(100)):
    try:
        chromos, stat, stats_sim = ga_step(chromos, is_manual=is_manual, n_lines=n_lines, n_vip_lines=n_vip_lines)
    except StopIteration:
        print("Stopped at",i)
        break
    stats.append(stat)
    
    min_costs = np.append(min_costs, min([get_cost(s) for s in stats_sim]))
    all_chromos = np.append(all_chromos, [chromos], axis=0)
    
    min_cost = min_costs[-1]
    if not is_manual and i>=early_stopping_steps         and all(min_cost==min_costs[- early_stopping_steps:]):
        print('early stops at',i)
        print('result', min_cost)
        break
stats = np.array(stats)


# In[ ]:


plot_data = np.array(min_costs)
plot_data[np.isinf(plot_data)]=plot_data[~np.isinf(plot_data)].max()*1.2 #Replace inf with finite numbers

plt.plot(plot_data)
plt.title('Минимальная найденная стоимость')
plt.ylabel('Минимальное расстояние')
plt.xlabel('Итерация')
#plt.yscale('log')
plt.show()


# In[ ]:


best_chromo_idx = np.argmin([get_cost(s) for s in stats_sim])
best_chromo = chromos[best_chromo_idx]
best_stat = stats_sim[best_chromo_idx]


# In[ ]:


client_ds, op_ds, st = sm.run_simulation(best_chromo, n_lines=n_lines, n_vip_lines=n_vip_lines, verb=True, only_stat=False)


# In[ ]:


op_time_ds = pd.DataFrame(best_chromo.reshape(5,3), index=range(7,12), columns=['gold','silver','regular'])
op_time_ds


# In[ ]:


sm.plot_clients_no_lines(client_ds)


# In[ ]:


sm.plot_clients_waitings(client_ds)


# In[ ]:


sm.plot_clients_success(client_ds)


# # Эксперимент 7

# In[ ]:


early_stopping_steps = 30
is_manual = False
MUTATION_P = 0.3
n_lines = 55
n_vip_lines = 10


# In[ ]:


chromos = np.array([generate_gene() for i in range(N)])
all_chromos = np.array([chromos])
stats_sim = np.array([sm.run_simulation(c, n_lines=n_lines, n_vip_lines=n_vip_lines, verb=False, only_stat=True) for c in chromos])
min_costs = np.array([min([get_cost(stat) for stat in stats_sim])])
stats = []


# In[ ]:


for i in tqdm_notebook(range(100)):
    try:
        chromos, stat, stats_sim = ga_step(chromos, is_manual=is_manual, n_lines=n_lines, n_vip_lines=n_vip_lines)
    except StopIteration:
        print("Stopped at",i)
        break
    stats.append(stat)
    
    min_costs = np.append(min_costs, min([get_cost(s) for s in stats_sim]))
    all_chromos = np.append(all_chromos, [chromos], axis=0)
    
    min_cost = min_costs[-1]
    if not is_manual and i>=early_stopping_steps         and all(min_cost==min_costs[- early_stopping_steps:]):
        print('early stops at',i)
        print('result', min_cost)
        break
stats = np.array(stats)


# In[ ]:


plot_data = np.array(min_costs)
plot_data[np.isinf(plot_data)]=plot_data[~np.isinf(plot_data)].max()*1.2 #Replace inf with finite numbers

plt.plot(plot_data)
plt.title('Минимальная найденная стоимость')
plt.ylabel('Минимальное расстояние')
plt.xlabel('Итерация')
#plt.yscale('log')
plt.show()


# In[ ]:


best_chromo_idx = np.argmin([get_cost(s) for s in stats_sim])
best_chromo = chromos[best_chromo_idx]
best_stat = stats_sim[best_chromo_idx]


# In[ ]:


client_ds, op_ds, st = sm.run_simulation(best_chromo, n_lines=n_lines, n_vip_lines=n_vip_lines, verb=True, only_stat=False)


# In[ ]:


op_time_ds = pd.DataFrame(best_chromo.reshape(5,3), index=range(7,12), columns=['gold','silver','regular'])
op_time_ds


# In[ ]:


sm.plot_clients_no_lines(client_ds)


# In[ ]:


sm.plot_clients_waitings(client_ds)


# In[ ]:


sm.plot_clients_success(client_ds)


# # Эксперимент 8

# In[ ]:


early_stopping_steps = 30
is_manual = False
MUTATION_P = 0.3
n_lines = 60
n_vip_lines = 0


# In[ ]:


chromos = np.array([generate_gene() for i in range(N)])
all_chromos = np.array([chromos])
stats_sim = np.array([sm.run_simulation(c, n_lines=n_lines, n_vip_lines=n_vip_lines, verb=False, only_stat=True) for c in chromos])
min_costs = np.array([min([get_cost(stat) for stat in stats_sim])])
stats = []


# In[ ]:


for i in tqdm_notebook(range(100)):
    try:
        chromos, stat, stats_sim = ga_step(chromos, is_manual=is_manual, n_lines=n_lines, n_vip_lines=n_vip_lines)
    except StopIteration:
        print("Stopped at",i)
        break
    stats.append(stat)
    
    min_costs = np.append(min_costs, min([get_cost(s) for s in stats_sim]))
    all_chromos = np.append(all_chromos, [chromos], axis=0)
    
    min_cost = min_costs[-1]
    if not is_manual and i>=early_stopping_steps         and all(min_cost==min_costs[- early_stopping_steps:]):
        print('early stops at',i)
        print('result', min_cost)
        break
stats = np.array(stats)


# In[ ]:


plot_data = np.array(min_costs)
plot_data[np.isinf(plot_data)]=plot_data[~np.isinf(plot_data)].max()*1.2 #Replace inf with finite numbers

plt.plot(plot_data)
plt.title('Минимальная найденная стоимость')
plt.ylabel('Минимальное расстояние')
plt.xlabel('Итерация')
#plt.yscale('log')
plt.show()


# In[ ]:


best_chromo_idx = np.argmin([get_cost(s) for s in stats_sim])
best_chromo = chromos[best_chromo_idx]
best_stat = stats_sim[best_chromo_idx]


# In[ ]:


client_ds, op_ds, st = sm.run_simulation(best_chromo, n_lines=n_lines, n_vip_lines=n_vip_lines, verb=True, only_stat=False)


# In[ ]:


op_time_ds = pd.DataFrame(best_chromo.reshape(5,3), index=range(7,12), columns=['gold','silver','regular'])
op_time_ds


# In[ ]:


sm.plot_clients_no_lines(client_ds)


# In[ ]:


sm.plot_clients_waitings(client_ds)


# In[ ]:


sm.plot_clients_success(client_ds)


# # Эксперимент 9

# In[ ]:


early_stopping_steps = 30
is_manual = False
MUTATION_P = 0.3
n_lines = 60
n_vip_lines = 5


# In[ ]:


chromos = np.array([generate_gene() for i in range(N)])
all_chromos = np.array([chromos])
stats_sim = np.array([sm.run_simulation(c, n_lines=n_lines, n_vip_lines=n_vip_lines, verb=False, only_stat=True) for c in chromos])
min_costs = np.array([min([get_cost(stat) for stat in stats_sim])])
stats = []


# In[ ]:


for i in tqdm_notebook(range(100)):
    try:
        chromos, stat, stats_sim = ga_step(chromos, is_manual=is_manual, n_lines=n_lines, n_vip_lines=n_vip_lines)
    except StopIteration:
        print("Stopped at",i)
        break
    stats.append(stat)
    
    min_costs = np.append(min_costs, min([get_cost(s) for s in stats_sim]))
    all_chromos = np.append(all_chromos, [chromos], axis=0)
    
    min_cost = min_costs[-1]
    if not is_manual and i>=early_stopping_steps         and all(min_cost==min_costs[- early_stopping_steps:]):
        print('early stops at',i)
        print('result', min_cost)
        break
stats = np.array(stats)


# In[ ]:


plot_data = np.array(min_costs)
plot_data[np.isinf(plot_data)]=plot_data[~np.isinf(plot_data)].max()*1.2 #Replace inf with finite numbers

plt.plot(plot_data)
plt.title('Минимальная найденная стоимость')
plt.ylabel('Минимальное расстояние')
plt.xlabel('Итерация')
#plt.yscale('log')
plt.show()


# In[ ]:


best_chromo_idx = np.argmin([get_cost(s) for s in stats_sim])
best_chromo = chromos[best_chromo_idx]
best_stat = stats_sim[best_chromo_idx]


# In[ ]:


client_ds, op_ds, st = sm.run_simulation(best_chromo, n_lines=n_lines, n_vip_lines=n_vip_lines, verb=True, only_stat=False)


# In[ ]:


op_time_ds = pd.DataFrame(best_chromo.reshape(5,3), index=range(7,12), columns=['gold','silver','regular'])
op_time_ds


# In[ ]:


sm.plot_clients_no_lines(client_ds)


# In[ ]:


sm.plot_clients_waitings(client_ds)


# In[ ]:


sm.plot_clients_success(client_ds)


# # Эксперимент 10

# In[ ]:


early_stopping_steps = 30
is_manual = False
MUTATION_P = 0.3
n_lines = 60
n_vip_lines = 10


# In[ ]:


chromos = np.array([generate_gene() for i in range(N)])
all_chromos = np.array([chromos])
stats_sim = np.array([sm.run_simulation(c, n_lines=n_lines, n_vip_lines=n_vip_lines, verb=False, only_stat=True) for c in chromos])
min_costs = np.array([min([get_cost(stat) for stat in stats_sim])])
stats = []


# In[ ]:


for i in tqdm_notebook(range(100)):
    try:
        chromos, stat, stats_sim = ga_step(chromos, is_manual=is_manual, n_lines=n_lines, n_vip_lines=n_vip_lines)
    except StopIteration:
        print("Stopped at",i)
        break
    stats.append(stat)
    
    min_costs = np.append(min_costs, min([get_cost(s) for s in stats_sim]))
    all_chromos = np.append(all_chromos, [chromos], axis=0)
    
    min_cost = min_costs[-1]
    if not is_manual and i>=early_stopping_steps         and all(min_cost==min_costs[- early_stopping_steps:]):
        print('early stops at',i)
        print('result', min_cost)
        break
stats = np.array(stats)


# In[ ]:


plot_data = np.array(min_costs)
plot_data[np.isinf(plot_data)]=plot_data[~np.isinf(plot_data)].max()*1.2 #Replace inf with finite numbers

plt.plot(plot_data)
plt.title('Минимальная найденная стоимость')
plt.ylabel('Минимальное расстояние')
plt.xlabel('Итерация')
#plt.yscale('log')
plt.show()


# In[ ]:


best_chromo_idx = np.argmin([get_cost(s) for s in stats_sim])
best_chromo = chromos[best_chromo_idx]
best_stat = stats_sim[best_chromo_idx]


# In[ ]:


client_ds, op_ds, st = sm.run_simulation(best_chromo, n_lines=n_lines, n_vip_lines=n_vip_lines, verb=True, only_stat=False)


# In[ ]:


op_time_ds = pd.DataFrame(best_chromo.reshape(5,3), index=range(7,12), columns=['gold','silver','regular'])
op_time_ds


# In[ ]:


sm.plot_clients_no_lines(client_ds)


# In[ ]:


sm.plot_clients_waitings(client_ds)


# In[ ]:


sm.plot_clients_success(client_ds)

