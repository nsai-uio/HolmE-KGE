import argparse
import json
import os
import pickle as pkl
import numpy as np
import collections


from kg_dataset import KGDataset


data_dir = os.path.join('data', 'FB237')

train_data_dir = os.path.join(data_dir, 'train.pickle')
valid_data_dir = os.path.join(data_dir, 'valid.pickle')
test_data_dir = os.path.join(data_dir, 'test.pickle')

with open(train_data_dir, "rb") as in_file:
        train_examples = pkl.load(in_file)

with open(valid_data_dir, "rb") as in_file:
        valid_examples = pkl.load(in_file)

with open(test_data_dir, "rb") as in_file:
        test_examples = pkl.load(in_file)

data_size = np.max(train_examples, axis= 0)
n_entity = int(max(data_size[0], data_size[2]) + 1)
n_relation = int(data_size[1] + 1)

print('#training triples: ', len(train_examples))
print('#valid triples: ', len(valid_examples))
print('#test triples: ', len(test_examples))
print('#entities: ', n_entity)
print('#relations: ', n_relation)

train_facts = {} # each query (h,r) has a list of tails [t]: {(h,r): [t]}
ent_rel = {} # each entity has a set of rel: {ent: {set of rel}}
ent_ent = {} # each connecting entity pair (ordered) has a rel: {(ent1, ent2): rel}
for triple in train_examples:
    head, rel, tail = triple
    if (head, rel) not in train_facts:
        train_facts[(head, rel)] = [tail]
    else:
        train_facts[(head, rel)].append(tail)

    if (tail, rel+n_relation) not in train_facts:
        train_facts[(tail, rel+n_relation)] = [head]
    else:
        train_facts[(tail, rel+n_relation)].append(head)

    if head not in ent_rel:
        ent_rel[head] = {rel}
    else:
        ent_rel[head].add(rel)
    if tail not in ent_rel:
        ent_rel[tail] = {rel+n_relation}
    else:
        ent_rel[tail].add(rel+n_relation)

    ent_ent[(head, tail)] = rel


# load graph
graph_name = 'WN18RR_graph.pkl'
with open(os.path.join(data_dir,  graph_name), "rb") as in_file:
        g = pkl.load(in_file)

compos_pattern_all = g.getComposPattern()

compos_combination_list = ['10:-191-428', '68:-188-310', '12:-9-4']
h2_dataname = 'compos_h2_1'
h2_data_file = os.path.join('data', h2_dataname)
if not os.path.exists(h2_data_file):
     os.makedirs(h2_data_file)


new_train = set()
new_test = set()
compos_combination = compos_combination_list[0]
r1=191
r2=428
r3=10
length = len(compos_pattern_all[compos_combination])
r3_num = 0.1 * length
r1_tr_freq=0
r3_tr_freq=0
r3_tst_freq=0

relations = {r1,r2,r3}
entities, relations = set(), set()

for path in compos_pattern_all[compos_combination]:
    a, r1, b, r2, c = path
    tri1 = (a, r1, b)
    tri2 = (b, r2, c)
    tri3 = (a, r3, c)
    new_train.add(tri1)
    new_train.add(tri2)
    r1_tr_freq += 1
    if r3_num >0:
        new_train.append(tri3)
        r3_tr_freq += 1
        r3_num -= 1
    else:
        r3_tst_freq+=1
        new_test.append(tri3)
    entities.add(a)
    entities.add(b)
    entities.add(c)
    
ent2idx = {x: i for (i, x) in enumerate(sorted(entities))}
rel2idx = {x: i for (i, x) in enumerate(sorted(relations))}

print('new train: ', len(new_train))

examples = []
for tri in new_train:
     h,r,t=tri
     examples.append([h,r,t])
with open(os.path.join(h2_data_file, 'train.pickle'), 'wb') as file:
    pkl.dump(examples, file) 

data_all = examples

examples = []
for tri in new_test:
     h,r,t=tri
     examples.append([h,r,t])
data_all += examples
valid_examples = examples[:len(examples)//2]
test_examples = examples[len(examples)//2:]
with open(os.path.join(h2_data_file, 'test.pickle'), 'wb') as file:
    pkl.dump(test_examples, file) 
with open(os.path.join(h2_data_file, 'valid.pickle'), 'wb') as file:
    pkl.dump(valid_examples, file) 
    
print('new test: ', len(test_examples))
print(r1_tr_freq, r3_tr_freq, r3_tst_freq)



n_relations = len(rel2idx)
lhs_filters = collections.defaultdict(set)
rhs_filters = collections.defaultdict(set)
for tri in data_all:
    h,r,t=tri
    [h,r,t] = [ent2idx[h],  rel2idx[r], ent2idx[t]]
    rhs_filters[(h,r)].add(t)
    lhs_filters[(t, r + n_relations)].add(h)
lhs_final = {}
rhs_final = {}
for k, v in lhs_filters.items():
    lhs_final[k] = sorted(list(v))
for k, v in rhs_filters.items():
    rhs_final[k] = sorted(list(v))

dataset_filters = {"lhs": lhs_final, "rhs": rhs_final}

with open(os.path.join(h2_data_file, "to_skip.pickle"), "wb") as save_file:
    pkl.dump(dataset_filters, save_file)