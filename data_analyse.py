import os
import numpy as np
import pickle as pkl


data_name = 'WN18RR_new'
data_dir = os.path.join('datasets', 'data', data_name)

train_data_dir = os.path.join(data_dir, 'train.pickle')
valid_data_dir = os.path.join(data_dir, 'valid.pickle')
test_data_dir = os.path.join(data_dir, 'test.pickle')

with open(train_data_dir, "rb") as in_file:
        train_examples = pkl.load(in_file)

with open(valid_data_dir, "rb") as in_file:
        valid_examples = pkl.load(in_file)

with open(test_data_dir, "rb") as in_file:
        test_examples = pkl.load(in_file)
    
class graph:
    def __init__(self, data, max_hop) -> None:
        self.data = data
        self.max_hop = max_hop
        self.nodes = set()
        self.rels = set()

        # {ent1: {relA: [ent10, ent11...]}}
        self.paths = {} 

        # {ent1: [ent10, ent11, ...]}, i.e. multi-relations are not considered
        self.connects = {}
        
        self.relNum = 1000

        self.addElement()
        self.nodeNum = len(self.nodes)
        self.relNum = len(self.rels)

        self.addPaths()
        self.addConnectivity()
        self.getCompositionPattern(max_hop)
      
    
    def addElement(self):
        for tri in self.data:         
            h,r,t = tri
            if h not in self.nodes:
               self.nodes.add(h)
            if t not in self.nodes:
               self.nodes.add(t)
            if r not in self.rels:
               self.rels.add(r)

    def addPaths(self):
        for tri in self.data:
            h,r,t = tri
            if h not in self.paths:
                self.paths[h] = {r:[t]}
            else:
                if r not in self.paths[h]:
                    self.paths[h][r] = [t]
                else:
                    self.paths[h][r].append(t)
            rev_r = r + self.relNum
            if t not in self.paths:
                self.paths[t] = {rev_r:[h]}
            else:
                if rev_r not in self.paths[t]:
                    self.paths[t][rev_r] = [h]
                else:
                    self.paths[t][rev_r].append(h)

    def addConnectivity(self):
        for tri in self.data:
            h,r,t = tri
            if h not in self.connects:
                self.connects[h] = [t]
            else:
                self.connects[h].append(t)
            if t not in self.connects:
                self.connects[t] = [h]
            else:
                self.connects[t].append(h)
            



    def findConnectivity(self, head, tail, hop):
        if hop > 0:
            candidates = self.connects[head]
            if tail in candidates:
                return True
            for ent in candidates:
                if self.findConnectivity(ent, tail, hop-1):
                    return True
        return False
    
    def getCompositionPattern(self, hop):
        self.compos_pattern_full = {}
        # self.compos_pattern = {}
        # {rel1: {[rel2,rel3]: [[full path1], [full path2]..], [rel2, rel4],...]: [[full path1]] }, rel2: {}}

        for tri in self.data:
            h0,r0,t0 = tri
            if r0 not in self.compos_pattern_full:
                self.compos_pattern_full[r0] = {}
            collect_rel_paths = []
            tmp_rel_path = ''
            tmp_ent_in_path = []
            tmp_full_paths = []
            self.compositionPath(collect_rel_paths, tmp_rel_path, tmp_ent_in_path, tmp_full_paths, h0, t0, hop)

            for rel_path, full_path in zip(collect_rel_paths, tmp_full_paths):
                if rel_path not in self.compos_pattern_full[r0]:
                    self.compos_pattern_full[r0][rel_path] = [full_path]
                else:
                    self.compos_pattern_full[r0][rel_path].append(full_path)
        

    def compositionPath(self, collect_rel_paths, tmp_rel_path, tmp_ent_in_path, tmp_full_paths, head, tail, hop):
        # print(tmp_ent_in_path)
        # print(type(tmp_ent_in_path))
        if hop < 1:
            return 
        tmp_ent_in_path.append(head)
        for rel in self.paths[head]:
            for ent in self.paths[head][rel]:
                if ent == tail:   
                    # rel_path = tmp_rel_path.copy()
                    # rel_path.append(rel)
                    rel_path = tmp_rel_path +'-'+str(rel)
                    collect_rel_paths.append(rel_path)
                    rel_list = [int(i) for i in rel_path.split('-')[1:]]
                    tmp_full_path = [val for pair in zip(tmp_ent_in_path, rel_list) for val in pair]
                    tmp_full_path.append(tail)
                    tmp_full_paths.append(tmp_full_path)
                else:    
                    if ent not in tmp_ent_in_path:
                    # new_tmp_rel_path = tmp_rel_path.copy()
                    # new_tmp_rel_path.append(rel)
                        new_tmp_rel_path = tmp_rel_path +'-'+str(rel)
                        self.compositionPath(collect_rel_paths, new_tmp_rel_path, tmp_ent_in_path, tmp_full_paths, ent, tail, hop-1)
                    
            
        return
    
    def searchComposPattern(self, query, hop = 6, threshold = 5):
        h,r,t = query
        try:
            rel_paths = [pstr for pstr in self.compos_pattern_full[r].keys() if len(self.compos_pattern_full[r][pstr])>=threshold]
        except:
            print(len(self.compos_pattern_full[r]))
        for rp in rel_paths:
            rel_list = [int(i) for i in rp.split('-')[1:]]
            if len(rel_list)<=hop and self.checkComposPatthern(rel_list, h, t, 0):
                return True
        return False

            
    def checkComposPatthern(self, rel_path, head, tail, i):
        if i >= len(rel_path):
            return False
        if rel_path[i] not in self.paths[head]:
            return False
        if tail in self.paths[head][rel_path[i]]:
            return True
        for ent in self.paths[head][rel_path[i]]:
            return self.checkComposPatthern(rel_path, ent, tail, i+1)
        
    def searchConnectivity(self, query, threshold = 5):
        h,r,t = query
        return self.findConnectivity(h,t,threshold)
    
    def getComposPattern(self):
        cp = {}
        for rel in self.compos_pattern_full:
            cp[rel] = {}
            for rel_path in self.compos_pattern_full[rel]:
                cp[rel][rel_path] = len(self.compos_pattern_full[rel][rel_path])
        return cp
    
    def getComposPatternFull(self):
        return self.compos_pattern_full
        

g = graph(train_examples, max_hop=2)


stats_out = data_name + '_graph.pkl'

with open(os.path.join(data_dir, stats_out), 'wb') as file:
    pkl.dump(g, file) 


# WN18RR: 24; FB15k-237: 81; YAGO3-10: 321
pattern_freq_threshold = 24
print('pattern_freq_threshold: ', pattern_freq_threshold)
compos_pattern_all = g.getComposPatternFull()

compos_rel_set = set()
compos_tri = {}
compos_num = 0
for r3 in compos_pattern_all:
    paths = compos_pattern_all[r3]
    for path in paths:
        if len(path.split('-'))==3:
            if r3 not in compos_tri:
                compos_tri[r3] = {}
            cp_tri=set()
            for p in paths[path]:
                a, r1, b, r2, c = p
                tri1 = (a, r1, b)
                tri2 = (b, r2, c)
                tri3 = (a, r3, c)
                cp_tri.add(tri1)
                cp_tri.add(tri2)
                cp_tri.add(tri3)
            compos_tri[r3][path] = len(cp_tri)


total_cp_tri = 0
rel_tri = {}
for r3 in compos_tri:
    for path in compos_tri[r3]:
        if compos_tri[r3][path]>=pattern_freq_threshold:
            compos_num += 1
            total_cp_tri += compos_tri[r3][path]
            compos_rel_set.add(r3)
            r1, r2 = [int(r) for r in path.split('-')[1:3]]
            compos_rel_set.add(r1)
            compos_rel_set.add(r2)
            if r3 not in rel_tri:
                rel_tri[r3] = compos_tri[r3][path]

print('compos rel #: ', len(compos_rel_set)//2)
print('compos_num: ', compos_num)
print('avg. compos tri: ', total_cp_tri/compos_num)

cp_rel_tri = 0
for tri in train_examples:
    h,r,t = tri
    if r in compos_rel_set:
        cp_rel_tri+= 1

print('cp_rel_tri: ', cp_rel_tri)


