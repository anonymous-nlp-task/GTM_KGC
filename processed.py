import pickle
import numpy as np
import networkx as nx
import os
from helper import find_2_hop_paths

class processed():
    def __init__(self):
        # ICEWS14, FB15k-237, FB15k-237N,  NELL, WN18RR
        self.dataset = "WN18RR"
        if self.dataset == "ICEWS14":
            self.dataset_dir = "./data/original_data/" + self.dataset
            self.train_data = self.load_quadruples(self.dataset_dir, 'train')
            self.test_data = self.load_quadruples(self.dataset_dir, 'test')
            self.dev_data = self.load_quadruples(self.dataset_dir, 'valid')
            all_data = self.train_data + self.test_data + self.dev_data
            all_data = np.array(all_data)
            head_entity = all_data[:, 0]
            self.relation = np.unique(all_data[:, 1])
            tail_entity = all_data[:, 2]
            self.time = np.unique(all_data[:, 3])
            entity = np.concatenate((head_entity, tail_entity), axis=0)
            self.entity = np.unique(entity)

        self.data_path = "./data/processed/" + self.dataset + "/"
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)


    def load_quadruples(self, inPath, fileName):
        with open(os.path.join(inPath, fileName), 'r', encoding='utf-8') as fr:
            quadrupleList = []
            for id, line in enumerate(fr):
                line = line.replace('\n', '')
                line_split = line.split('\t')
                head = line_split[0]
                rel = line_split[1]
                tail = line_split[2]
                time = line_split[3]
                quadrupleList.append([head, rel, tail, time])

        return quadrupleList

    def load_factruples(self, inPath, fileName):
        with open(os.path.join(inPath, fileName), 'r') as fr:
            quadrupleList = []
            for id, line in enumerate(fr):
                if id == 0:
                    continue
                line_split = line.split()
                head = int(line_split[0])
                rel = int(line_split[2])
                tail = int(line_split[1])
                if self.dataset == "ICEWS14":
                    time = int(line_split[3])
                    quadrupleList.append([head, rel, tail, time])
                else:
                    quadrupleList.append([head, rel, tail])

        return np.array(quadrupleList)

    def generate_tkg_processed_data(self):
        entity_num = len(self.entity)
        entity_id = {}
        file = self.data_path + "entity2id.txt"
        f = open(file, 'w', encoding='utf-8')
        f.write(str(entity_num) + '\n')
        for i, e in enumerate(self.entity):
            f.write(str(e) + '\t' + str(i) + '\n')
            entity_id[e] = i
        f.close()

        file = self.data_path + "entityid2name.txt"
        f = open(file, 'w', encoding='utf-8')
        f.write(str(entity_num) + '\n')
        for i, e in enumerate(self.entity):
            f.write(str(i) + '\t' + str(e) + '\n')
        f.close()

        relation_id = {}
        relation_num = len(self.relation)
        file = self.data_path + "relation2id.txt"
        f = open(file, 'w', encoding='utf-8')
        f.write(str(relation_num) + '\n')
        for i, r in enumerate(self.relation):
            f.write(str(r) + '\t' + str(i) + '\n')
            relation_id[r] = i
        f.close()

        file = self.data_path + "relationid2name.txt"
        f = open(file, 'w', encoding='utf-8')
        f.write(str(relation_num) + '\n')
        for i, r in enumerate(self.relation):
            f.write(str(i) + '\t' + str(r) + '\n')
        f.close()

        time_id = {}
        time_num = len(self.time)
        file = self.data_path + "time2id.txt"
        f = open(file, 'w', encoding='utf-8')
        f.write(str(time_num) + '\n')
        for i, t in enumerate(self.time):
            f.write(str(t) + '\t' + str(i) + '\n')
            time_id[t] = i
        f.close()

        file = self.data_path + "timeid2name.txt"
        f = open(file, 'w', encoding='utf-8')
        f.write(str(time_num) + '\n')
        for i, t in enumerate(self.time):
            f.write(str(i) + '\t' + str(t) + '\n')
        f.close()


        trd_num = len(self.train_data)
        file = self.data_path + "train2id.txt"
        f = open(file, 'w', encoding='utf-8')
        f.write(str(trd_num) + '\n')
        for [h, r, o, t] in self.train_data:
            h_id = entity_id[h]
            r_id = relation_id[r]
            o_id = entity_id[o]
            t_id = time_id[t]
            f.write(str(h_id) + ' ' + str(o_id) + ' ' + str(r_id) + ' ' + str(t_id) + '\n')
        f.close()

        ddnum = len(self.dev_data)
        file = self.data_path + "valid2id.txt"
        f = open(file, 'w', encoding='utf-8')
        f.write(str(ddnum) + '\n')
        for [h, r, o, t] in self.dev_data:
            h_id = entity_id[h]
            r_id = relation_id[r]
            o_id = entity_id[o]
            t_id = time_id[t]
            f.write(str(h_id) + ' ' + str(o_id) + ' ' + str(r_id) + ' ' + str(t_id) + '\n')
        f.close()

        tednum = len(self.test_data)
        file = self.data_path + "test2id.txt"
        f = open(file, 'w', encoding='utf-8')
        f.write(str(tednum) + '\n')
        for [h, r, o, t] in self.test_data:
            h_id = entity_id[h]
            r_id = relation_id[r]
            o_id = entity_id[o]
            t_id = time_id[t]
            f.write(str(h_id) + ' ' + str(o_id) + ' ' + str(r_id) + ' ' + str(t_id) + '\n')
        f.close()


    def generate_kg_processed_data(self):
        '''
        1. entity2id.txt entityid2name.txt
        '''
        self.dataset_dir = "./data/original_data/" + self.dataset
        self.entity = []
        if self.dataset == 'NELL':
            with open(os.path.join(self.dataset_dir, 'entity2id.txt'), 'r', encoding='utf-8') as fr:
                for id, line in enumerate(fr):
                    if id == 0:
                        continue
                    line = line.replace('\n', '')
                    line_split = line.split('\t')[0]
                    self.entity.append(line_split)
        elif self.dataset == 'WN18RR':
            with open(os.path.join(self.dataset_dir, 'entityid2name.txt'), 'r', encoding='utf-8') as fr:
                for id, line in enumerate(fr):
                    if id == 0:
                        continue
                    line = line.replace('\n', '')
                    line_split = line.split('\t')[1]
                    line_split = line_split.replace(' , ', ' ')
                    self.entity.append(line_split)
        else:  # FB15k-237 (entityid2name)
            entity2id = []
            with open(os.path.join(self.dataset_dir, 'entity2id.txt'), 'r', encoding='utf-8') as fr:
                for id, line in enumerate(fr):
                    if id == 0:
                        continue
                    line = line.replace('\n', '')
                    line_split = line.split('\t')[0]
                    entity2id.append(line_split)
            entityid2name = []
            with open(os.path.join(self.dataset_dir, 'entityid2name.txt'), 'r', encoding='utf-8') as fr:
                for id, line in enumerate(fr):
                    if id == 0:
                        continue
                    line = line.replace('\n', '')
                    line_split = line.split('\t')[1]
                    entityid2name.append(line_split)

            for id, e2i in enumerate(entity2id):
                self.entity.append(entityid2name[id] + ' (' +  e2i + ')')


        entity_num = len(self.entity)
        entity_id = {}
        file = self.data_path + "entity2id.txt"
        f = open(file, 'w', encoding='utf-8')
        f.write(str(entity_num) + '\n')
        for i, e in enumerate(self.entity):
            f.write(str(e) + '\t' + str(i) + '\n')
            entity_id[e] = i
        f.close()

        file = self.data_path + "entityid2name.txt"
        f = open(file, 'w', encoding='utf-8')
        f.write(str(entity_num) + '\n')
        for i, e in enumerate(self.entity):
            f.write(str(i) + '\t' + str(e) + '\n')
        f.close()

        '''
        2. relation2id.txt relationid2name.txt
        '''
        self.relation = []
        with open(os.path.join(self.dataset_dir, 'relationid2name.txt'), 'r', encoding='utf-8') as fr:
            for id, line in enumerate(fr):
                if id == 0:
                    continue
                line = line.replace('\n', '')
                line_split = line.split('\t')[1]
                if 'FB15k' in self.dataset:
                    tokens = line_split.split(' , ')
                    line_split = ""
                    for token in tokens:
                        token = token.replace('.', '')
                        line_split += token + '/'
                    line_split = line_split[:-1]
                self.relation.append(line_split)

        relation_id = {}
        relation_num = len(self.relation)
        file = self.data_path + "relation2id.txt"
        f = open(file, 'w', encoding='utf-8')
        f.write(str(relation_num) + '\n')
        for i, r in enumerate(self.relation):
            f.write(str(r) + '\t' + str(i) + '\n')
            relation_id[r] = i
        f.close()

        file = self.data_path + "relationid2name.txt"
        f = open(file, 'w', encoding='utf-8')
        f.write(str(relation_num) + '\n')
        for i, r in enumerate(self.relation):
            f.write(str(i) + '\t' + str(r) + '\n')
        f.close()

        '''
        3. train valid test
        '''
        self.train_data = []
        with open(os.path.join(self.dataset_dir, 'train2id.txt'), 'r', encoding='utf-8') as fr:
            for id, line in enumerate(fr):
                if id == 0:
                    continue
                line = line.replace('\n', '')
                line_split = line.split()
                self.train_data.append([line_split[0], line_split[1], line_split[2]])
        trd_num = len(self.train_data)
        file = self.data_path + "train2id.txt"
        f = open(file, 'w', encoding='utf-8')
        f.write(str(trd_num) + '\n')
        for [h, o, r] in self.train_data:
            f.write(str(h) + ' ' + str(o) + ' ' + str(r) + '\n')
        f.close()


        self.test_data = []
        with open(os.path.join(self.dataset_dir, 'test2id.txt'), 'r', encoding='utf-8') as fr:
            for id, line in enumerate(fr):
                if id == 0:
                    continue
                line = line.replace('\n', '')
                line_split = line.split()
                self.test_data.append([line_split[0], line_split[1], line_split[2]])
        trd_num = len(self.test_data)
        file = self.data_path + "test2id.txt"
        f = open(file, 'w', encoding='utf-8')
        f.write(str(trd_num) + '\n')
        for [h, o, r] in self.test_data:
            f.write(str(h) + ' ' + str(o) + ' ' + str(r) + '\n')
        f.close()


        self.valid_data = []
        with open(os.path.join(self.dataset_dir, 'valid2id.txt'), 'r', encoding='utf-8') as fr:
            for id, line in enumerate(fr):
                if id == 0:
                    continue
                line = line.replace('\n', '')
                line_split = line.split()
                self.valid_data.append([line_split[0], line_split[1], line_split[2]])
        trd_num = len(self.valid_data)
        file = self.data_path + "valid2id.txt"
        f = open(file, 'w', encoding='utf-8')
        f.write(str(trd_num) + '\n')
        for [h, o, r] in self.valid_data:
            f.write(str(h) + ' ' + str(o) + ' ' + str(r) + '\n')
        f.close()


    def find_k_hop_neighbors_with_weights(self, G, node, k):

        current_level_neighbors = {node}
        all_neighbors = set(current_level_neighbors)
        edges_with_weights = {}

        for _ in range(k):
            next_level_neighbors = set()
            for neighbor in current_level_neighbors:
                for n in G.neighbors(neighbor):
                    if n not in all_neighbors:
                        next_level_neighbors.add(n)
                    edge = (neighbor, n)
                    if edge in G.edges:
                        # edges_with_weights[edge] = G.edges[edge].get('weight', 1)
                        edges_with_weights[edge] = G.edges[edge]

            current_level_neighbors = next_level_neighbors - all_neighbors

            all_neighbors.update(current_level_neighbors)

        return all_neighbors, edges_with_weights

    def generation_tkg_store(self):

        self.dataset_dir = "./data/processed/" + self.dataset + "/"
        self.train_data = self.load_factruples(self.dataset_dir, 'train2id.txt')
        self.test_data = self.load_factruples(self.dataset_dir, 'test2id.txt')
        self.dev_data = self.load_factruples(self.dataset_dir, 'valid2id.txt')
        self.train_data = np.concatenate((self.train_data, self.dev_data), axis=0)
        all_data = np.concatenate((self.train_data, self.test_data), axis=0)
        head_l = all_data[:, 0]
        rel_l = all_data[:, 1]
        tail_l = all_data[:, 2]
        time_l = all_data[:, 3]
        # max_idx = time_l.max()  # 时间最大编号
        # t_feat = int(max_idx / self.time_granularity) + 1
        max_idx = max(head_l.max(), tail_l.max())  # 节点最大编号
        n_feat = max_idx + 1
        r_feat = max(rel_l) + 1
        adj_node_list = [[] for _ in range(n_feat)]  # 空的邻接矩阵，根据头实体构建
        for head, tail, rel, time in zip(head_l, tail_l, rel_l, time_l):  # head, tail, rel, time
            adj_node_list[head].append([tail, rel, time])  # 有向边,head节点所连接的节点信息

        edges_list = []
        for data in all_data:
            head_time = str(data[0]) + '_' + str(data[3])
            tail_time = str(data[2]) + '_' + str(data[3])
            edges_list.append((head_time, tail_time, {"edge": data[1]}))
            # 加入反向边
            edges_list.append((tail_time, head_time, {"edge": data[1] + r_feat}))
        # 需要加辅助边，相同实体，不同时间的连接，关系就是最大关系
        for node, concats in enumerate(adj_node_list):
            if len(concats) == 0:
                continue
            times = np.array(concats)[:, 2]
            times = np.unique(times)
            sorted_indices = np.argsort(times)[::-1]
            times = times[sorted_indices]
            currt = times[0]
            for time in times[1:]:
                head_time = str(node) + '_' + str(currt)
                tail_time = str(node) + '_' + str(time)
                edges_list.append((head_time, tail_time, {"edge": r_feat * 2}))
                edges_list.append((tail_time, head_time, {"edge": r_feat * 2}))
                currt = time

        # 封装成 nx.DiGraph()
        G = nx.DiGraph(edges_list)
        graph = {"graph": G}
        with open(self.data_path + 'Graph.pkl', 'wb') as fo:
            pickle.dump(graph, fo)
            fo.close()

        all_nodes = list(G.nodes)
        edges_list = []
        for node in all_nodes:
            two_hop_paths = find_2_hop_paths(G, node)
            for path in two_hop_paths:
                node1, node2, node3 = path[0], path[1], path[2]
                sr = G[node1][node2]['edge']
                tr = G[node2][node3]['edge']
                if (sr, tr) not in edges_list:
                    edges_list.append((sr, tr))
        G = nx.DiGraph(edges_list)
        graph = {"meta_graph": G}
        with open(self.data_path + 'Meta_Graph.pkl', 'wb') as fo:
            pickle.dump(graph, fo)
            fo.close()


    def generation_kg_store(self):
        self.dataset_dir = "./data/processed/" + self.dataset + "/"
        self.train_data = self.load_factruples(self.dataset_dir, 'train2id.txt')
        self.test_data = self.load_factruples(self.dataset_dir, 'test2id.txt')
        self.dev_data = self.load_factruples(self.dataset_dir, 'valid2id.txt')
        self.train_data = np.concatenate((self.train_data, self.dev_data), axis=0)
        all_data = np.concatenate((self.train_data, self.test_data), axis=0)
        edges_list = []
        head_l = all_data[:, 0]
        rel_l = all_data[:, 1]
        tail_l = all_data[:, 2]
        r_feat = max(rel_l) + 1
        for data in all_data:
            head = data[0]
            tail = data[2]
            edges_list.append((head, tail, {"edge": data[1]}))

            edges_list.append((tail, head, {"edge": data[1] + r_feat}))

        G = nx.DiGraph(edges_list)


        graph = {"graph": G}
        with open(self.data_path + 'Graph.pkl', 'wb') as fo:
            pickle.dump(graph, fo)
            fo.close()


        all_nodes = list(G.nodes)
        edges_list = []
        for node in all_nodes:
            two_hop_paths = find_2_hop_paths(G, node)
            for path in two_hop_paths:
                node1, node2, node3 = path[0], path[1], path[2]
                sr = G[node1][node2]['edge']
                tr = G[node2][node3]['edge']
                if (sr, tr) not in edges_list:
                    edges_list.append((sr, tr))
        G = nx.DiGraph(edges_list)
        graph = {"meta_graph": G}
        with open(self.data_path + 'Meta_Graph.pkl', 'wb') as fo:  # 保存图信息
            pickle.dump(graph, fo)
            fo.close()

if __name__ == '__main__':
    process = processed()

    if process.dataset == "ICEWS14":
        process.generate_tkg_processed_data()
        process.generation_tkg_store()
    else:
        process.generate_kg_processed_data()
        process.generation_kg_store()












