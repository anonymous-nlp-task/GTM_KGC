import os
from tqdm import tqdm
import scipy.sparse as sp
from collections import defaultdict as ddict
from collections import Counter
import numpy as np
import pandas as pd
import torch
import networkx as nx
from torch.nn.utils.rnn import pad_sequence
import pygtrie



def get_num(dataset_path, dataset, mode='entity'):  # mode: {entity, relation}
    return int(open(os.path.join(dataset_path, dataset, mode + '2id.txt')).readline().strip())


def read(dataset_path, dataset, temporal, filename):
    file_name = os.path.join(dataset_path, dataset, filename)
    with open(file_name) as file:
        lines = file.read().strip().split('\n')
    n_triples = int(lines[0])
    triples = []
    for line in lines[1:]:
        split = line.split(' ')
        if temporal:
            for i in range(4):
                split[i] = int(split[i])
        else:
            for i in range(3):
                split[i] = int(split[i])
        triples.append(split)
    assert n_triples == len(triples), 'number of triplets is not correct.'
    return triples

def read_file(dataset_path, dataset, filename):
    id2name = []
    file_name = os.path.join(dataset_path, dataset, filename)
    with open(file_name, encoding='utf-8') as file:
        lines = file.read().strip('\n').split('\n')

    for i in range(1, len(lines)):
        ids, name = lines[i].split('\t')
        id2name.append(name)
    return id2name

def read_name(dataset_path, dataset, temporal):
    ent_name_file = 'entityid2name.txt'
    rel_name_file = 'relationid2name.txt'
    id_time_file = 'timeid2name.txt'
    ent_name_list = read_file(dataset_path, dataset, ent_name_file)
    rel_name_list = read_file(dataset_path, dataset, rel_name_file)

    if temporal:
        id_time_list = read_file(dataset_path, dataset, id_time_file)
        time_id_list = {time: id for id, time in enumerate(id_time_list)}
        return ent_name_list, rel_name_list, id_time_list, time_id_list
    else:
        return ent_name_list, rel_name_list, [], []


def load_factruples(dataset_path, dataset, fileName):
    with open(os.path.join(dataset_path, dataset, fileName), 'r') as fr:
        quadrupleList = []
        for id, line in enumerate(fr):
            if id == 0:
                continue
            line_split = line.split()
            head = int(line_split[0])
            rel = int(line_split[2])
            tail = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])

    return np.array(quadrupleList)


def get_ground_truth(configs, triples):
    # tail_ground_truth, head_ground_truth, rel_ground_truth = ddict(list), ddict(list), ddict(list)
    tail_ground_truth, head_ground_truth = ddict(list), ddict(list)
    for triple in triples:
        if configs.temporal:
            head, tail, rel, time = triple
            tail_ground_truth[(head, rel, time)].append(tail)
            head_ground_truth[(tail, rel, time)].append(head)
            # rel_ground_truth[(head, tail, time)].append(rel)
        else:
            head, tail, rel = triple
            tail_ground_truth[(head, rel)].append(tail)
            head_ground_truth[(tail, rel)].append(head)
            # rel_ground_truth[(head, tail)].append(rel)
    # return tail_ground_truth, head_ground_truth, rel_ground_truth
    return tail_ground_truth, head_ground_truth

def get_next_token_dict(configs, ent_token_ids_in_trie, prefix_trie):
    neg_candidate_mask = []
    next_token_dict = {(): [32099] * configs.n_ent}
    for ent_id in tqdm(range(configs.n_ent)):
        rows, cols = [0], [32099]
        input_ids = ent_token_ids_in_trie[ent_id]
        for pos_id in range(1, len(input_ids)):
            cur_input_ids = input_ids[:pos_id]
            if tuple(cur_input_ids) in next_token_dict:
                cur_tokens = next_token_dict[tuple(cur_input_ids)]
            else:
                seqs = prefix_trie.keys(prefix=cur_input_ids)
                cur_tokens = [seq[pos_id] for seq in seqs]
                next_token_dict[tuple(cur_input_ids)] = Counter(cur_tokens)
            cur_tokens = list(set(cur_tokens))
            rows.extend([pos_id] * len(cur_tokens))
            cols.extend(cur_tokens)
        sparse_mask = sp.coo_matrix(([1] * len(rows), (rows, cols)), shape=(len(input_ids), configs.vocab_size), dtype=np.long)
        neg_candidate_mask.append(sparse_mask)
    return neg_candidate_mask, next_token_dict

def construct_prefix_trie(ent_token_ids_in_trie):
    trie = pygtrie.Trie()
    for input_ids in ent_token_ids_in_trie:
        trie[input_ids] = True
    return trie

def batchify(output_dict, key, padding_value=None, return_list=False):
    tensor_out = [out[key] for out in output_dict]
    if return_list:
        return tensor_out
    if not isinstance(tensor_out[0], torch.LongTensor) and not isinstance(tensor_out[0], torch.FloatTensor):
        tensor_out = [torch.LongTensor(value) for value in tensor_out]
    if padding_value is None:
        tensor_out = torch.stack(tensor_out, dim=0)
    else:
        tensor_out = pad_sequence(tensor_out, batch_first=True, padding_value=padding_value)
    return tensor_out

def _get_performance(ranks, dataset):
    ranks = np.array(ranks, dtype=np.float)
    out = dict()
    out['mr'] = ranks.mean(axis=0)
    out['mrr'] = (1. / ranks).mean(axis=0)
    out['hit1'] = np.sum(ranks == 1, axis=0) / len(ranks)
    out['hit3'] = np.sum(ranks <= 3, axis=0) / len(ranks)
    out['hit10'] = np.sum(ranks <= 10, axis=0) / len(ranks)
    if dataset == 'NELL':
        out['hit5'] = np.sum(ranks <= 5, axis=0) / len(ranks)
    return out

def get_performance(model, tail_ranks, head_ranks):
    tail_out = _get_performance(tail_ranks, model.configs.dataset)
    head_out = _get_performance(head_ranks, model.configs.dataset)
    mr = np.array([tail_out['mr'], head_out['mr']])
    mrr = np.array([tail_out['mrr'], head_out['mrr']])
    hit1 = np.array([tail_out['hit1'], head_out['hit1']])
    hit3 = np.array([tail_out['hit3'], head_out['hit3']])
    hit10 = np.array([tail_out['hit10'], head_out['hit10']])

    if model.configs.dataset == 'NELL':
        val_mrr = tail_out['mrr'].item()
        model.log('val_mrr', val_mrr)
        hit5 = np.array([tail_out['hit5'], head_out['hit5']])
        perf = {'mrr': mrr, 'mr': mr, 'hit@1': hit1, 'hit@3': hit3, 'hit@5': hit5, 'hit@10': hit10}
    else:
        val_mrr = mrr.mean().item()
        model.log('val_mrr', val_mrr)
        perf = {'mrr': mrr, 'mr': mr, 'hit@1': hit1, 'hit@3': hit3, 'hit@10': hit10}
    perf = pd.DataFrame(perf, index=['tail ranking', 'head ranking'])
    perf.loc['mean ranking'] = perf.mean(axis=0)
    for hit in ['hit@1', 'hit@3', 'hit@5', 'hit@10']:
        if hit in list(perf.columns):
            perf[hit] = perf[hit].apply(lambda x: '%.2f%%' % (x * 100))
    return perf


def get_temporal_simplify_path(adj_node_list, triple, tgtidx, G, r_feat, path_k, mode):
    if mode == 0:  # tail
        facts = adj_node_list[np.where(adj_node_list[:, 1] == tgtidx)]
        facts = np.delete(facts, np.where(
            (facts[:, 0] == triple[0]) & (facts[:, 1] == triple[1]) & (facts[:, 2] == triple[2]) & (
                    facts[:, 3] == triple[3])), axis=0)
        index = np.argsort(np.abs(facts[:, 3] - triple[3]))
        facts = facts[index]
        simplify_path = []
        for fact in facts:
            temp = []
            head_time = str(triple[0]) + '_' + str(triple[3])
            tail_time = str(fact[1]) + '_' + str(fact[3])
            try:
                shortest_path = nx.shortest_path(G, source=head_time, target=tail_time)
                for i in range(len(shortest_path) - 1):
                    node1 = shortest_path[i]
                    node2 = shortest_path[i + 1]
                    r = G[node1][node2]['edge']
                    h = node1.split('_')[0]
                    o = node2.split('_')[0]
                    t = node1.split('_')[1]
                    temp.append((h, o, r, t))

                temp = [tp for tp in temp if r_feat * 2 > tp[2]]
                if temp not in simplify_path:
                    simplify_path.append(temp)
                if len(simplify_path) >= path_k:
                    break
            except:
                pass

        return simplify_path

    else:
        # head
        facts = adj_node_list[np.where(adj_node_list[:, 0] == tgtidx)]
        facts = np.delete(facts, np.where(
            (facts[:, 0] == triple[0]) & (facts[:, 1] == triple[1]) & (facts[:, 2] == triple[2]) & (
                    facts[:, 3] == triple[3])), axis=0)
        index = np.argsort(np.abs(facts[:, 3] - triple[3]))
        facts = facts[index]
        simplify_path = []
        for fact in facts:
            temp = []
            head_time = str(triple[1]) + '_' + str(triple[3])
            tail_time = str(fact[0]) + '_' + str(fact[3])
            try:
                shortest_path = nx.shortest_path(G, source=head_time, target=tail_time)
                for i in range(len(shortest_path) - 1):
                    node1 = shortest_path[i]
                    node2 = shortest_path[i + 1]
                    r = G[node1][node2]['edge']
                    if r > r_feat:
                        pass
                    h = node1.split('_')[0]
                    o = node2.split('_')[0]
                    t = node1.split('_')[1]
                    temp.append((h, o, r, t))

                temp = [tp for tp in temp if r_feat * 2 > tp[2]]
                if temp not in simplify_path:
                    simplify_path.append(temp)
                if len(simplify_path) >= path_k:
                    break
            except:
                pass
        return simplify_path

def get_simplify_path(triple, tgtidx, G, path_k, cutoff, mode):
    # cutoff = 4
    if mode == 0:  # tail
        paths = list(nx.all_simple_paths(G, source=triple[0], target=tgtidx, cutoff=cutoff))
    else:
        paths = list(nx.all_simple_paths(G, source=triple[1], target=tgtidx, cutoff=cutoff))
    simplify_path = []
    for shortest_path in paths:
        temp = []
        for i in range(len(shortest_path) - 1):
            h = shortest_path[i]
            o = shortest_path[i + 1]
            r = G[h][o]['edge']
            temp.append((h, o, r))

        temp = [sublist for sublist in temp if sublist != (triple[0], triple[1], triple[2])]
        if len(temp) > 0 and temp not in simplify_path:
            simplify_path.append(temp)
        if len(simplify_path) >= path_k:
            break
    return simplify_path

def get_split_dataset_for_meta_paths(bridging_point, meta_paths):
    '''
    input:meta_paths
    output:[{input:xxx, label:xxx},{....}]
    '''
    dataset = []
    for meta_path in meta_paths:
        stoper = True
        idx = -1
        while stoper:
            if abs(idx) > bridging_point:
                break
            input = [meta_path[0], -2] + meta_path[idx:]
            label = meta_path[1:idx]
            if len(meta_path) - (len(input) - 1) <= 0:
                stoper = False
            else:
                dataset.append({'input': input, 'label': label})
                idx = idx - 1

    return dataset

def find_k_hop_neighbors_with_weights(G, node, k):
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

def find_2_hop_paths(G, start_node):
    two_hop_paths = []

    for neighbor in G.neighbors(start_node):
        for second_neighbor in G.neighbors(neighbor):
            if second_neighbor != start_node:
                two_hop_paths.append([start_node, neighbor, second_neighbor])

    return two_hop_paths

def find_temporal_bridge_edge(adj_node_list, triple, tgtidx, G, MG, path_k, r_feat, mode):
    mask_ent = -100
    if mode == 0:
        facts = adj_node_list[np.where(adj_node_list[:, 1] == tgtidx)]
        facts = np.delete(facts, np.where(
            (facts[:, 0] == triple[0]) & (facts[:, 1] == triple[1]) & (facts[:, 2] == triple[2]) & (
                    facts[:, 3] == triple[3])), axis=0)
        index = np.argsort(np.abs(facts[:, 3] - triple[3]))
        facts = facts[index]
        simplify_path = []
        for fact in facts:
            tail_time = str(fact[1]) + '_' + str(fact[3])
            source_rel = triple[2]
            k_hop_neighbors, edges_with_weights = find_k_hop_neighbors_with_weights(G, tail_time, 1)
            for edge, weight in edges_with_weights.items():
                # temp = [(triple[0], fact[1], triple[2], triple[3])]
                temp = []
                target_rel = weight['edge']
                try:
                    rel_path = nx.shortest_path(MG, source=source_rel, target=target_rel)
                    for i, rel in enumerate(rel_path[1:]):
                        if i == 0:
                            h = triple[0]
                        else:
                            h = mask_ent

                        if i + 1 == len(rel_path[1:]):
                            o = tgtidx
                        else:
                            o = mask_ent
                        if rel < r_feat * 2:
                            h = int(edge[0].split('_')[0])
                            t = int(edge[0].split('_')[1])
                            o = int(edge[1].split('_')[0])
                            temp.append((h, o, rel, t))
                except:
                    pass
                if [(triple[0], fact[1], triple[2], triple[3])] != temp:
                    if temp not in simplify_path:
                        simplify_path.append(temp)
                    if len(simplify_path) >= path_k:
                        break
            if len(simplify_path) >= path_k:
                break
        return simplify_path

    else:
        facts = adj_node_list[np.where(adj_node_list[:, 0] == tgtidx)]
        facts = np.delete(facts, np.where(
            (facts[:, 0] == triple[0]) & (facts[:, 1] == triple[1]) & (facts[:, 2] == triple[2]) & (
                    facts[:, 3] == triple[3])), axis=0)
        index = np.argsort(np.abs(facts[:, 3] - triple[3]))
        facts = facts[index]
        simplify_path = []
        for fact in facts:
            tail_time = str(fact[0]) + '_' + str(fact[3])
            source_rel = triple[2]
            k_hop_neighbors, edges_with_weights = find_k_hop_neighbors_with_weights(G, tail_time, 1)
            for edge, weight in edges_with_weights.items():
                temp = [(triple[1], fact[0], triple[2], triple[3])]
                target_rel = weight['edge']
                try:
                    rel_path = nx.shortest_path(MG, source=source_rel, target=target_rel)
                    for rel in rel_path:
                        if rel < r_feat * 2:
                            h = int(edge[0].split('_')[0])
                            t = int(edge[0].split('_')[1])
                            o = int(edge[1].split('_')[0])
                            temp.append((h, o, rel, t))
                except:
                    pass
                if [(triple[1], fact[0], triple[2], triple[3])] != temp:
                    if temp not in simplify_path:
                        simplify_path.append(temp)
                    if len(simplify_path) >= path_k:
                        break
            if len(simplify_path) >= path_k:
                break
        return simplify_path

def find_bridge_edge(triple, tgtidx, G, MG, path_k, r_feat, mode):
    simplify_path = []
    if mode == 0:
        source_rel = triple[2]
        k_hop_neighbors, edges_with_weights = find_k_hop_neighbors_with_weights(G, tgtidx, 1)
        for edge, weight in edges_with_weights.items():
            temp = []
            target_rel = weight['edge']
            try:
                rel_path = nx.shortest_path(MG, source=source_rel, target=target_rel)
                for i, rel in enumerate(rel_path):
                    if i == 0:
                        h = triple[0]
                        temp.append(h)
                    elif i + 1 == len(rel_path):
                        o = tgtidx
                        temp.append(o)
                    else:
                        temp.append(rel)

            except:
                pass
            if len(temp) > 0:
                if temp not in simplify_path:
                    simplify_path.append(temp)
                if len(simplify_path) >= path_k:
                    break

        return simplify_path

    else:
        source_rel = triple[2]
        k_hop_neighbors, edges_with_weights = find_k_hop_neighbors_with_weights(G, tgtidx, 1)
        for edge, weight in edges_with_weights.items():
            # temp = [(triple[1], tgtidx, triple[2])]
            temp = []
            target_rel = weight['edge']
            try:
                rel_path = nx.shortest_path(MG, source=source_rel, target=target_rel)
                for i, rel in enumerate(rel_path):
                    if i == 0:
                        h = triple[1]
                        temp.append(h)
                    elif i + 1 == len(rel_path):
                        o = tgtidx
                        temp.append(o)
                    else:
                        temp.append(rel)
            except:
                pass
            if len(temp) > 0:
                if temp not in simplify_path:
                    simplify_path.append(temp)
                if len(simplify_path) >= path_k:
                    break
        return simplify_path





