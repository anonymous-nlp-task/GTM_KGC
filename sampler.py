import pickle
import argparse
import os
import numpy as np
import warnings
import networkx as nx
import sqlite3
from helper import read, get_simplify_path, get_temporal_simplify_path
import time
#ICEWS14, FB15k-237, FB15k-237N,  NELL, WN18RR
warnings.filterwarnings('ignore', category=DeprecationWarning)
parser = argparse.ArgumentParser()
parser.add_argument('-dataset_path', type=str, default='./data/processed')
parser.add_argument('-dataset', dest='dataset', default='FB15k-237', help='Dataset to use, WN18RR, FB15k-237, FB15k-237N, ICEWS14, NELL')
parser.add_argument('-temporal', action='store_true', default=False, help='')
parser.add_argument('-path_k', default=1, type=int, help='')
parser.add_argument('-cutoff', default=2, type=int, help='WN18RR=4, FB15k-237=2, FB15k-237N=2, NELL=3')
parser.add_argument('-max_sampling_num', default=20, type=int, help='')
parser.add_argument('-pretrained_model', type=str, default='./models/t5-base', help='')
configs = parser.parse_args()

with open(configs.dataset_path + '/' + configs.dataset + '/Graph.pkl', 'rb') as f:
    Graph = pickle.load(f)['graph']
G = nx.DiGraph(Graph)


def load_factruples(dataset, inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        for id, line in enumerate(fr):
            if id == 0:
                continue
            line_split = line.split()
            head = int(line_split[0])
            rel = int(line_split[2])
            tail = int(line_split[1])
            if dataset == "ICEWS14":
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
            else:
                quadrupleList.append([head, rel, tail])

    return np.array(quadrupleList)

def tkg_sampler_main(filepath, obj, mode):
    dataset_dir = configs.dataset_path + '/' + configs.dataset
    train_data = load_factruples(configs.dataset, dataset_dir, 'train2id.txt')
    test_data = load_factruples(configs.dataset, dataset_dir, 'test2id.txt')
    dev_data = load_factruples(configs.dataset, dataset_dir, 'valid2id.txt')
    train_data = np.concatenate((train_data, dev_data), axis=0)
    all_data = np.concatenate((train_data, test_data), axis=0)
    permutation = np.array([0, 2, 1, 3])
    all_data = all_data[:, permutation]
    rel_l = all_data[:, 2]
    r_feat = max(rel_l) + 1

    if os.path.exists(filepath + 'sampler_{}_{}.pkl'.format(mode, configs.path_k)):
        os.remove(filepath + 'sampler_{}_{}.pkl'.format(mode, configs.path_k))
        time.sleep(1)

    if mode == 'tail':
        with open(filepath + 'sampler_{}_{}.pkl'.format(mode, configs.path_k), 'ab') as fo:
            for id, (key, val) in enumerate(obj.items()):
                inf = {}
                triple = eval(key)
                inf['query'] = triple
                inf['label'] = triple[1]
                inf['canidate'] = val
                canidate_paths = []
                for cnx in val:
                    simplify_path = get_temporal_simplify_path(all_data, triple, cnx, G, r_feat, configs.path_k, 0)
                    # if len(simplify_path) == 0 and configs.extended_edge > 0:
                    #     temp = []
                    #     temp.append((triple[0], cnx, -1, triple[3]))
                    #     simplify_path.append(temp)
                    canidate_paths.append(simplify_path)
                inf['canidate_paths'] = canidate_paths
                # canidates.append(inf)
                if len(simplify_path) > 0:
                    pickle.dump(inf, fo)


    else:
        with open(filepath + 'sampler_{}_{}.pkl'.format(mode, configs.path_k), 'ab') as fo:
            for id, (key, val) in enumerate(obj.items()):
                inf = {}
                triple = eval(key)
                inf['query'] = triple
                inf['label'] = triple[0]
                inf['canidate'] = val
                canidate_paths = []
                for cnx in val:
                    simplify_path = get_temporal_simplify_path(all_data, triple, cnx, G, r_feat, configs.path_k, 1)
                    # if len(simplify_path) == 0 and configs.extended_edge > 0:
                    #     temp = []
                    #     temp.append((cnx, triple[1], -1, triple[3]))
                    #     simplify_path.append(temp)
                    canidate_paths.append(simplify_path)
                inf['canidate_paths'] = canidate_paths
                if len(simplify_path) > 0:
                    pickle.dump(inf, fo)
            fo.close()

    print("successful, building candidate store!")

def kg_sampler_main(filepath, obj, mode):

    if os.path.exists(filepath + 'sampler_{}_{}.pkl'.format(mode, configs.path_k)):
        os.remove(filepath + 'sampler_{}_{}.pkl'.format(mode, configs.path_k))
        time.sleep(1)


    if mode == 'tail':
        with open(filepath + 'sampler_{}_{}.pkl'.format(mode, configs.path_k), 'ab') as fo:
            for id, (key, val) in enumerate(obj.items()):
                inf = {}
                triple = eval(key)
                inf['query'] = triple
                inf['label'] = triple[1]
                inf['canidate'] = val
                canidate_paths = []
                for cnx in val:
                    simplify_path = get_simplify_path(triple, cnx, G, configs.path_k, configs.cutoff, 0)
                    canidate_paths.append(simplify_path)
                inf['canidate_paths'] = canidate_paths
                pickle.dump(inf, fo)
            fo.close()
    else:
        with open(filepath + 'sampler_{}_{}.pkl'.format(mode, configs.path_k), 'ab') as fo:
            for id, (key, val) in enumerate(obj.items()):
                inf = {}
                triple = eval(key)
                inf['query'] = triple
                inf['label'] = triple[0]
                inf['canidate'] = val
                canidate_paths = []
                for cnx in val:
                    simplify_path = get_simplify_path(triple, cnx, G, configs.path_k, configs.cutoff, 1)
                    canidate_paths.append(simplify_path)
                inf['canidate_paths'] = canidate_paths
                pickle.dump(inf, fo)
            fo.close()

    print("successful, building candidate store!")

def sampling_store(file_path, filename):
    train_triples = read(configs.dataset_path, configs.dataset, configs.temporal, 'train2id.txt')
    # test_triples = read(configs.dataset_path, configs.dataset, configs.temporal, 'test2id.txt')
    valid_triples = read(configs.dataset_path, configs.dataset, configs.temporal, 'valid2id.txt')
    train_triples = train_triples + valid_triples
    if os.path.exists(file_path + filename):
        conn = sqlite3.connect(file_path + filename)
        cursor = conn.cursor()
        cursor.execute("DELETE from query_result")
        cursor.execute("DELETE from sampling_tail")
        cursor.execute("DELETE from sampling_head")
        cursor.execute("DELETE from seed_tail")
        cursor.execute("DELETE from seed_head")
        cursor.execute("update sqlite_sequence set seq = 0 where name = 'query_result'")
        cursor.execute("update sqlite_sequence set seq = 0 where name = 'sampling_tail'")
        cursor.execute("update sqlite_sequence set seq = 0 where name = 'sampling_head'")
        cursor.execute("update sqlite_sequence set seq = 0 where name = 'seed_tail'")
        cursor.execute("update sqlite_sequence set seq = 0 where name = 'seed_head'")
        conn.commit()
    else:
        conn = sqlite3.connect(file_path + filename)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE query_result(id INTEGER, query TEXT, mode TEXT, flag TEXT)''')
        cursor.execute('''CREATE TABLE sampling_tail(id INTEGER PRIMARY KEY AUTOINCREMENT, query TEXT, pos TEXT, neg TEXT, pid INTEGER)''')
        cursor.execute('''CREATE TABLE sampling_head(id INTEGER PRIMARY KEY AUTOINCREMENT, query TEXT, pos TEXT, neg TEXT, pid INTEGER)''')
        cursor.execute('''CREATE TABLE seed_tail(id INTEGER PRIMARY KEY AUTOINCREMENT, query TEXT, pos TEXT, neg TEXT, pid INTEGER)''')
        cursor.execute('''CREATE TABLE seed_head(id INTEGER PRIMARY KEY AUTOINCREMENT, query TEXT, pos TEXT, neg TEXT, pid INTEGER)''')
        conn.commit()

    file_tail_name = 'sampler_tail_{}.pkl'.format(configs.path_k)
    file_head_name = 'sampler_head_{}.pkl'.format(configs.path_k)
    pid = 1
    index = 0
    with open(file_path + file_tail_name, 'rb') as fo:
        while True:
            try:
                data = pickle.load(fo)
                if data['label'] in data['canidate']:
                    canindex = data['canidate'].index(data['label'])
                    canidate_paths = data['canidate_paths']
                    if len(canidate_paths[canindex]) > 0:
                        query = data['query']
                        poset = canidate_paths[canindex]
                        del canidate_paths[canindex]
                        ngset = []
                        for canpas in canidate_paths:
                            for pa in canpas:
                                ngset.append(pa)

                        if len(ngset) > 0 and len(poset) > 0:
                            if query in train_triples:
                                sql = '''INSERT INTO query_result (id, query, mode, flag) VALUES ({}, '{}', '{}', '{}')
                                '''.format(pid, str(query), 'tail', 'train')
                                cursor.execute(sql)
                            else:
                                sql = '''INSERT INTO query_result (id, query, mode, flag) VALUES ({}, '{}', '{}', '{}')
                                '''.format(pid, str(query), 'tail', 'test')
                                cursor.execute(sql)
                            cursor.execute("INSERT INTO sampling_tail (query, pos, neg, pid) VALUES (?, ?, ?, ?);", (str(query), str(poset), str(ngset), pid))
                            cursor.execute("INSERT INTO seed_tail (query, pos, neg, pid) VALUES (?, ?, ?, ?);",
                                           (str(query), str([configs.max_sampling_num for _ in range(len(poset))]), str([configs.max_sampling_num for _ in range(len(ngset))]), pid))
                            pid = pid + 1

                index = index + 1
            except EOFError:
                break
    index = 0
    with open(file_path + file_head_name, 'rb') as fo:
        while True:
            try:
                data = pickle.load(fo)
                if data['label'] in data['canidate']:
                    canindex = data['canidate'].index(data['label'])
                    canidate_paths = data['canidate_paths']
                    if len(canidate_paths[canindex]) > 0:
                        query = data['query']
                        poset = canidate_paths[canindex]
                        del canidate_paths[canindex]
                        ngset = []
                        for canpas in canidate_paths:
                            for pa in canpas:
                                ngset.append(pa)

                        if len(ngset) > 0 and len(poset) > 0:
                            if query in train_triples:
                                sql = '''INSERT INTO query_result (id, query, mode, flag) VALUES ({}, '{}', '{}', '{}')
                                '''.format(pid, str(query), 'head', 'train')
                                cursor.execute(sql)
                            else:
                                sql = '''INSERT INTO query_result (id, query, mode, flag) VALUES ({}, '{}', '{}', '{}')
                                '''.format(pid, str(query), 'head', 'test')
                                cursor.execute(sql)
                            cursor.execute("INSERT INTO sampling_head (query, pos, neg, pid) VALUES (?, ?, ?, ?);",
                                           (str(query), str(poset), str(ngset), pid))
                            cursor.execute("INSERT INTO seed_head (query, pos, neg, pid) VALUES (?, ?, ?, ?);",
                                           (str(query), str([configs.max_sampling_num for _ in range(len(poset))]),
                                            str([configs.max_sampling_num for _ in range(len(ngset))]), pid))
                            pid = pid + 1


                index = index + 1
            except EOFError:

                break

    conn.commit()
    conn.close()





if __name__ == '__main__':

    # 读取预测的尾实体候选路径
    filepath = './checkpoint/{}-train_model/'.format(configs.dataset)
    name = 'predict_tails.pkl'
    with open(filepath + name, 'rb') as f:
        obj = pickle.load(f)
    if configs.temporal:
        tkg_sampler_main(filepath, obj, 'tail')
    else:
        kg_sampler_main(filepath, obj, 'tail')


    filepath = './checkpoint/{}-train_model/'.format(configs.dataset)
    name = 'predict_heads.pkl'
    with open(filepath + name, 'rb') as f:
        obj = pickle.load(f)
    if configs.temporal:
        tkg_sampler_main(filepath, obj, 'head')
    else:
        kg_sampler_main(filepath, obj, 'head')

    filepath = './checkpoint/{}-train_model/'.format(configs.dataset)
    filename = 'store.db'
    sampling_store(filepath, filename)












