import torch
import pytorch_lightning as pl
from transformers import T5Tokenizer
from helper import batchify

from torch.utils.data import Dataset, DataLoader


class Bridger_TrainDataset(Dataset):
    def __init__(self, configs, tokenizer, train_triples, name_list_dict):
        self.configs = configs
        self.train_triples = train_triples
        self.tokenizer = tokenizer
        self.rel_name_list = name_list_dict['rel_name_list']

    def __len__(self):
        return len(self.train_triples)

    def __getitem__(self, index):
        # print("============================self.train_triples[index]", self.train_triples[index])
        input, label = self.train_triples[index]['input'], self.train_triples[index]['label']
        src = ''
        tgt = '<extra_id_0>'
        for idx, inp in enumerate(input):
            if inp > 0:
                if idx + 1 < len(input):
                    src = src + self.rel_name_list[inp] + ', '
                else:
                    src = src + self.rel_name_list[inp] + '. '
            else:
                src = src + '<extra_id_0>' + ', '

        for idx, lbl in enumerate(label):
            if idx + 1 < len(input):
                tgt = tgt + self.rel_name_list[lbl] + ', '
            else:
                tgt = tgt + self.rel_name_list[lbl] + '<extra_id_1>'

        tokenized_src = self.tokenizer(src, max_length=self.configs.src_max_length, truncation=True)
        source_ids = tokenized_src.input_ids
        source_mask = tokenized_src.attention_mask
        tokenized_tgt = self.tokenizer(tgt, max_length=self.configs.train_tgt_max_length, truncation=True)
        target_ids = tokenized_tgt.input_ids
        target_mask = tokenized_tgt.attention_mask

        out = {
                'source_ids': source_ids,
                'source_mask': source_mask,
                'target_ids': target_ids,
                'target_mask': target_mask,
        }

        return out

    def collate_fn(self, data):
        agg_data = dict()
        agg_data['source_ids'] = batchify(data, 'source_ids', padding_value=0)
        agg_data['source_mask'] = batchify(data, 'source_mask', padding_value=0)
        agg_data['target_ids'] = batchify(data, 'target_ids', padding_value=0)
        agg_data['target_mask'] = batchify(data, 'target_mask', padding_value=0)
        return agg_data

class Bridger_TestDataset(Dataset):
    def __init__(self, configs, tokenizer, test_triples, name_list_dict):  # mode: {tail, head}
        self.configs = configs
        self.test_triples = test_triples
        self.rel_name_list = name_list_dict['rel_name_list']
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.test_triples)

    def __getitem__(self, index):
        # print("============================self.test_triples[index]", self.test_triples[index])
        input, label = self.test_triples[index]['input'], self.test_triples[index]['label']
        src = ''
        tgt = '<extra_id_0>'
        for idx, inp in enumerate(input):
            if inp > 0:
                if idx + 1 < len(input):
                    src = src + self.rel_name_list[inp] + ', '
                else:
                    src = src + self.rel_name_list[inp] + '. '
            else:
                src = src + '<extra_id_0>' + ', '

        for idx, lbl in enumerate(label):
            if idx + 1 < len(input):
                tgt = tgt + self.rel_name_list[lbl] + ', '
            else:
                tgt = tgt + self.rel_name_list[lbl] + '<extra_id_1>'

        tokenized_src = self.tokenizer(src, max_length=self.configs.src_max_length, truncation=True)
        source_ids = tokenized_src.input_ids
        source_mask = tokenized_src.attention_mask

        out = {
            'source_ids': source_ids,
            'source_mask': source_mask,
            'source_names': src,
            'target_names': tgt,
        }

        return out

    def collate_fn(self, data):
        agg_data = dict()
        agg_data['source_ids'] = batchify(data, 'source_ids', padding_value=0)
        agg_data['source_mask'] = batchify(data, 'source_mask', padding_value=0)
        agg_data['source_names'] = [dt['source_names'] for dt in data]
        agg_data['target_names'] = [dt['target_names'] for dt in data]
        return agg_data

class BridgerDataModule(pl.LightningDataModule):
    def __init__(self, configs, train_triples, valid_triples, test_triples, name_list_dict, running_model='train_model'):
        super().__init__()
        self.configs = configs
        self.train_triples = train_triples
        self.valid_triples = valid_triples
        self.test_triples = test_triples
        # ent_name_list, rel_name_list .type: list
        self.name_list_dict = name_list_dict
        self.running_model = running_model

        self.tokenizer = T5Tokenizer.from_pretrained(configs.pretrained_model)
        self.train_both = None
        self.valid_both = None
        self.test_both= None

    def prepare_data(self):
        self.train_both = Bridger_TrainDataset(self.configs, self.tokenizer, self.train_triples, self.name_list_dict)
        self.valid_both = Bridger_TestDataset(self.configs, self.tokenizer, self.valid_triples, self.name_list_dict)
        self.test_both = Bridger_TestDataset(self.configs, self.tokenizer, self.test_triples, self.name_list_dict)

    def train_dataloader(self):

        train_loader = DataLoader(self.train_both,
                                  batch_size=self.configs.batch_size,
                                  shuffle=True,
                                  collate_fn=self.train_both.collate_fn,
                                  pin_memory=True,
                                  num_workers=self.configs.num_workers)
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(self.valid_both,
                                       batch_size=self.configs.val_batch_size,
                                       shuffle=False,
                                       collate_fn=self.valid_both.collate_fn,
                                       pin_memory=True,
                                       num_workers=self.configs.num_workers)
        return valid_loader
        # return [valid_head_loader, valid_tail_loader]

    def test_dataloader(self):
        test_loader = DataLoader(self.test_both,
                                      batch_size=self.configs.val_batch_size,
                                      shuffle=False,
                                      collate_fn=self.test_both.collate_fn,
                                      pin_memory=True,
                                      num_workers=self.configs.num_workers)
        return test_loader

class Generator_TrainDataset(Dataset):
    def __init__(self, configs, tokenizer, train_triples, name_list_dict, prefix_trie_dict):
        self.configs = configs
        self.train_triples = train_triples
        self.tokenizer = tokenizer
        self.original_ent_name_list = name_list_dict['original_ent_name_list']
        self.ent_name_list = name_list_dict['ent_name_list']
        self.rel_name_list = name_list_dict['rel_name_list']
        self.time_name_list = name_list_dict['id_time_list']
        self.ent_token_ids_in_trie = prefix_trie_dict['ent_token_ids_in_trie']
    def __len__(self):
        return len(self.train_triples) * 2

    def __getitem__(self, index):
        train_triple = self.train_triples[index // 2]
        mode = 'tail' if index % 2 == 0 else 'head'

        if self.configs.temporal:
            head, tail, rel, time = train_triple
            head_name, tail_name, rel_name, time_name = self.original_ent_name_list[head], self.original_ent_name_list[tail], self.rel_name_list[rel], self.time_name_list[time]
            if mode == 'tail':
                src = head_name + ' | ' + rel_name + ' | ' + '<extra_id_0>' + ' | ' + time_name
                tgt = '<extra_id_0>' + tail_name + '<extra_id_1>'
                tgt_ids = tail
            else:
                src = '<extra_id_0>' + ' | ' + rel_name + ' | ' + tail_name + ' | ' + time_name
                tgt = '<extra_id_0>' + head_name + '<extra_id_1>'
                tgt_ids = head

        else:
            head, tail, rel = train_triple
            head_name, tail_name, rel_name = self.original_ent_name_list[head], self.original_ent_name_list[tail], self.rel_name_list[rel]

            if mode == 'tail':
                src = head_name + ' | ' + rel_name + ' | ' + '<extra_id_0>'
                tgt = '<extra_id_0>' + tail_name + '<extra_id_1>'
                tgt_ids = tail
            else:
                src = '<extra_id_0>' + ' | ' + rel_name + ' | ' + tail_name
                tgt = '<extra_id_0>' + head_name + '<extra_id_1>'
                tgt_ids = head



        tokenized_src = self.tokenizer(src, max_length=self.configs.src_max_length, truncation=True)
        source_ids = tokenized_src.input_ids
        source_mask = tokenized_src.attention_mask
        tokenized_tgt = self.tokenizer(tgt, max_length=self.configs.train_tgt_max_length, truncation=True)
        target_ids = tokenized_tgt.input_ids
        target_mask = tokenized_tgt.attention_mask
        target_names = self.ent_name_list[tgt_ids]

        ent_rel = torch.LongTensor([head, rel]) if mode == 'tail' else torch.LongTensor([tail, rel])
        out = {
                'source_ids': source_ids,
                'source_mask': source_mask,
                'target_ids': target_ids,
                'target_mask': target_mask,
                'tgt_ids': tgt_ids,
                'target_names': target_names,
                'train_triple': train_triple,
                'ent_rel': ent_rel,
        }

        return out

    def collate_fn(self, data):
        agg_data = dict()
        agg_data['source_ids'] = batchify(data, 'source_ids', padding_value=0)
        agg_data['source_mask'] = batchify(data, 'source_mask', padding_value=0)
        agg_data['target_ids'] = batchify(data, 'target_ids', padding_value=0)
        agg_data['target_mask'] = batchify(data, 'target_mask', padding_value=0)
        agg_data['target_names'] = [dt['target_names'] for dt in data]
        agg_data['tgt_ids'] = [dt['tgt_ids'] for dt in data]
        agg_data['train_triple'] = batchify(data, 'train_triple', return_list=True)
        agg_data['ent_rel'] = batchify(data, 'ent_rel')
        return agg_data

class Generator_TestDataset(Dataset):
    def __init__(self, configs, tokenizer, test_triples, name_list_dict, prefix_trie_dict, mode):  # mode: {tail, head}
        self.configs = configs
        self.test_triples = test_triples
        self.original_ent_name_list = name_list_dict['original_ent_name_list']
        self.ent_name_list = name_list_dict['ent_name_list']
        self.rel_name_list = name_list_dict['rel_name_list']
        self.time_name_list = name_list_dict['id_time_list']
        self.ent_token_ids_in_trie = prefix_trie_dict['ent_token_ids_in_trie']
        self.tokenizer = tokenizer
        self.mode = mode

    def __len__(self):
        return len(self.test_triples)

    def __getitem__(self, index):
        test_triple = self.test_triples[index]
        if self.configs.temporal:
            head, tail, rel, time = test_triple
            head_name, tail_name, rel_name, time_name = self.original_ent_name_list[head], self.original_ent_name_list[tail], self.rel_name_list[rel], self.time_name_list[time]
            if self.mode == 'tail':
                src = head_name + ' | ' + rel_name + ' | ' + '<extra_id_0>' + ' | ' + time_name
                tgt_ids = tail
            else:
                src = '<extra_id_0>' + ' | ' + rel_name + ' | ' + tail_name + ' | ' + time_name
                tgt_ids = head
        else:
            head, tail, rel = test_triple
            head_name, tail_name, rel_name = self.original_ent_name_list[head], self.original_ent_name_list[tail], self.rel_name_list[rel]

            if self.mode == 'tail':
                src = head_name + ' | ' + rel_name + ' | ' + '<extra_id_0>' + ' | '
                tgt_ids = tail
            else:
                src = '<extra_id_0>' + ' | ' + rel_name + ' | ' + tail_name + ' | '
                tgt_ids = head

        tokenized_src = self.tokenizer(src, max_length=self.configs.src_max_length, truncation=True)
        source_ids = tokenized_src.input_ids
        source_mask = tokenized_src.attention_mask
        source_names = src
        target_names = self.ent_name_list[tgt_ids]

        # ent_rel = test_triple[[0, 2]] if self.mode == 'tail' else test_triple[[1, 2]]
        ent_rel = torch.LongTensor([head, rel]) if self.mode == 'tail' else torch.LongTensor([tail, rel])
        out = {
            'source_ids': source_ids,
            'source_mask': source_mask,
            'source_names': source_names,
            'tgt_ids': tgt_ids,
            'target_names': target_names,
            'test_triple': test_triple,
            'ent_rel': ent_rel
        }
        return out

    def collate_fn(self, data):
        agg_data = dict()
        agg_data['source_ids'] = batchify(data, 'source_ids', padding_value=0)
        agg_data['source_mask'] = batchify(data, 'source_mask', padding_value=0)
        agg_data['source_names'] = [dt['source_names'] for dt in data]
        agg_data['target_names'] = [dt['target_names'] for dt in data]
        agg_data['tgt_ids'] = [dt['tgt_ids'] for dt in data]
        agg_data['test_triple'] = batchify(data, 'test_triple', return_list=True)
        agg_data['ent_rel'] = batchify(data, 'ent_rel')
        return agg_data


class GeneratorDataModule(pl.LightningDataModule):
    def __init__(self, configs, train_triples, valid_triples, test_triples, name_list_dict, prefix_trie_dict, running_model='train_model'):
        super().__init__()
        self.configs = configs
        self.train_triples = train_triples
        self.valid_triples = valid_triples
        self.test_triples = test_triples
        # ent_name_list, rel_name_list .type: list
        self.name_list_dict = name_list_dict
        self.prefix_trie_dict = prefix_trie_dict
        self.running_model = running_model

        self.tokenizer = T5Tokenizer.from_pretrained(configs.pretrained_model)
        self.train_both = None
        self.valid_tail, self.valid_head = None, None
        self.test_tail, self.test_head = None, None

    def prepare_data(self):
        self.train_both = Generator_TrainDataset(self.configs, self.tokenizer, self.train_triples, self.name_list_dict, self.prefix_trie_dict)
        self.valid_tail = Generator_TestDataset(self.configs, self.tokenizer, self.valid_triples, self.name_list_dict, self.prefix_trie_dict, 'tail')
        self.valid_head = Generator_TestDataset(self.configs, self.tokenizer, self.valid_triples, self.name_list_dict, self.prefix_trie_dict, 'head')
        self.test_tail = Generator_TestDataset(self.configs, self.tokenizer, self.test_triples, self.name_list_dict, self.prefix_trie_dict, 'tail')
        self.test_head = Generator_TestDataset(self.configs, self.tokenizer, self.test_triples, self.name_list_dict, self.prefix_trie_dict, 'head')

    def train_dataloader(self):

        train_loader = DataLoader(self.train_both,
                                  batch_size=self.configs.batch_size,
                                  shuffle=True,
                                  collate_fn=self.train_both.collate_fn,
                                  pin_memory=True,
                                  num_workers=self.configs.num_workers)
        return train_loader

    def val_dataloader(self):
        valid_tail_loader = DataLoader(self.valid_tail,
                                       batch_size=self.configs.val_batch_size,
                                       shuffle=False,
                                       collate_fn=self.valid_tail.collate_fn,
                                       pin_memory=True,
                                       num_workers=self.configs.num_workers)
        valid_head_loader = DataLoader(self.valid_head,
                                       batch_size=self.configs.val_batch_size,
                                       shuffle=False,
                                       collate_fn=self.valid_head.collate_fn,
                                       pin_memory=True,
                                       num_workers=self.configs.num_workers)
        return [valid_tail_loader, valid_head_loader]
        # return [valid_head_loader, valid_tail_loader]

    def test_dataloader(self):
        test_tail_loader = DataLoader(self.test_tail,
                                      batch_size=self.configs.val_batch_size,
                                      shuffle=False,
                                      collate_fn=self.test_tail.collate_fn,
                                      pin_memory=True,
                                      num_workers=self.configs.num_workers)
        test_head_loader = DataLoader(self.test_head,
                                      batch_size=self.configs.val_batch_size,
                                      shuffle=False,
                                      collate_fn=self.test_head.collate_fn,
                                      pin_memory=True,
                                      num_workers=self.configs.num_workers)
        return [test_tail_loader, test_head_loader]
        # return [test_head_loader, test_tail_loader]

class Evaluator_TrainDataset(Dataset):
    def __init__(self, configs, tokenizer, trainset):
        self.configs = configs
        self.trainset = trainset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.trainset)

    def __getitem__(self, index):
        triple = self.trainset[index]
        out = {
            'triple': triple,
        }
        return out

    def collate_fn(self, data):
        agg_data = dict()
        agg_data['triple'] = [dt['triple'] for dt in data]
        return agg_data

class Evaluator_TestDataset(Dataset):
    def __init__(self, configs, tokenizer, testset):  # mode: {tail, head}
        self.configs = configs
        self.testset = testset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.testset)

    def __getitem__(self, index):
        triple = self.testset[index]
        out = {
            'triple': triple,
        }
        return out

    def collate_fn(self, data):
        agg_data = dict()
        # agg_data['input_ids'] = batchify(data, 'input_ids', padding_value=0)
        # agg_data['input_mask'] = batchify(data, 'input_mask', padding_value=0)
        # agg_data['query_text'] = [dt['query_text'] for dt in data]
        agg_data['triple'] = [dt['triple'] for dt in data]
        return agg_data

class EvaluatorDataModule(pl.LightningDataModule):
    def __init__(self, configs, trainset, validset, testset):
        super().__init__()
        self.configs = configs
        self.trainset = trainset
        self.validset = validset
        self.testset = testset

        self.tokenizer = T5Tokenizer.from_pretrained(configs.pretrained_model)
        self.train_both = None
        self.valid_tail, self.valid_head = None, None
        self.test_tail, self.test_head = None, None

    def prepare_data(self):
        self.train_both = Evaluator_TrainDataset(self.configs, self.tokenizer, self.trainset)
        self.valid_both = Evaluator_TestDataset(self.configs, self.tokenizer, self.validset)
        self.test_both = Evaluator_TestDataset(self.configs, self.tokenizer, self.testset)

    def train_dataloader(self):
        train_loader = DataLoader(self.train_both,
                                  batch_size=self.configs.batch_size,
                                  collate_fn=self.train_both.collate_fn,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=self.configs.num_workers)
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(self.valid_both,
                                  batch_size=self.configs.val_batch_size,
                                  collate_fn=self.valid_both.collate_fn,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=self.configs.num_workers)
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_both,
                                 batch_size=self.configs.val_batch_size,
                                 collate_fn=self.test_both.collate_fn,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=self.configs.num_workers)
        return test_loader


