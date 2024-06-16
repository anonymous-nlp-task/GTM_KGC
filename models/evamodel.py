import os
import re
import sqlite3
import random
import pickle
import networkx as nx
import numpy as np
import torch
import time
from multiprocessing.pool import ThreadPool
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score, accuracy_score, recall_score
from models.modified_model.modified_T5 import ModifiedT5ForConditionalGeneration
from transformers.optimization import Adafactor
class EvaluatorFinetuner(pl.LightningModule):
    def __init__(self, configs, tokenizer, ent_name_list,  rel_name_list, time_name_list, tailseed, headseed, cuda):
        super().__init__()
        self.save_hyperparameters()
        self.configs = configs
        self.tokenizer = tokenizer
        self.ent_name_list = ent_name_list
        self.rel_name_list = rel_name_list
        self.time_name_list = time_name_list
        self.T5ForConditionalGeneration = ModifiedT5ForConditionalGeneration.from_pretrained(configs.pretrained_model)
        self.history = {'perf': ..., 'loss': []}
        self.conn = sqlite3.connect(configs.save_dir + '/store.db')
        self.cursor = self.conn.cursor()
        self.tailseed = tailseed
        self.headseed = headseed
        self.cuda = cuda

    def collate_fn(self, data):
        agg_data = dict()
        agg_data['source_ids'] = self.batchify(data, 'source_ids', padding_value=0)
        agg_data['source_mask'] = self.batchify(data, 'source_mask', padding_value=0)
        agg_data['target_ids'] = self.batchify(data, 'target_ids', padding_value=0)
        agg_data['target_mask'] = self.batchify(data, 'target_mask', padding_value=0)
        return agg_data

    def batchify(self, output_dict, key, padding_value=None, return_list=False):
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


    def prompt(self, tup, pathset, mode):
        '''
        Construct prompt
        '''
        if mode == 'tail':
            qh = self.ent_name_list[tup[0]]
            qr = self.rel_name_list[tup[2]]
            input_text = 'predict tail: '
            if self.configs.temporal:
                qt = self.time_name_list[tup[3]]
                input_text += '({h}, {r}, <extra_id_0>, {t}) | '.format(h=qh, r=qr, t=qt)
            else:
                input_text += '({h}, {r}, <extra_id_0>) | '.format(h=qh, r=qr)
        else:
            qo = self.ent_name_list[tup[1]]
            qr = self.rel_name_list[tup[2]]
            input_text = 'predict head: '
            if self.configs.temporal:
                qt = self.time_name_list[tup[3]]
                input_text += '(<extra_id_0>, {r}, {o}, {t}) | '.format(o=qo, r=qr, t=qt)
            else:
                input_text += '(<extra_id_0>, {r}, {o}) | '.format(o=qo, r=qr)

        for idx, tup in enumerate(pathset):
            if tup[2] >= self.configs.n_rel:
                o = self.ent_name_list[int(tup[0])]
                h = self.ent_name_list[int(tup[1])]
                r = self.rel_name_list[tup[2] - self.configs.n_rel]
            else:
                h = self.ent_name_list[int(tup[0])]
                o = self.ent_name_list[int(tup[1])]
                r = self.rel_name_list[tup[2]]

            if self.configs.temporal:
                t = self.time_name_list[int(tup[3])]
                input_text += '({h}, {r}, {o}, {t})'.format(h=h, r=r, o=o, t=t)

            else:
                input_text += '({h}, {r}, {o})'.format(h=h, r=r, o=o)

            if idx + 1 < len(pathset):
                input_text += ', '
            else:
                input_text += '.'
                # if self.configs.path_style == 'ent_rel':
                #     input_text += ', {o}.'.format(o=o)
                # else:
                #     input_text += '.'
        return input_text

    def bridge_prompt(self, tup, pathset, mode):
        '''
        Construct bridge prompt
        '''
        if mode == 'tail':
            qh = self.ent_name_list[tup[0]]
            qr = self.rel_name_list[tup[2]]
            input_text = 'predict tail: '
            if self.configs.temporal:
                qt = self.time_name_list[tup[3]]
                input_text += '({h}, {r}, <extra_id_0>, {t}) | '.format(h=qh, r=qr, t=qt)
            else:
                input_text += '({h}, {r}, <extra_id_0>) | '.format(h=qh, r=qr)
        else:
            qo = self.ent_name_list[tup[1]]
            qr = self.rel_name_list[tup[2]]
            input_text = 'predict head: '
            if self.configs.temporal:
                qt = self.time_name_list[tup[3]]
                input_text += '(<extra_id_0>, {r}, {o}, {t}) | '.format(o=qo, r=qr, t=qt)
            else:
                input_text += '(<extra_id_0>, {r}, {o}) | '.format(o=qo, r=qr)

        for idx, tup in enumerate(pathset):
            if tup[2] >= self.configs.n_rel:
                o = self.ent_name_list[int(tup[0])]
                h = self.ent_name_list[int(tup[1])]
                r = self.rel_name_list[tup[2] - self.configs.n_rel]
            else:
                h = self.ent_name_list[int(tup[0])]
                o = self.ent_name_list[int(tup[1])]
                r = self.rel_name_list[tup[2]]
            if idx == 0:
                input_text += '{h}, '.format(h=h)
            if self.configs.temporal:
                t = self.time_name_list[int(tup[3])]
                input_text += '({r}, {t})'.format(r=r, t=t)
            else:
                input_text += '{r}'.format(r=r)

            if idx + 1 < len(pathset):
                input_text += ', '
            else:
                input_text += ', {o}.'.format(o=o)
        return input_text

    def postive_sampling(self, tup, path_list, seed_list, out, mode):
        if len(path_list) > 1:
            index = np.array([_ for _ in range(len(path_list))])
            weights = seed_list / np.sum(seed_list)
            sampling_idx = np.random.choice(a=index, size=1, p=weights, replace=False)[0]
            positive_path = path_list[sampling_idx]
            if seed_list[sampling_idx] > 0:
                seed_list[sampling_idx] = seed_list[sampling_idx] - 1
            else:
                seed_list[sampling_idx] = 1
        else:
            positive_path = path_list[0]
        input_text = self.prompt(tup, positive_path, mode)

        target_text = '<extra_id_0>yes<extra_id_1>'
        tokenized_src = self.tokenizer(input_text, max_length=self.configs.src_max_length, truncation=True)
        source_ids = tokenized_src.input_ids
        source_mask = tokenized_src.attention_mask
        tokenized_tgt = self.tokenizer(target_text, max_length=self.configs.train_tgt_max_length, truncation=True)
        target_ids = tokenized_tgt.input_ids
        target_mask = tokenized_tgt.attention_mask
        out.append({'source_ids': source_ids, 'source_mask': source_mask, 'target_ids': target_ids,
                    'target_mask': target_mask})

        if self.configs.bridge: # generate bridge data
            input_text = self.bridge_prompt(tup, positive_path, mode)
            target_text = '<extra_id_0>yes<extra_id_1>'
            tokenized_src = self.tokenizer(input_text, max_length=self.configs.src_max_length, truncation=True)
            source_ids = tokenized_src.input_ids
            source_mask = tokenized_src.attention_mask
            tokenized_tgt = self.tokenizer(target_text, max_length=self.configs.train_tgt_max_length, truncation=True)
            target_ids = tokenized_tgt.input_ids
            target_mask = tokenized_tgt.attention_mask
            out.append({'source_ids': source_ids, 'source_mask': source_mask, 'target_ids': target_ids,
                        'target_mask': target_mask})
        return seed_list


    def negative_sampling(self, tup, path_list, seed_list, out, mode):
        if len(path_list) > 1:
            index = np.array([_ for _ in range(len(path_list))])
            weights = seed_list / np.sum(seed_list)
            sampling_idx = np.random.choice(a=index, size=1, p=weights, replace=False)[0]
            negative_path = path_list[sampling_idx]
            if seed_list[sampling_idx] > 0:
                seed_list[sampling_idx] = seed_list[sampling_idx] - 1
            else:
                seed_list[sampling_idx] = 1

        else:
            negative_path = path_list[0]

        input_text = self.prompt(tup, negative_path, mode)

        target_text = '<extra_id_0>no<extra_id_1>'
        tokenized_src = self.tokenizer(input_text, max_length=self.configs.src_max_length, truncation=True)
        source_ids = tokenized_src.input_ids
        source_mask = tokenized_src.attention_mask
        tokenized_tgt = self.tokenizer(target_text, max_length=self.configs.train_tgt_max_length, truncation=True)
        target_ids = tokenized_tgt.input_ids
        target_mask = tokenized_tgt.attention_mask
        out.append({'source_ids': source_ids, 'source_mask': source_mask, 'target_ids': target_ids,
                    'target_mask': target_mask})
        if self.configs.bridge:
            input_text = self.bridge_prompt(tup, negative_path, mode)
            target_text = '<extra_id_0>no<extra_id_1>'
            tokenized_src = self.tokenizer(input_text, max_length=self.configs.src_max_length, truncation=True)
            source_ids = tokenized_src.input_ids
            source_mask = tokenized_src.attention_mask
            tokenized_tgt = self.tokenizer(target_text, max_length=self.configs.train_tgt_max_length, truncation=True)
            target_ids = tokenized_tgt.input_ids
            target_mask = tokenized_tgt.attention_mask
            out.append({'source_ids': source_ids, 'source_mask': source_mask, 'target_ids': target_ids,
                        'target_mask': target_mask})
        return seed_list


    def training_step(self, batched_data, batch_idx):
        # src_ids, src_mask: .shape: (batch_size, padded_seq_len)
        # input_ids = batched_data['input_ids']
        # tuples = self.tokenizer.batch_decode([tokens for tokens in input_ids])
        triples = batched_data['triple']
        out = []
        for tupdict in triples:
            tup = tupdict['query']
            if tupdict['mode'] == 'tail':
                cursor = self.cursor.execute("SELECT * FROM sampling_tail where query=?", (tup,))
                poset, ngset = [], []
                for row in cursor:
                    poset, ngset = eval(row[2]), eval(row[3])
                poidx = self.tailseed[tup]['pos']
                ngidx = self.tailseed[tup]['neg']

                seed_list = self.postive_sampling(eval(tup), poset, poidx, out, tupdict['mode'])
                self.tailseed[tup]['pos'] = seed_list

                seed_list = self.negative_sampling(eval(tup), ngset, ngidx, out, tupdict['mode'])
                self.tailseed[tup]['neg'] = seed_list
            else:
                cursor = self.cursor.execute("SELECT * FROM sampling_head where query=?", (tup,))
                poset, ngset = [], []
                for row in cursor:
                    poset, ngset = eval(row[2]), eval(row[3])
                poidx = self.headseed[tup]['pos']
                ngidx = self.headseed[tup]['neg']

                seed_list = self.postive_sampling(eval(tup), poset, poidx, out, tupdict['mode'])
                self.headseed[tup]['pos'] = seed_list

                seed_list = self.negative_sampling(eval(tup), ngset, ngidx, out, tupdict['mode'])
                self.headseed[tup]['neg'] = seed_list

        agg_data = self.collate_fn(out)

        # target_ids, target_mask, labels: .shape: (batch_size, padded_seq_len)
        source_ids = agg_data['source_ids'].to(self.cuda)
        source_mask = agg_data['source_mask'].to(self.cuda)
        target_ids = agg_data['target_ids'].to(self.cuda)
        labels = target_ids.clone()
        labels[labels[:, :] == self.trainer.datamodule.tokenizer.pad_token_id] = -100

        # ent_rel .shape: (batch_size, 2)
        source_emb = self.T5ForConditionalGeneration.encoder.embed_tokens(source_ids)
        # batch_size, seq_len, model_dim = inputs_emb.shape
        output = self.T5ForConditionalGeneration(inputs_embeds=source_emb, attention_mask=source_mask, labels=labels,
                                                 output_hidden_states=True)
        if self.configs.learning_strategy == 1:
            # 1：Use only generated loss
            loss = torch.mean(output.loss)
        elif self.configs.learning_strategy == 2:
            # 2：Logarithmic contrast loss
            gloss = torch.mean(output.loss)
            logits = output.logits
            probs = F.softmax(logits, dim=-1)
            yes_token_id = self.tokenizer.encode("yes")[0]
            yes_score = probs[:, 1, yes_token_id]
            yprob = yes_score[torch.where(labels[:, 1] == 4273)]  # 4273->yes
            nprob = yes_score[torch.where(labels[:, 1] != 4273)]  # 150->no
            yloss = 0
            count = 0
            for npb in nprob:
                for ypb in yprob:
                    count += 1
                    yloss += torch.log(1 + torch.exp((npb - ypb) * self.configs.lamda))
            yloss = yloss / count
            loss = yloss + gloss
        else:
            # 3：Max contrast loss
            gloss = torch.mean(output.loss)
            logits = output.logits
            probs = F.softmax(logits, dim=-1)
            yes_token_id = self.tokenizer.encode("yes")[0]
            yes_score = probs[:, 1, yes_token_id]
            yprob = yes_score[torch.where(labels[:, 1] == 4273)]  # 4273->yes
            nprob = yes_score[torch.where(labels[:, 1] != 4273)]  # 150->no
            yloss = 0
            count = 0
            for npb in nprob:
                for ypb in yprob:
                    count += 1
                    margin = 1
                    yloss = torch.max(torch.tensor([0, npb - ypb + margin]))
            yloss = yloss / count

            loss = yloss + gloss
        self.history['loss'].append(loss.detach().item())
        self.log('val_loss', loss, on_step=True)
        return {'loss': loss}

    def validation_step(self, batched_data, batch_idx):
        if self.current_epoch < self.configs.skip_n_val_epoch:
            return
        triples = batched_data['triple']
        out = []
        for tupdict in triples:
            tup = tupdict['query']
            if tupdict['mode'] == 'tail':
                cursor = self.cursor.execute("SELECT * FROM sampling_tail where query=?", (tup,))
                poset, ngset = [], []
                for row in cursor:
                    poset, ngset = eval(row[2]), eval(row[3])
                poidx = self.tailseed[tup]['pos']
                ngidx = self.tailseed[tup]['neg']
                # postive sampling
                seed_list = self.postive_sampling(eval(tup), poset, poidx, out, tupdict['mode'])
                self.tailseed[tup]['pos'] = seed_list
                # negative sampling
                seed_list = self.negative_sampling(eval(tup), ngset, ngidx, out, tupdict['mode'])
                self.tailseed[tup]['neg'] = seed_list
            else:
                cursor = self.cursor.execute("SELECT * FROM sampling_head where query=?", (tup,))
                poset, ngset = [], []
                for row in cursor:
                    poset, ngset = eval(row[2]), eval(row[3])
                poidx = self.headseed[tup]['pos']
                ngidx = self.headseed[tup]['neg']
                # postive sampling
                seed_list = self.postive_sampling(eval(tup), poset, poidx, out, tupdict['mode'])
                self.headseed[tup]['pos'] = seed_list
                # negative sampling
                seed_list = self.negative_sampling(eval(tup), ngset, ngidx, out, tupdict['mode'])
                self.headseed[tup]['neg'] = seed_list

        agg_data = self.collate_fn(out)

        # target_ids, target_mask, labels: .shape: (batch_size, padded_seq_len)
        source_ids = agg_data['source_ids'].to(self.cuda)
        source_mask = agg_data['source_mask'].to(self.cuda)
        target_ids = agg_data['target_ids'].to(self.cuda)
        labels = target_ids.clone()
        labels[labels[:, :] == self.trainer.datamodule.tokenizer.pad_token_id] = -100

        # generated_text .type: list(str) .len: batch_size * num_beams
        generated_text, clue_label = self.decode(source_ids, source_mask, target_ids)
        y_true = []
        y_pred = []
        for idx in range(len(clue_label)):
            if clue_label[idx] == "yes":
                y_true.append(1)
            else:
                y_true.append(0)
        for idx in range(len(generated_text)):
            if generated_text[idx] == "yes":
                y_pred.append(1)
            else:
                y_pred.append(0)
        out = {'y_true': y_true, 'y_pred': y_pred}
        return out

    def decode(self, input_ids, input_mask, target_ids):
        def _extract(generated_text):
            compiler = re.compile(r'<extra_id_0>(.*)<extra_id_1>')
            extracted_text = []
            for text in generated_text:
                match = compiler.search(text)
                if match is None:
                    # text = text.strip().lstrip('<pad> <extra_id_0>')
                    extracted_text.append(text.strip())
                else:
                    extracted_text.append(match.group(1).strip())
            return extracted_text

        inputs_emb = self.T5ForConditionalGeneration.encoder.embed_tokens(input_ids)
        input_mask = input_mask

        outputs = self.T5ForConditionalGeneration.generate(inputs_embeds=inputs_emb,
                                                           attention_mask=input_mask,
                                                           max_length=self.configs.eval_tgt_max_length,
                                                           return_dict_in_generate=True,
                                                           output_scores=True,)

        # outputs = self.T5ForConditionalGeneration(inputs_embeds=inputs_emb, attention_mask=input_mask)

        raw_generated_text = self.trainer.datamodule.tokenizer.batch_decode(outputs['sequences'])
        generated_text = _extract(raw_generated_text)
        if self.configs.running_model == "test_model":
            raw_label = self.trainer.datamodule.tokenizer.batch_decode(target_ids)
            tgt_label = _extract(raw_label)
        else:
            scores = outputs['scores'][1]  # [seq_len, batch_size, num_labels]
            tgt_label = []
            for score in scores:
                probs = F.softmax(score, dim=-1)
                yes_token_id = self.tokenizer.encode("yes")[0]
                yes_score = probs[yes_token_id]
                tgt_label.append(yes_score.item())

        return generated_text, tgt_label


    def validation_epoch_end(self, outs):
        if self.current_epoch < self.configs.skip_n_val_epoch:
            return
        report_f1 = []
        report_acc = []
        for out in outs:
            f1 = f1_score(out['y_true'], out['y_pred'])
            acc = recall_score(out['y_true'], out['y_pred'])
            report_f1.append(f1)
            report_acc.append(acc)
        f1_value = np.sum(np.array(report_f1)) / len(report_f1)
        acc_value = np.sum(np.array(report_acc)) / len(report_acc)
        print("F1 score:{}".format(f1_value))
        print("Acc score:{}".format(acc_value))

    def test_step(self, batched_data, batch_idx):
        return self.validation_step(batched_data, batch_idx)

    def test_epoch_end(self, outs):
        self.validation_epoch_end(outs)

    def configure_optimizers(self):
        if self.configs.optim == 'Adafactor':
            optim = Adafactor(self.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=self.configs.lr)
        else:
            optim = torch.optim.Adam(self.parameters(), lr=self.configs.lr)
        return optim
