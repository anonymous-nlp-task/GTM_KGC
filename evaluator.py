import os
import sqlite3
import argparse
import os
import warnings
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import T5Tokenizer
from transformers import T5Config
from models.evamodel import EvaluatorFinetuner
from data import EvaluatorDataModule
from helper import get_num, read_name
from callbacks import PrintingCallback


def main():
    ## read triples
    conn = sqlite3.connect(configs.save_dir + '/store.db')
    cursor = conn.cursor()
    sql = "SELECT * FROM query_result where flag='train'"
    cursor = cursor.execute(sql)
    trainset, validset, testset = [], [], []
    tailseed, headseed = {}, {}
    for row in cursor:
        id, query, mode = row[0], row[1], row[2]
        trainset.append({"id": id, "query": query, "mode": mode})
    sql = "SELECT * FROM query_result where flag='test'"
    cursor = cursor.execute(sql)
    for row in cursor:
        id, query, mode = row[0], row[1], row[2]
        testset.append({"id": id, "query": query, "mode": mode})

    sql = "SELECT * FROM seed_tail"
    cursor = cursor.execute(sql)
    for row in cursor:
        query, pos, neg = row[1], row[2], row[3]
        tailseed[query] = {'pos': eval(pos), 'neg': eval(neg)}


    sql = "SELECT * FROM seed_head"
    cursor = cursor.execute(sql)
    for row in cursor:
        query, pos, neg = row[1], row[2], row[3]
        headseed[query] = {'pos': eval(pos), 'neg': eval(neg)}

    conn.close()


    validset = testset

    # all_input = train_input + test_input
    # all_target = train_target + test_target

    # construct name list  #
    original_ent_name_list, rel_name_list, id_time_list, time_id_list = read_name(configs.dataset_path, configs.dataset,
                                                                                  configs.temporal)
    tokenizer = T5Tokenizer.from_pretrained(configs.pretrained_model)
    ent_token_ids_in_trie = tokenizer(['<extra_id_0>' + ent_name + '<extra_id_1>' for ent_name in original_ent_name_list], max_length=configs.train_tgt_max_length, truncation=True).input_ids

    ent_name_list = tokenizer.batch_decode([tokens[1:-2] for tokens in ent_token_ids_in_trie])

    filename = 'lm-evaluator-{epoch:02d}-{val_loss:.4f}'

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=configs.save_dir,
        filename=filename,
        mode='min'
    )

    printing_callback = PrintingCallback()

    gpu = [int(configs.gpu)] if torch.cuda.is_available() else 0
    trainer_params = {
        'gpus': gpu,
        'limit_train_batches': 0.3,
        'max_epochs': configs.epochs,  # 1000
        'checkpoint_callback': True,  # True
        'logger': False,  # TensorBoardLogger
        'num_sanity_val_steps': 0,
        'check_val_every_n_epoch': 1,
        'enable_progress_bar': True,
        'callbacks': [
            printing_callback,
            checkpoint_callback
        ],
    }
    trainer = pl.Trainer(**trainer_params)


    kw_args = {
        'ent_name_list': ent_name_list,
        'rel_name_list': rel_name_list,
        'time_name_list': id_time_list,
        'tailseed': tailseed,
        'headseed': headseed,
        'cuda': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }


    if configs.model_path == '' and configs.running_model == 'train_model':

        datamodule = EvaluatorDataModule(configs, trainset, validset, testset)
        print('train_model datamodule construction done.', flush=True)
        model = EvaluatorFinetuner(configs, tokenizer, **kw_args)
        trainer.fit(model, datamodule)
        model_path = checkpoint_callback.best_model_path
        print('training best model path:', model_path, flush=True)

    else:
        model_path = configs.model_path
        model_name = configs.model_name
        datamodule = EvaluatorDataModule(configs, trainset, validset, testset)
        model = EvaluatorFinetuner.load_from_checkpoint(model_path + model_name, strict=False, configs=configs, **kw_args)
        trainer.test(model, dataloaders=datamodule)


if __name__ == '__main__':

    warnings.filterwarnings('ignore', category=DeprecationWarning)
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset_path', type=str, default='./data/processed')
    parser.add_argument('-dataset', dest='dataset', default='FB15k-237',
                        help='Dataset to use, WN18RR, FB15k-237, FB15k-237N, ICEWS14, NELL')
    parser.add_argument('-model', default='T5Finetuner', help='Model Name')
    parser.add_argument('-gpu', type=str, default='0', help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('-seed', dest='seed', default=41504, type=int, help='Seed for randomization')
    parser.add_argument('-num_workers', type=int, default=4, help='Number of processes to construct batches')
    parser.add_argument('-save_dir', type=str, default='', help='')

    parser.add_argument('-pretrained_model', type=str, default='./models/t5-base', help='')
    parser.add_argument('-batch_size', default=15, type=int, help='Batch size, WN18RR=24, FB15k-237=32, FB15k-237N=32, ICEWS14=32, NELL')
    parser.add_argument('-val_batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('-num_beams', default=40, type=int, help='Number of samples from beam search')
    parser.add_argument('-num_beam_groups', default=1, type=int, help='')
    parser.add_argument('-src_max_length', default=512, type=int, help='')
    parser.add_argument('-train_tgt_max_length', default=512, type=int, help='')
    parser.add_argument('-eval_tgt_max_length', default=512, type=int, help='')
    parser.add_argument('-epoch', dest='epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('-lr', type=float, default=0.0005, help='Starting Learning Rate')
    parser.add_argument('-model_path', dest='model_path', default='', help='The path for reloading models')

    parser.add_argument('-optim', default='Adam', type=str, help='')
    parser.add_argument('-bridge', action='store_true', default=False, help='')
    parser.add_argument('-learning_strategy', default=2, type=int, help='loss function style')
    parser.add_argument('-lamda', default=20, type=int, help='loss2: contrast learning parameters')
    parser.add_argument('-skip_n_val_epoch', default=1000, type=int, help='Using train process')
    # parser.add_argument('-skip_n_val_epoch', default=0, type=int, help='Using test process')
    parser.add_argument('-temporal', action='store_true', default=False, help='')
    parser.add_argument('-running_model', type=str, default='train_model', help='[train_model, test_model, predict_model]')
    configs = parser.parse_args()
    n_ent = get_num(configs.dataset_path, configs.dataset, 'entity')
    n_rel = get_num(configs.dataset_path, configs.dataset, 'relation')
    configs.n_ent = n_ent
    configs.n_rel = n_rel
    configs.vocab_size = T5Config.from_pretrained(configs.pretrained_model).vocab_size
    configs.model_dim = T5Config.from_pretrained(configs.pretrained_model).d_model
    if configs.save_dir == '' and configs.running_model == 'train_model':  # if train and valid, makedires else not makedires
        configs.save_dir = os.path.join('./checkpoint', configs.dataset + '-train_model')
        os.makedirs(configs.save_dir, exist_ok=True)
    else:
        configs.save_dir = os.path.join('./checkpoint', configs.dataset + '-train_model')

    pl.seed_everything(configs.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_printoptions(profile='full')
    main()

