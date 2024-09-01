import random
from collections import Counter, OrderedDict
import glob
import io
import os
import pickle
import torch
from torchtext import data as dt
from torchtext.data import Field, Example
from tqdm import tqdm

from dataset import common

sequence_field = Field(sequential=True, use_vocab=False, dtype=torch.float)
constant_field = Field(sequential=False, use_vocab=False, dtype=torch.float)
fields_example = [(
    'PSD_u1', sequence_field
), ('PSD_u2', sequence_field), ('corr_clm', constant_field), ('F_corr_clm', constant_field)]


# pylint: disable=arguments-differ
# 转为example
def load_examples(paths, fields, data_dir, mode, filter_pred, num_shard):
    data_path_dir = data_dir + '/fatigue/' + mode + '-{}.pt'
    data_paths = [data_path_dir.format(i) for i in range(num_shard)]
    os.makedirs(os.path.dirname(data_paths[0]), exist_ok=True)
    writers = [open(data_path, 'wb') for data_path in data_paths]
    shard = 0
    for path in paths:
        print("processing {}".format(path))
        with io.open(path, mode='r', encoding='utf-8') as trg_file:
            lines = [line.strip().split(',') for line in trg_file if line.strip()]
            PSD_u1, PSD_u2, corr_clm, F_corr_clm = zip(
                *[(cors[0].strip(), cors[1].strip(), cors[2].strip()[0], cors[3].strip()[0]) for cors in lines if
                  len(cors) == 4])
        example = Example.fromlist([list(PSD_u1), list(PSD_u2), list(corr_clm), list(F_corr_clm)], fields)
        pickle.dump(example, writers[shard])
        shard = (shard + 1) % num_shard
    for writer in writers:
        writer.close()
    common.pickles_to_torch(data_paths)
    examples = torch.load(data_paths[0])
    return examples, data_paths, num_shard


# 划分强相关和弱相关
def split_str_corr_wea_corr(paths, data_dir):
    data_path_str = data_dir + '/fatigue' + '/str_corr.pt'
    data_path_wea = data_dir + '/fatigue' + '/wea_corr.pt'
    os.makedirs(os.path.dirname(data_dir), exist_ok=True)
    writer_str = open(data_path_str, 'wb')
    writer_wea = open(data_path_wea, 'wb')
    examples_pt_list = [torch.load(path) for path in paths]
    examples = [example for examples_pt in examples_pt_list for example in examples_pt]
    examples_str_corr = [example for example in examples if float(abs(example.corr_clm)) > 0.5]
    examples_wea_corr = [example for example in examples if float(abs(example.corr_clm)) < 0.5]
    for examples_str in examples_str_corr: pickle.dump(examples_str, writer_str)
    for examples_wea in examples_wea_corr: pickle.dump(examples_wea, writer_str)
    writer_str.close()
    writer_wea.close()
    common.pickles_to_torch(data_path_str)
    common.pickles_to_torch(data_path_wea)
    print('强相关和弱相关划分完成')
    return data_path_str, data_path_wea


# 划分训练集和测试集

def split_train_test(path, train_ratio):
    file_name = os.path.splitext(os.path.basename(path))[0]
    train_example_prfix = file_name
    val_example_prefix = file_name
    data_examples = torch.load(path)
    if not isinstance(data_examples, list):
        raise ValueError("The loaded error must be a list")
    random.shuffle(data_examples)
    num_train = int(train_ratio * len(data_examples))
    train_examples = data_examples[:num_train]
    val_examples = data_examples[num_train:]
    train_file_path = os.path.join(os.path.dirname(path), f'{train_example_prfix}_train_example.pt')
    example_file_path = os.path.join(os.path.dirname(path), f'{val_example_prefix}_val_example.pt')
    torch.save(train_examples, train_file_path)
    torch.save(val_examples, example_file_path)
    if os.path.exists(train_file_path):
        print('训练集，测试集划分成功')
    return train_examples, val_examples, train_file_path, example_file_path


class Fatigue_data(dt.Dataset):
    urls = ["http://www.statmt.org/lm-benchmark/"
            "1-billion-word-language-modeling-benchmark-r13output.tar.gz"]
    name = 'fatigue'
    dirname = ''

    @staticmethod
    def sort_key(ex):
        return len(ex.trg)

    @classmethod
    def custom_splits(cls,train_ratio, fields, data_dir, root='.data', **kwargs):
        filter_pred = kwargs['filter_pred']
        expected_dir = os.path.join(root, cls.name)
        path = (expected_dir if os.path.exists(expected_dir)
                else cls.download(root))
        fatigue_files = [
            os.path.join(path, f'dataset{i}', f'data_{k}.csv') for i in range(1, 4) for k in range(1, len(os.listdir(
                os.path.join(path, f'dataset{i}')
            )) + 1)
        ]
        _, example_paths, num_shard = load_examples(fatigue_files, fields, data_dir, 'all_data_', filter_pred, 2)
        data_path_str, data_path_wea = split_str_corr_wea_corr(example_paths, data_dir)
        str_train_examples, str_val_examples, str_train_file_path, str_example_file_path = \
            split_train_test(data_path_str, train_ratio=train_ratio)
        train_data_str = cls(str_train_examples, fields, **kwargs)
        val_data_str = cls(str_val_examples, fields, **kwargs)
        return train_data_str, val_data_str, str_train_file_path, str_example_file_path

def prepare_signal():




if __name__ == '__main__':
    pass
