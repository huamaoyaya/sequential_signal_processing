import io
import os.path
import torch
from torchtext.data import Field, Example
import pickle
path = os.path.join('.\\', 'fg.py')


def test_path():
    if os.path.exists(path):
        if os.path.isdir(path):
            print(f"文件夹 '{path}' 存在。")
        else:
            print(f"路径 '{path}' 存在，但不是一个文件夹。")
    else:
        print(f"文件夹 '{path}' 不存在。")
    return 0


def test_example():
    pass


def str_strip():
    string_1 = 'I love you'
    list_str = string_1.split(' ')
    return list_str


def test_list():
    a, b = [1, 2, 3]
    return a, b


def test_os():
    print(os.listdir(os.path.join('../.data/fatigue')))
    return 0


def test_example():
    path = '../fatigue_data'
    data_dirname = path + '/train.pt'
    writer = io.open(data_dirname,'wb')
    PSD_U1 = torch.tensor([0.1, 0.2, 0.3])
    Corr_um = torch.tensor([0.3])
    sequence_field = Field(sequential=True, use_vocab=False, dtype=torch.float)
    constant_field = Field(sequential=False, use_vocab=False, dtype=torch.float)
    fields_example = [('PSD_u1', sequence_field), ('label', constant_field)]
    example = Example.fromlist([list([PSD_U1]),list(Corr_um)],fields_example)
    pickle.dump(example, writer)
    print('写入成功')
    writer.close()
    return data_dirname
if __name__ == '__main__':
    test_os()


