from tensorflow.keras.utils import Sequence, to_categorical
from pathlib import Path
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pathlib
import torch
import torch.utils.data as data
from torchvision import transforms
import pathlib
import pandas as pd

DATA_DIR = '/home/higuchi/ssd/kits19/data'
train_patch = 'tumor_48x48x16'
val_patch = 'tumor_60x60x20'
train_ids = ['001', '002']
val_ids = ['001', '002']
class DataPathMaker():
    '''
    DataSetに渡すpath_listを作るためのDataFrameを作る
    今後の展望として、統計量を持ったDFを渡してその条件でlistを変えるようにする。
    '''
    # TODO:文字列の除去

    def __init__(self, data_dir, patch_dir_name='patch'):
        self.data_dir = pathlib.Path(data_dir)
        self.patch_dir_name = patch_dir_name

    def create_dataframe(self, id_list):
        data = []
        for patient_id in id_list:
            # TODO: case_00の部分もyamlから渡せるようにしたほうがよい
            patient_dir = self.data_dir / f'case_00{patient_id}' / self.patch_dir_name
            images = sorted(patient_dir.glob('patch_image_*.npy'))
            labels = sorted(patient_dir.glob('patch_no_onehot_*.npy'))
            if len(images) == 1 or len(labels) == 0:
                print(f'{patient_id} is no data')
            for image, label in zip(images, labels):
                data.append(['image', patient_id, image])
                data.append(['label', patient_id, label])
        return pd.DataFrame(data, columns=['type', 'id', 'path'])


train_path_df = DataPathMaker(DATA_DIR, patch_dir_name=train_patch).create_dataframe(train_ids)
val_path_df = DataPathMaker(DATA_DIR, patch_dir_name=val_patch).create_dataframe(val_ids)

train_im_list = train_path_df[train_path_df['type'] == 'image']['path'].astype(str).values
val_im_list = train_path_df[train_path_df['type'] == 'image']['path'].astype(str).values

train_lb_list = train_path_df[train_path_df['type'] == 'label']['path'].astype(str).values
val_lb_list = train_path_df[train_path_df['type'] == 'label']['path'].astype(str).values


class KitsDataSet(data.Dataset):
    '''
    loadした後のデータの処理のみを行う。
    input:train,val,testのdata_list(絞り込み済み) & label_list
    '''
    # TODO: ラベルが重ねってる部分の処理(binaly_labels)
    # TODO: Augment_code

    def __init__(self, img_list, label_list, transform, phase='train'):
        self.img_list = img_list
        self.label_list = label_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        im = np.load(self.img_list[index])
        lb = np.load(self.label_list[index])

        if self.transform:
            im, lb = self.transform(im, lb, self.phase)

        # B,D,C,H,Wに変換する
        # im = im.permute(3, 2, 0, 1)
        # im = np.transpose(im, (2, 3, 0, 1))
        im = np.transpose(im, (3, 2, 0, 1))

        # B,C,H,Wに変換する
        # lb = lb.permute(2, 0, 1)

        # numpy ver
        lb = to_categorical(lb, num_classes=3)
        # lb = np.transpose(lb, (2, 3, 0, 1))
        lb = np.transpose(lb, (3, 2, 0, 1))

        return im, lb


tr_DS = KitsDataSet(train_im_list, train_lb_list, phase='train', transform=None)
val_DS = KitsDataSet(val_im_list, val_lb_list, phase='val', transform=None)


print(tr_DS.__getitem__(0)[1].shape)
