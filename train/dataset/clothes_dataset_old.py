import os
import sys
import math
import json 
import torch
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from io import BytesIO
from dataloader import KVReader
from dataset.utils import chunk, pre_caption, worker_init_fn, KVSampler
import time
from torch.utils.data import DataLoader

def decode_tfrecord(raw_data, img_res=224):
    # import ipdb;ipdb.set_trace()
    context_features = {
        "title": tf.io.VarLenFeature(tf.string),
        "label_Style": tf.io.FixedLenFeature([], tf.int64, default_value=None),
        "label_Pattern_Type": tf.io.FixedLenFeature([], tf.int64, default_value=None),
        "label_Details": tf.io.FixedLenFeature([], tf.int64, default_value=None),
        "label_Type": tf.io.FixedLenFeature([], tf.int64, default_value=None),
        "label_Neckline": tf.io.FixedLenFeature([], tf.int64, default_value=None),
        "label_Sleeve_Length": tf.io.FixedLenFeature([], tf.int64, default_value=None),
        "label_Sleeve_Type": tf.io.FixedLenFeature([], tf.int64, default_value=None),
        "label_Waist_Line": tf.io.FixedLenFeature([], tf.int64, default_value=None),
        "label_Hem_Shaped": tf.io.FixedLenFeature([], tf.int64, default_value=None),
        "label_Length": tf.io.FixedLenFeature([], tf.int64, default_value=None),
        "label_Fit_Type": tf.io.FixedLenFeature([], tf.int64, default_value=None),
        "label_Fabric": tf.io.FixedLenFeature([], tf.int64, default_value=None),
        "label_Material": tf.io.FixedLenFeature([], tf.int64, default_value=None),
        "label_Season": tf.io.FixedLenFeature([], tf.int64, default_value=None),
        "label_Bra_Type": tf.io.FixedLenFeature([], tf.int64, default_value=None),
        "label_Bottom_Type": tf.io.FixedLenFeature([], tf.int64, default_value=None),
        "label_Top_Type": tf.io.FixedLenFeature([], tf.int64, default_value=None),
        "label_category_2": tf.io.FixedLenFeature([], tf.int64, default_value=None),
        "frame_binary": tf.io.VarLenFeature(tf.string),
        }



    #     "vid": tf.io.FixedLenFeature([], tf.string, default_value=None),
    #     "item_id": tf.io.FixedLenFeature([], tf.string, default_value=None),
    #     "country_code": tf.io.FixedLenFeature([], tf.string, default_value=None),
    #     "title": tf.io.FixedLenFeature([], tf.string, default_value=None),
    #     "ocr": tf.io.FixedLenFeature([], tf.string, default_value=None),
    #     "label": tf.io.FixedLenFeature([], tf.int64, default_value=None),
    #     "frame_binary": tf.io.VarLenFeature(tf.string),
    #     "frame_number": tf.io.FixedLenFeature([], tf.int64, default_value=None),
    #     "frame_width": tf.io.FixedLenFeature([], tf.int64, default_value=None),
    #     "frame_height": tf.io.FixedLenFeature([], tf.int64, default_value=None),
    # }
    sequences = {}
    contexts = tf.io.parse_single_example(raw_data, features=context_features)
    features = dict(**contexts, **sequences)
    frames = []
    frame_binary = tf.sparse.to_dense(tf.sparse.reorder(contexts['frame_binary']))
    title = tf.sparse.to_dense(tf.sparse.reorder(contexts['title'])).numpy()[0].decode('utf-8')

    for idx, item in enumerate(frame_binary):
        try:
            frame_this = tf.image.decode_jpeg(item, channels=3, dct_method='INTEGER_ACCURATE').numpy()[..., ::-1]
            frame_this = Image.fromarray(frame_this, "RGB")
            caption = "RightDesc"
        except Exception as e:
            print('error: ', e)
            caption = "WrongDesc"
            frame_this = Image.fromarray(np.zeros((img_res, img_res, 3)).astype(np.int8), "RGB")
        frames.append(frame_this)
    sample = {'images': frames, 'title': title}
    
    return sample


class ClothesDataset(torch.utils.data.Dataset):
    platforms = ['AE', 'TF']
    cat_seperate = '#'
    cat_inner_seperate = ':'
    platform_seperate = '-'
    tag_list = ['NonConforming', 'Prohibit', 'Sensitive', 'Special']

    def __init__(self,
                 data_path,
                 mode='train',
                 num_readers=16,
                 transform=None,
                 max_words=40,
                 wrong_tag='WrongImg',
                 img_res=224,
                 ):
        self.transform = transform
        self.data_path = data_path  # 数据存储路径
        self.num_readers = num_readers
        self.mode = mode

        self.max_words = max_words
        self.wrong_tag = wrong_tag  # 读取失败的图打上mask
        self.img_res = img_res

        self._init_data()

    def _init_data(self):
        self.reader = KVReader(self.data_path)
        self.raw_keys = self.reader.list_keys()
        # self.raw_keys = self.raw_keys[:256] # for debug
        print('raw_keys: ', len(self.raw_keys))
        #del reader

    def __len__(self):
        return len(self.raw_keys)

    def get_raw_item(self, i):
        # i = 10 # for debug
        k = self.raw_keys[i]
        value = self.reader.read_many([k])
        value = value[0]
        value = decode_tfrecord(value, img_res=self.img_res)
        image = value['images'][0] # curretnly first image
        title = value['title']
        return image, title

    def __getitem__(self, index):
        if isinstance(index, list):
            images, texts = [], []
            for idx in index:
                image, text = self.get_raw_item(idx)

                # apply transformation
                if self.transform:
                    image = self.transform(image)
            
                # caption
                text = pre_caption(text, self.max_words)
                images.append(image)
                texts.append(text)
            
            return np.stack(images), texts
        else:
            image, text = self.get_raw_item(index)
            # apply transformation
            if self.transform:
                image = self.transform(image)
            
            # caption
            text = pre_caption(text, self.max_words)
            return np.stack([image]), [text]


def create_dataloader(datasets, samplers, batch_size, num_workers, is_trains=[True], collate_fns=[None]):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = False
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
            # worker_init_fn=worker_init_fn,
        )
        loaders.append(loader)
    return loaders

if __name__ == "__main__":
    trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=torch.tensor(0),
        std=torch.tensor(0.5)
        )
    ])
    dataset = ClothesDataset(
        data_path="RuntimeDataset/final_shein_cloth/test",
        transform=trans
    )
    print(len(dataset))
    sampler = KVSampler(dataset=dataset, batch_size=128)
    data_loader = create_dataloader(
        datasets=[dataset],
        samplers=[sampler],
        batch_size=[None],
        num_workers=[0],
    )[0]

    for batch_idx, batch in enumerate(data_loader):
        images, texts = batch
        print(images.shape)
        print(texts[0])



