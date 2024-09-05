import os
import sys
import math
import json
import torch
import numpy as np
import pandas as pd

# import tensorflow as tf
import multiprocessing as mp
from tqdm import tqdm
from torchvision import transforms
from io import BytesIO
from dataloader import KVReader
from dataset.utils import chunk, pre_caption, worker_init_fn, KVSampler
import time
from torch.utils.data import DataLoader
from PIL import Image


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
    frame_binary = tf.sparse.to_dense(tf.sparse.reorder(contexts["frame_binary"]))
    title = tf.sparse.to_dense(tf.sparse.reorder(contexts["title"])).numpy()[0].decode("utf-8")

    for idx, item in enumerate(frame_binary):
        try:
            frame_this = tf.image.decode_jpeg(item, channels=3, dct_method="INTEGER_ACCURATE").numpy()[..., ::-1]
            frame_this = Image.fromarray(frame_this, "RGB")
            caption = "RightDesc"
        except Exception as e:
            print("error: ", e)
            caption = "WrongDesc"
            frame_this = Image.fromarray(np.zeros((img_res, img_res, 3)).astype(np.int8), "RGB")
        frames.append(frame_this)
    sample = {"images": frames, "title": title}

    return sample


def chunk(iterable, chunk_size):
    ret = []
    for record in iterable:
        ret.append(record)
        if len(ret) == chunk_size:
            yield ret
            ret = []
    if ret:
        yield ret


def get_keys(args):
    return KVReader(*args).list_keys()


class ClothesKVDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        mode="train",
        num_readers=2,
        preprocess_train=None,
        max_words=None,
        img_res=224,
    ):
        self.preprocess_train = preprocess_train
        self.path = data_path  # 数据存储路径
        self.num_readers = num_readers
        self.mode = mode

        self.max_words = max_words
        self.img_res = img_res
        self.column_names = ["image", "text"]
        with mp.Pool(1) as p:
            self.raw_keys = p.map(get_keys, [(data_path, num_readers)])[0]
            # self.raw_keys = self.raw_keys[:256] # for debug
        # Uncomment the following lines if num_workers == 0
        # self.reader = KVReader(dataset.path, dataset.num_readers)
        print("raw_keys: ", len(self.raw_keys))
        print("raw_keys: ", self.raw_keys[:2])

    def with_transform(self, preprocess_train):
        self.preprocess_train = preprocess_train
        return self

    def __len__(self):
        return len(self.raw_keys)

    def get_raw_item(self, i):
        # i = 10 # for debug
        k = self.raw_keys[i]
        value = self.reader.read_many([k])
        value = value[0]
        value = decode_tfrecord(value, img_res=self.img_res)
        image = value["images"][0]  # curretnly first image
        text = value["title"]
        # print(image.height, image.width)
        # if not 0.74<=image.width/image.height<=0.76:
        #     print("*************", image.height, image.width, image.width/image.height, "*************", )
        #     print("*************", image.height, image.width, image.width/image.height, "*************", )
        #     print("*************", image.height, image.width, image.width/image.height, "*************", )
        #     print("*************", image.height, image.width, image.width/image.height, "*************", )
        return image, text

    def __getitem__(self, index):
        if isinstance(index, list):
            # print('getitem list')
            examples = {self.column_names[0]: [], self.column_names[1]: []}
            for idx in index:
                image, text = self.get_raw_item(idx)
                # apply transformation
                # caption
                text = pre_caption(text, self.max_words)
                examples[self.column_names[0]].append(image)
                examples[self.column_names[1]].append(text)
            examples = self.preprocess_train(examples)
            return examples
        else:
            # print('getitem not list')
            image, text = self.get_raw_item(index)
            # apply transformation
            # caption
            text = pre_caption(text, self.max_words)
            examples = {self.column_names[0]: [image], self.column_names[1]: [text]}
            examples = self.preprocess_train(examples)
            return examples


class ClothesTestKVDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        mode="test",
        num_readers=2,
        preprocess_train=None,
        max_words=None,
        img_res=512,
    ):
        self.preprocess_train = preprocess_train
        self.path = data_path  # 数据存储路径
        self.num_readers = num_readers
        self.mode = mode

        self.max_words = max_words
        self.column_names = ["image", "text"]
        with mp.Pool(1) as p:
            self.raw_keys = p.map(get_keys, [(data_path, num_readers)])[0]
        print("raw_keys: ", len(self.raw_keys))

    def __len__(self):
        return len(self.raw_keys)

    def get_raw_item(self, i):
        # i = 10 # for debug
        k = self.raw_keys[i]
        value = self.reader.read_many([k])
        value = value[0]
        value = decode_tfrecord(value, img_res=512)
        image = value["images"][0]  # curretnly first image
        text = value["title"]
        return image, text

    def __getitem__(self, index):
        # print('getitem not list')
        image, text = self.get_raw_item(index)
        text = pre_caption(text, self.max_words)
        return {"image": np.array(image), "text": text, "index": index}


class KVSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, dataset, batch_size, num_replicas, rank, shuffle=True, drop_last=False):
        super(KVSampler, self).__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, drop_last=drop_last)
        self.batch_size = batch_size

    def __iter__(self):
        iterable = super(KVSampler, self).__iter__()
        return chunk(iterable, self.batch_size)

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    # Avoid "cannot pickle KVReader object" error
    dataset.reader = KVReader(dataset.path, dataset.num_readers)


def create_dataloader(datasets, samplers, batch_size, num_workers, is_trains=[True], collate_fns=[None]):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
        if is_train:
            shuffle = sampler is None
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
    trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=torch.tensor(0), std=torch.tensor(0.5))])
