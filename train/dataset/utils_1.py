import re


def pre_question(question, max_ques_words):
    question = (
        re.sub(
            r"([,.'!?\"()*#:;~])",
            "",
            question.lower(),
        )
        .replace("-", " ")
        .replace("/", " ")
    )
    question = question.rstrip(" ")

    # truncate question
    question_words = question.split(" ")
    if len(question_words) > max_ques_words:
        question = " ".join(question_words[:max_ques_words])

    return question


def pre_caption(caption, max_words):
    caption = (
        re.sub(
            r"([,.'!?\"()*#:;~])",
            "",
            caption.lower(),
        )
        .replace("-", " ")
        .replace("/", " ")
        .replace("<person>", "person")
    )

    caption = re.sub(
        r"\s{2,}",
        " ",
        caption,
    )
    caption = caption.rstrip("\n")
    caption = caption.strip(" ")

    # truncate caption
    caption_words = caption.split(" ")
    if caption_words[0].lower() == "shein":
        caption_words = caption_words[1:]
    if max_words is not None and len(caption_words) > max_words:
        caption_words = caption_words[:max_words]
    caption = " ".join(caption_words)
    # caption不能为空
    if caption == "":
        caption = "a cloth"
    # if not caption.isascii():
    #     # caption = 'a product'
    # print(caption)
    return caption


# from vqaTools.vqaEval import VQAEval
# from refTools.evaluation.refEvaluation import RefEvaluation

import json
import os
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from tqdm import tqdm


from dataloader import KVReader
import torch


def chunk(iterable, chunk_size, drop_last=False):
    ret = []
    for record in iterable:
        ret.append(record)
        if len(ret) == chunk_size:
            yield ret
            ret = []
    drop_last = drop_last and len(ret) != chunk_size
    if not drop_last:
        yield ret


def get_keys(args):
    return KVReader(*args).list_keys()


def worker_init_fn(_):  # 读数据的worker
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.reader = KVReader(dataset.data_path, dataset.num_readers)


class KVDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, dataset, batch_size, num_replicas, rank, shuffle=True, drop_last=False):
        super(KVDistributedSampler, self).__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, drop_last=drop_last)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        iterable = super(KVDistributedSampler, self).__iter__()
        return chunk(iterable, self.batch_size, self.drop_last)

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size


class KVSampler(torch.utils.data.SequentialSampler):
    def __init__(self, dataset, batch_size):
        super(KVSampler, self).__init__(dataset)
        self.batch_size = batch_size
        self.num_samples = len(dataset)

    def __iter__(self):
        iterable = super(KVSampler, self).__iter__()
        return chunk(iterable, self.batch_size)

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size


def filter_text(text):
    from_list = ["！", "＂", "＃", "＄", "％", "＆", "＇", "（", "）", "＊", "＋", "，", "－", "．", "／", "：", "；", "＜", "＝", "＞", "？", "＠", "［", "］", "＾", "＿", "｀", "｛", "｜", "｝", "～", "￮", "【", "】", "\u00a0", "。", "、"]
    to_list = ["!", '"', "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", ":", ";", "<", "=", ">", "?", "@", "[", "]", "^", "_", "`", "{", "|", "}", "~", ".", "[", "]", " ", ".", ","]
    trans_dict = str.maketrans({f: h for f, h in zip(from_list, to_list)})

    def text_preprocess(text):
        """ "处理掉html标签，在右侧标签处加一些句号"""
        text = text.translate(trans_dict)
        text = re.sub("<img .*?>", "", text)
        text = re.sub("<p *>|<li *>|<ul *>|\-|#|•|&\s?amp;|\*|&\s?nbsp;|<--sep--\s*\d+\s*/>|=|&gt;|●", " ", text)
        text = re.sub("</p>|</li>|</ul>", ". ", text)
        text = re.sub("\.+(\.|\s)+", ". ", text)  # 多个句号夹杂空格
        text = re.sub("^\s*\.+(\.|\s)*", "", text)  # 开头的句号
        text = re.sub("\s+", " ", text)  # 多个连续空格

        return text.strip()

    return text_preprocess(text)
