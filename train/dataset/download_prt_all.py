# from hdfs_utils import batch_download_from_hdfs, get_hdfs_list
# import os

# def download_meta(remote_path, local_path, names):
#     if isinstance(names, str):
#         names = [names]
    
#     assert isinstance(names, list)
#     names = set(names)

#     os.makedirs(local_path, exist_ok=True)
#     src_list = get_hdfs_list(remote_path)
#     src_list = [src for src in src_list if src.split("/")[-1].split(".")[0] in names]
#     print(src_list)

#     batch_download_from_hdfs(src_list, local_path, mp_size=10)

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--root_path', default='RuntimeDataset')
#     parser.add_argument('--pretrain_path', 
#             default="hdfs://harunava/home/byte_ecom_magellan_supply_chain/user/liushengzhe/train_shopee/0704_ALBEF_PklEN/checkpoint.pth",
#     )
#     args = parser.parse_args()

#     download_meta(
#         "hdfs://harunava/home/byte_ecom_magellan_supply_chain/user/zhouziqi/common_share/product_prt/rec_dataset_multimodal/",
#         local_path=args.root_path, 
#         names=['PRT_ALL_CAT_TRANS'])

#     download_meta(
#         "hdfs://harunava/home/byte_ecom_magellan_supply_chain/user/zhouziqi/common_share/product_prt/rec_dataset_meta/", 
#         local_path=args.root_path, 
#         names=[
#             'prt_rec_meta_AllCat_trans_0', 'prt_rec_meta_AllCat_trans_1', 'prt_rec_meta_AllCat_trans_2', 
#             'prt_rec_meta_AllCat_trans_3', 'prt_rec_meta_AllCat_trans_4', 'prt_rec_meta_AllCat_trans_5',
#             'prt_rec_meta_AllCat_trans_6', 'prt_rec_meta_AllCat_trans_7', 'prt_rec_meta_AllCat_trans_8', 
#             'prt_rec_meta_AllCat_trans_9',
#         ]
#     )
#     pretrain_path = args.pretrain_path
#     local_path = args.root_path
#     os.system(f"hdfs dfs -get {pretrain_path} {local_path}")
    
#     tts_bmk_path= "hdfs://harunava/home/byte_ecom_magellan_supply_chain/user/zhouziqi/common_share/product_prt/test_bmk/20220720_TTS_UPDATE_ALL.pkl"
#     local_path = args.root_path
#     os.system(f"hdfs dfs -get {tts_bmk_path} {local_path}")
    
#     os.system('hdfs dfs -get hdfs://harunava/home/byte_ecom_magellan_supply_chain/user/zhouziqi/common_share/product_prt/test_bmk/GB_prt_tag2cat.json GB_prt_tag2cat.json')

import os
import pickle as pkl
import random
import gc
import argparse

from collections import defaultdict
from dataloader import merge, KVReader
from dataset.hdfs_utils import batch_download_from_hdfs, get_hdfs_list


def download_meta(remote_path, local_path, name):
    os.makedirs(local_path, exist_ok=True)
    src_list = get_hdfs_list(remote_path)
    src_list = [src for src in src_list if name in src.split("/")[-1].split(".")[0]]
    print(src_list)

    batch_download_from_hdfs(src_list, local_path, mp_size=10)


def prepare_dataset_and_meta(args):
    arnold_paths, meta_paths = [], []

    for data_name in args.data_name:
        data_fnames = [os.path.join(args.root_path, data_name, os.path.splitext(fname)[0]) for fname in os.listdir(os.path.join(args.root_path, data_name)) if fname.endswith('.index')]
        arnold_paths.extend(data_fnames)
        meta_fnames = [os.path.join(args.root_path, data_name, fname) for fname in os.listdir(os.path.join(args.root_path, data_name)) if fname.endswith('.pkl')]
        meta_paths.extend(meta_fnames)
    
    # merge all arnold dataset files
    print('merging arnold dataset...')
    print(arnold_paths)
    arnold_merge_path = os.path.join(args.root_path, args.merge_name)
    merge(arnold_paths, arnold_merge_path)
    print('merged to {}'.format(arnold_merge_path))

    # merge all meta files and split into train-val folds
    print('merging meta...')
    meta_overall = merge_meta_data(meta_paths)

    print('splitting meta into train/val, #folds = {}'.format(args.num_fold))
    meta_folds = split_meta(meta_overall, num_fold=args.num_fold)
    for idx, meta in enumerate(meta_folds):
        meta_fname = os.path.join(args.root_path, "{}_meta_{}.pkl".format(args.merge_name, idx))
        print('saving fold {} / {} to {} ...'.format(idx, args.num_fold, meta_fname))
        pkl.dump(meta, open(meta_fname, 'wb'))
    
    print('Arnold dataset and meta data prepared successfully')
    
def merge_meta_data(meta_paths):
    meta_overall = []
    for path in meta_paths:
        meta_overall += pkl.load(open(path, 'rb'))

    return meta_overall

def split_meta(meta_overall, num_fold=10, cat_balance=True):
    samps_by_cat = defaultdict(list)
    for samp in meta_overall:
        pid_plat, label = samp.split('#')
        samps_by_cat[label].append(samp)
    
    meta_list = [[] for _ in range(num_fold)]  
    for cat, samps in samps_by_cat.items():
        assert len(samps) > 0, 'empty sample list'
        num_samp_per_fold = len(samps) // num_fold
        if cat_balance and num_samp_per_fold < 1:
            # make sure at least one sample of any category exists in every fold by randomly duplicating samples
            # if the number of samples of a category is insufficient to be distributted in all folds
            samps_ext = [samps[random.randint(0, len(samps) - 1)] for _ in range(num_fold - len(samps))]
            samps.extend(samps_ext)
            num_samp_per_fold = len(samps) // num_fold
            assert num_samp_per_fold == 1

        random.shuffle(samps)
        if num_samp_per_fold >= 1:
            for idx in range(num_fold):
                if idx == num_fold - 1:
                    meta_list[idx].extend(samps)
                    del samps[:]
                else:
                    meta_list[idx].extend(samps[:num_samp_per_fold])
                    del samps[:num_samp_per_fold]
            assert len(samps) == 0
        else:
            idx = 0
            num_samp_per_fold = 1
            while len(samps) > 0:
                meta_list[idx].extend(samps[:min(len(samps), num_samp_per_fold)])
                del samps[:min(len(samps), num_samp_per_fold)]
                idx += 1
    
    return meta_list


def remove_nonexist_sample_meta(keyset, meta):
    # remove samples from meta that does not exist in arnold dataset
    new_meta, removed = {}, {}
    for pid, samp in meta.items():
        if 'arnold_key' in samp:
            key = samp['arnold_key']
        else:
            key = "{}-{}#{}".format(pid, samp['platform'], samp['label'])
        if key in keyset:
            new_meta[pid] = samp
        else:
            removed[pid] = samp

    assert len(removed) == len(meta) - len(new_meta)
    print('removed {} samples from meta'.format(len(removed)))
    del keyset
    gc.collect()
    return new_meta, removed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default='RuntimeDataset')
    parser.add_argument('--pretrain_path', default="hdfs://harunava/home/byte_ecom_magellan_supply_chain/user/huzhengjiang/PRT_Project/models/ALBFU/pretrain/checkpoint.pth")
    parser.add_argument('--remote_data_path', default="hdfs://harunava/home/byte_ecom_magellan_supply_chain/user/huzhengjiang/PRT_Project/data/prt_train/allcats_0922,hdfs://harunava/home/byte_ecom_magellan_supply_chain/user/huzhengjiang/PRT_Project/data/prt_train/allcats_0922")
    parser.add_argument('--remote_meta_path', default="hdfs://harunava/home/byte_ecom_magellan_supply_chain/user/huzhengjiang/PRT_Project/data/prt_train/allcats_0922,hdfs://harunava/home/byte_ecom_magellan_supply_chain/user/huzhengjiang/PRT_Project/data/prt_train/allcats_0922")
    parser.add_argument('--data_name', default="prt_allcats_0922,prt_train_for_allcat_0922")
    parser.add_argument('--meta_name', default="prt_allcat_meta_0922,prt_train_for_allcat_meta_0922")
    parser.add_argument('--merge_name', default='prt_train_allcats')
    parser.add_argument('--num_fold', type=int, default=10)

    args = parser.parse_args()

    args.remote_data_path = args.remote_data_path.split(',')
    args.remote_meta_path = args.remote_meta_path.split(',')
    args.data_name = args.data_name.split(',')
    args.meta_name = args.meta_name.split(',')
    assert len(args.remote_data_path) == len(args.data_name) == len(args.meta_name) == len(args.remote_meta_path)
    random.seed(0)

    for rmt_data, data_name, rmt_meta, meta_name in zip(args.remote_data_path, args.data_name, args.remote_meta_path, args.meta_name):
        # mm_dataset
        download_meta(
            rmt_data,
            local_path=os.path.join(args.root_path, data_name), 
            name=data_name)
        download_meta(
            rmt_meta,
            local_path=os.path.join(args.root_path, data_name),
            name=meta_name)

    prepare_dataset_and_meta(args)

    pretrain_path = args.pretrain_path
    local_path = args.root_path
    os.system(f"hdfs dfs -get {pretrain_path} {local_path}")
    tts_bmk_path = 'hdfs://harunava/home/byte_ecom_magellan_supply_chain/user/zhouziqi/common_share/product_prt/test_bmk/20220811_APPEND_30NewCats.pkl'
    local_path = os.path.join(args.root_path, "TTS_BMK.pkl")
    os.system(f"hdfs dfs -get {tts_bmk_path} {local_path}")

    os.system('hdfs dfs -get hdfs://harunava/home/byte_ecom_magellan_supply_chain/user/zhouziqi/common_share/product_prt/test_bmk/GB_prt_tag2cat.json GB_prt_tag2cat.json')
