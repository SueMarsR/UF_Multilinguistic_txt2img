#!/usr/bin/python3
import tensorflow as tf
import sys
import os
import argparse

from dataloader import KVReader, merge
from download_prt_all import download_meta


def prepare_dataset(args):
    
    for data_name, merge_name in zip(args.data_name, args.merge_name):
        arnold_paths = []
        data_fnames = [os.path.join(args.root_path, data_name, os.path.splitext(fname)[0]) for fname in os.listdir(os.path.join(args.root_path, data_name)) if fname.endswith('.index')]
        arnold_paths.extend(data_fnames)
    
        # merge all arnold dataset files
        print('merging arnold dataset...')
        print(arnold_paths)
        arnold_merge_path = os.path.join(args.root_path, args.merge_dir_name, merge_name)
        merge(arnold_paths, arnold_merge_path)
        print('merged to {}'.format(arnold_merge_path))
        print('Arnold dataset prepared successfully')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default='RuntimeDataset')
    parser.add_argument('--remote_data_path', 
                        default=['hdfs://harunava/home/byte_ecom_magellan_supply_chain/user/zhangjun.69/clothing_attribute/train',
                                'hdfs://harunava/home/byte_ecom_magellan_supply_chain/user/zhangjun.69/clothing_attribute/test']
                        )
    parser.add_argument('--data_name', default=["final_shein_cloth_train", "final_shein_cloth_test"])
    parser.add_argument('--data_train_test', default="final_shein_cloth_test")
    parser.add_argument('--merge_dir_name', default="final_shein_cloth")
    parser.add_argument('--merge_name', default=["train", "test"])

    args = parser.parse_args()
    for remote_data_path, data_name in zip(args.remote_data_path, args.data_name):
        args.local_path = os.path.join(args.root_path, data_name)
        os.makedirs(args.local_path, exist_ok=True)
        download_meta(remote_data_path, args.local_path, data_name)
    os.makedirs(os.path.join(args.root_path, args.merge_dir_name), exist_ok=True)
    prepare_dataset(args)


    
