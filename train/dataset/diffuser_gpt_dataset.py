import os
import torch
import csv
from torchvision import transforms
from dataset.utils import pre_caption
from PIL import Image
import random


class DiffuserGPTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        images_base_path,
        mode="train",
        num_readers=8,
        preprocess_train=None,
        max_words=None,
        img_res=224,
    ):
        self.preprocess_train = preprocess_train
        self.path = data_path  # 数据存储路径
        self.images_base_path = images_base_path  # 图片存储路径
        self.num_readers = num_readers
        self.mode = mode

        self.max_words = max_words
        self.column_names = ["image", "short_text", "long_text"]

        with open(self.path, "r+") as f:
            data = csv.DictReader(f)
            self.raw_keys = []
            for _ in data:
                if _["image_path"] is not None and _["image_path"] != "":
                    self.raw_keys.append(_)

        # self.raw_keys = self.raw_keys[:256] # for debug
        # Uncomment the following lines if num_workers == 0
        # self.reader = KVReader(dataset.path, dataset.num_readers)
        print("raw_keys: ", len(self.raw_keys))

    def with_transform(self, preprocess_train):
        self.preprocess_train = preprocess_train
        return self

    def __len__(self):
        return len(self.raw_keys)

    def get_raw_item(self, idx):
        # i = 10 # for debug
        try:
            image_path = os.path.join(self.images_base_path, self.raw_keys[idx]["image_path"])
            # text = self.raw_keys[idx]["prompt"]
            fine_grained_prompt = self.raw_keys[idx]["fine-grained prompt"]
            coarse_grained_prompt = self.raw_keys[idx]["fine translation prompt"]
            image = Image.open(image_path).convert("RGB")
            return image, fine_grained_prompt, coarse_grained_prompt
        except:
            print("error idx: ", idx)
            print("error raw_keys: ", self.raw_keys[idx])
            idx = random.randint(0, len(self.raw_keys))
            image_path = os.path.join(self.images_base_path, self.raw_keys[idx]["image_path"])
            # text = self.raw_keys[idx]["prompt"]
            fine_grained_prompt = self.raw_keys[idx]["fine-grained prompt"]
            coarse_grained_prompt = self.raw_keys[idx]["coarse-grained prompt"]
            image = Image.open(image_path).convert("RGB")
            return image, fine_grained_prompt, coarse_grained_prompt

    def __getitem__(self, index):
        if isinstance(index, list):
            print("getitem list idx")
            examples = {self.column_names[0]: [], self.column_names[1]: [], self.column_names[2]: []}
            for idx in index:
                image, long_text, short_text = self.get_raw_item(idx)
                # apply transformation
                # caption
                # short_text = long_text[: round(len(long_text) / 2)]
                
                # Change the short_text to be a random subset of the long_text
                # _ = long_text.split(",")
                # if len(_) > 1:
                #     short_text = ",".join(_[: random.randint(1, len(_))])
                # else:
                #     short_text = long_text[: random.randint(1, len(long_text))]

                short_text = pre_caption(short_text, self.max_words)
                long_text = pre_caption(long_text, self.max_words)
                examples[self.column_names[0]].append(image)
                examples[self.column_names[1]].append(short_text)
                examples[self.column_names[2]].append(long_text)
            examples = self.preprocess_train(examples)
            return examples
        else:
            # print('getitem not list')
            image, long_text, short_text = self.get_raw_item(index)
            # apply transformation
            # caption
            # _ = long_text.split(",")
            # if len(_) > 1:
            #     short_text = ",".join(_[: random.randint(1, len(_))])
            # else:
            #     short_text = long_text[: random.randint(1, len(long_text))]
            # short_text = long_text[: round(len(long_text) / 2)]
            short_text = pre_caption(short_text, self.max_words)
            long_text = pre_caption(long_text, self.max_words)
            examples = {self.column_names[0]: [image], self.column_names[1]: [short_text], self.column_names[2]: [long_text]}
            # print("examples: ", examples)
            examples = self.preprocess_train(examples)
            return examples


if __name__ == "__main__":
    trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=torch.tensor(0), std=torch.tensor(0.5))])
