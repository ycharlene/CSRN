from typing import List
from torch.utils.data import Dataset
import os.path as osp
import logging
import torch
from transformers import  AutoTokenizer
from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
from prettytable import PrettyTable
import random
import regex as re
import copy
from model import open_clip
from transformers import AutoModelForCausalLM, AutoTokenizer

class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("IRRA.dataset")

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
                self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
                self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))


def tokenize(caption: str, tokenizer, text_length=120, truncate=True) -> torch.LongTensor: # 原text_length=77 729

    # sot_token = tokenizer.encoder["<|startoftext|>"]
    # eot_token = tokenizer.encoder["<|endoftext|>"]
    # tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    # result = torch.zeros(text_length, dtype=torch.long)

    # if len(tokens) > text_length:
    #     if truncate:
    #         tokens = tokens[:text_length]
    #         tokens[-1] = eot_token
    #     else:
    #         raise RuntimeError(
    #             f"Input {caption} is too long for context length {text_length}"
    #         )
    # result[:len(tokens)] = torch.tensor(tokens)

    # SIGLIP 分词
    # tokens = tokenizer(caption).squeeze(0)
    # result = torch.where(tokens == 1, torch.tensor(0, dtype=tokens.dtype), tokens)
    # print(len(result))
    # print(result)

    # SIGLIP2 分词
    try:
        tokens = tokenizer(caption, context_length=text_length).squeeze(0)
    except TypeError:
        tokens = tokenizer(caption).squeeze(0)
    # print(tokens)
    result = torch.where(tokens == 1, torch.tensor(0, dtype=tokens.dtype), tokens)
    # print(len(result))
    # print(result)

    # # Qwen 分词
    # tokens = tokenizer(
    #     caption,
    #     padding="max_length",
    #     truncation=True,
    #     max_length=120,
    #     return_tensors="pt"
    # )
    # result = tokens["input_ids"].squeeze(0)
    # result_mask = tokens["attention_mask"].squeeze(0)
    # # print(result)
    # # print(result_mask)
    return result

def tokenize2(text):
    tokenizer = open_clip.get_tokenizer('local-dir:/home/hpc/LAB-data/disk-4T/syc_data/irra-v1/irra/IRRA-main_v2/ViT-L-16-SigLIP2-256')
    inputs = tokenizer(
        text, 
        padding="max_length", 
        max_length=64,
        truncation=True, 
        return_tensors="pt"
    ).to("cuda:0")
    return inputs

class ImageTextDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

        self.tokenizer2 = AutoTokenizer.from_pretrained(self.vision_tower_name)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, captions = self.dataset[index]
        # print('11',captions)
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        tokens = tokenize(captions, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': tokens,
        }

        return ret


class ImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)
    
    def getIndexByPath(self,img_path):
        try:
            index = self.img_paths.index(img_path)
            return index
        except ValueError:
            print(f'{img_path} cannot find in the list.')

    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        # 热力图使用img_path, 训练删除
        return pid, img


class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 text_length: int = 77,
                 truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        # self.tokenizer = SimpleTokenizer()
        self.tokenizer = open_clip.get_tokenizer('local-dir:/home/hpc/LAB-data/disk-4T/syc_data/irra-v1/irra/IRRA-main_v2/ViT-L-16-SigLIP2-256')

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]
        # 热力图使用 text， 训练删除
        # text = caption
        # print('caption1',caption)
        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        # print('caption1',caption)
        return pid, caption


class ImageTextMLMDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 64,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate

        # self.tokenizer = SimpleTokenizer()
        self.tokenizer = open_clip.get_tokenizer('local-dir:/home/hpc/LAB-data/disk-4T/syc_data/irra-v1/irra/IRRA-main_v2/ViT-L-16-SigLIP2-256')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # pid, image_id, img_path, caption = self.dataset[index]
        # img = read_image(img_path)
        # if self.transform is not None:
        #     img = self.transform(img)
        
        # caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        # mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy())

       
        # ret = {
        #     'pids': pid,
        #     'image_ids': image_id,
        #     'images': img,
        #     'caption_ids': caption_tokens,
        #     'mlm_ids': mlm_tokens,
        #     'mlm_labels': mlm_labels,
        #     'caption':caption,
        # }

        # return ret

        pid, image_id, img_path, caption, _ = self.dataset[index]
        # pid, image_id, img_path, caption = self.dataset[index]

        # print('pid',pid)
        # print('img_path',img_path)
        # print('caption1111',caption)  
        # print("********************** caption **********************")
        # print(caption)
        # print("*****************************************************")
        # print('image_id',image_id)
        # print('captions type:', type(caption)) 
       
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        # from .test_nltk import texts_nltk_clip
        # phrase_vectors = texts_nltk_clip(captions)
        # print(phrase_vectors)
        # print("phrase_vectors.shape 1  ",phrase_vectors.shape)

        # print("caption ", caption)
        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate) 

        # caption_tokens = tokenize2(text=caption)
        # caption_tokens2 = caption_tokens["input_ids"][0]
        # caption_tokens["input_ids"] = caption_tokens2

        # caption_tokens = tokenize2(text=caption)["input_ids"][0]

        # print("caption_toekn.shape---- ", caption_tokens.shape)
        # print("caption_tokens.shape ", caption_tokens.shape)
        # print("caption_tokens.shape", caption_tokens.shape)
        
        # # print('caption_tokens.shape ',caption_tokens.shape)
        # caption_tokens = tokenize(captions[0],tokenizer=self.tokenizer,text_length=self.text_length,truncate=self.truncate)
        # # print('cap_tokens.shape ',cap_tokens.shape)
    
        # for cap in captions[1:]:
        #     cap_tokens1 = tokenize(cap,tokenizer=self.tokenizer,text_length=self.text_length,truncate=self.truncate) 
        #     # cap_tokens = torch.cat((cap_tokens,cap_tokens1),dim=0)  #行合并（385）
        #     caption_tokens = caption_tokens+cap_tokens1 #列叠加（77）
        # print('caption_tokens231:',caption_tokens)
        mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy())
        # print(caption_tokens)
        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': caption_tokens,
            # 'caption_ids': caption,
            'mlm_ids': mlm_tokens,
            'mlm_labels': mlm_labels,
            # 'caption':cap_tokens,
            # 'phrase_vectors': phrase_vectors
        }

        return ret
        # # return img,caption


    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        # # CLIP
        # mask = self.tokenizer.encoder["<|mask|>"] # 49405
        # token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405

        # # print("tokens.shape ", tokens.shape)
        # labels = []
        # for i, token in enumerate(tokens):
        #     if 0 < token < 49405:
        #         prob = random.random()
        #         # mask token with 15% probability
        #         if prob < 0.15:
        #             prob /= 0.15

        #             # 80% randomly change token to mask token
        #             if prob < 0.8:
        #                 tokens[i] = mask

        #             # 10% randomly change token to random token
        #             elif prob < 0.9:
        #                 tokens[i] = random.choice(token_range)

        #             # -> rest 10% randomly keep current token

        #             # append current token to output (we will predict these later)
        #             labels.append(token)
        #         else:
        #             # no masking token (will be ignored by loss function later)
        #             labels.append(0)
        #     else:
        #         labels.append(0)
        
        # if all(l == 0 for l in labels):
        #     # at least mask 1
        #     labels[1] = tokens[1]
        #     tokens[1] = mask


        # SIGLIP
        mask = 4 # <unk>
        token_range = list(range(217, 255968)) # 217 ~ 255967

        labels = []
        for i, token in enumerate(tokens):
            if 216 < token < 255968:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)
        
        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)