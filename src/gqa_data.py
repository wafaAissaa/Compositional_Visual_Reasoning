# coding=utf-8
# Copyleft 2019 project LXRT.

import json, pickle
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import torch
from torch.utils.data import Dataset

from  param import args
from  utils import load_obj_tsv

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000


class GQADataset:
    """
    A GQA data example in json file:
    {
        "img_id": "2375429",
        "label": {
            "pipe": 1.0
        },
        "question_id": "07333408",
        "sent": "What is on the white wall?"
    }
    """
    def __init__(self, splits: str):
        self.name = splits
        # self.splits = splits.split(',')
        self.splits = splits
        # Loading datasets to data
        if 'all' not in splits:
            self.data = json.load(open("data/%s/%s.json" % (splits, splits)))
        else:
            self.data = json.load(open("/media/Data_XXII/wafa/data/%s.json" % splits))
            print('using data from /media/Data_XXII/wafa/data/%s.json' % splits)

        '''
        VALID LAYOUTS CHANGE TO avoid CUDA out of memory
        with open('data/layouts_valid.json') as f:
            layouts = json.load(f)
        keys = layouts.keys()
        
        self.data = [ d for d in self.data if d["question_id"] in keys]'''

        print("Load %d data from split(s) %s." % (len(self.data), self.name))
        #print(self.data[0])
        # List to dict (for evaluation and others)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        # Answers
        self.ans2label = json.load(open("data/trainval_ans2label.json"))
        self.label2ans = json.load(open("data/trainval_label2ans.json"))
        assert len(self.ans2label) == len(self.label2ans)
        for ans, label in self.ans2label.items():
            assert self.label2ans[label] == ans

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


class GQABufferLoader():
    def __init__(self):
        pass

    def load_data(self, name, number):
        #if name == 'testdev':
        if 'testdev' in name or 'challenge' in name:
            path = "data/testdev/gqa_testdev_obj36.tsv"
            print('loading data/gqa_testdev_obj36.tsv')
        if 'test' in name or 'challenge' in name:
            path = "/media/Data2/wafa/test2015_obj36.tsv"
            print('loading /media/Data2/wafa/test2015_obj36.tsv')
        elif name == 'small':
            path = "data/splits/" + name + '.tsv'
        else:
            # path = "data/vg_gqa_imgfeat/vg_gqa_obj36.tsv"
            path = "/media/Data2/wafa/vg_gqa_obj36.tsv"
            # path = "data/splits/"+name+'.tsv'
            print('loading data/vg_gqa_obj36.tsv')
        return load_obj_tsv(path, topk=number)


gqa_buffer_loader = GQABufferLoader()


"""
Example in obj tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
"""
class GQATorchDataset(Dataset):
    def __init__(self, dataset: GQADataset):
        super().__init__()
        self.raw_dataset = dataset

        '''if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            #CHECK HERE!'''
        topk = -1

        # Loading detection features to img_data
        # Since images in train and valid both come from Visual Genome,
        # buffer the image loading to save memory.
        img_data = []
        if 'testdev' in dataset.splits or 'testdev_all' in dataset.splits:     # Always loading all the data in testdev
            img_data.extend(gqa_buffer_loader.load_data('testdev', topk))
        else:
            img_data.extend(gqa_buffer_loader.load_data(args.train, topk))
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features !!
        '''if 'all' in args.train:
            #feature_files = [f.split('.')[0] for f in listdir('/media/wafa/wafa_disk/%s/features' % args.train) if isfile(join('/media/wafa/wafa_disk/%s/features' % args.train, f))]
            feature_files = pickle.load(open('/media/Data2/wafa/train_all/copied3.p', 'rb'))
        else: feature_files = []'''
        '''self.data = []
        self.qid2datum = {}
        for i, datum in enumerate(self.raw_dataset.data):
            self.qid2datum[datum['question_id']] = datum

        diff = set(self.qid2datum.keys()) - set(feature_files)
        ("the remaining files are %d from %d" % (len(diff), len(self.raw_dataset.data)))
        for qid in diff:
            #if datum['img_id'] in self.imgid2img and ('all' in args.train and datum['question_id'] not in feature_files or 'all' not in args.train):
            self.data.append(self.qid2datum[qid])'''
            # else:
                # print(datum['img_id'], 'not in .tsv')
                # print(datum['question_id'])
        self.data = self.raw_dataset.data#[:3000000]
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

        # only kept the data not already featured in train_all

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        boxes = img_info['boxes'].copy()
        feats = img_info['features'].copy()
        assert len(boxes) == len(feats) == obj_num

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        # Create target
        if args.train not in ['testdev', 'submit', 'test_all', 'challenge_all']:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                if ans in self.raw_dataset.ans2label:
                    target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, feats, boxes, ques #, target
            #return ques_id, img_id, self.raw_dataset.ans2label[ans], feats, boxes, img_info['objects_id'], img_info['attrs_id'], ques, target
        #elif args.train in ['testdev', 'submit']:
        else:
            return ques_id, feats, boxes, ques


class GQAEvaluator:
    def __init__(self, dataset: GQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump the result to a GQA-challenge submittable json file.
        GQA json file submission requirement:
            results = [result]
            result = {
                "questionId": str,      # Note: it's a actually an int number but the server requires an str.
                "prediction": str
            }

        :param quesid2ans: A dict mapping question id to its predicted answer.
        :param path: The file path to save the json file.
        :return:
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'questionId': ques_id,
                    'prediction': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)


