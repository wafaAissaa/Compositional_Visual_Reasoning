import json, shutil
import os
import os.path
import pickle
import sys, time
import matplotlib
from src.param import args
from functions_init import init_modules
from src.modeling import BertEmbeddings, BertPreTrainedModel
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tree import *
from src.tokenization import BertTokenizer
if args.argument_type == 'bert':
    from transformers import BertModel as BertModelTranformers
    from transformers import BertTokenizer as BertTokenizerTranformers
import numpy as np
from gensim.models import fasttext
from gensim.test.utils import datapath
#from autoencoder import Autoencoder
import random
#import paramiko
import pandas as pd
import h5py
from collections import Counter

USE_SSH = False

TARGET_SIZE = 16 # 14 + bos + eos
MAX_GQA_LENGTH = 29 #from gqa_model
HAS_ID_LAYOUTS = ['select', 'relateSub', 'fusion', 'verifyRelObj', 'relateObj', 'verifyRelSub', 'relateAttr', 'chooseRel']
HAS_ID_LAYOUTS_ATT = ['select', 'relateSub', 'fusion', 'relateObj', 'relateAttr']
HAS_OUT_ATT = ['select', 'relateSub', 'fusion', 'relateObj', 'relateAttr', 'filterAttr', 'filterNot', 'filterPos']

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def CL_sampler(data, layouts_file, data_size, step, module_weighting='answer_module_freq', dict_losses={}):

    nb_concepts, len_layout = CL_scheduler(step, nb_iter=args.CL_iter)
    print('nb_concepts, len_layout ', nb_concepts, len_layout)
    weights = []
    valid_qids = []
    data_sample = []
    layouts_sample = {}
    layouts = pd.read_hdf(layouts_file, str(nb_concepts))
    if not args.CL_no_len_layouts:
        print('CL_no_len_layouts is not activated')
        layouts = layouts[layouts['target'].map(len).isin(len_layout)]

    if module_weighting == 'inter_losses':
        if dict_losses == {k: [] for k in MODULES[3:]}:
            print('init weights as 1 for all')
            modules_weights = {k: 1 for k in MODULES[3:]}
        else:
            modules_weights = {k: 1 if v[-1] == 0 else v[-1] for k, v in dict_losses.items()}
            for k, v in modules_weights.items():
                if outputs_type[k] == 'att':
                    modules_weights[k] = v*3
                elif outputs_type[k] == 'bool':
                    modules_weights[k] = v*5

    elif 'module_param' == module_weighting:
        modules_weights = {'select': 590592, 'fusion': 0, 'relateObj': 1180416, 'relateSub': 1180416, 'relateAttr': 1180416, 'filterAttr': 590592, 'filterNot': 590592, 'filterPos': 590592, 'exist': 39, 'verifyRelObj': 1180416, 'verifyRelSub': 1180416, 'verifyAttr': 1180416, 'verifyPos': 1180416, 'and': 0, 'or': 0, 'different': 1180416, 'differentAll': 1180416, 'same': 1180416, 'sameAll': 1180416, 'chooseName': 2594304, 'chooseRel': 2594304, 'chooseAttr': 2594304, 'choosePos': 2594304, 'queryName': 2004480, 'queryAttr': 2004480, 'queryPos': 2004480, 'common': 2004480, 'answerLogic': 0, 'compare': 2594304}
        for k, v in modules_weights.items():
            if v < 300000: modules_weights[k] = 300000

    elif 'layouts_freq' == module_weighting:
        layouts_count = layouts['target'].apply(tuple).value_counts()
        summ = layouts_count.sum()
        layouts_weights = {k: (1 / len(layouts_count)) * (1 / (v / summ)) for k, v in layouts_count.iteritems()}

    else:
        modules_count = {k: 0 for k in MODULES[3:]}

        for k in modules_count.keys():
            if module_weighting == 'answer_module_freq':
                modules_count[k] = layouts[layouts['target'].map(lambda x: x[-1]) == k].shape[0]
            if module_weighting == 'all_module_freq':
                modules_count[k] = layouts['target'].map(lambda x: x[1:].count(k)).sum()

        modules_count = {k: v for k, v in modules_count.items() if v}
        summ = sum(list(modules_count.values()))
        modules_weights = {k: (1 / len(modules_count)) * (1 / (v / summ)) for k, v in modules_count.items()}

    if module_weighting == 'random':
        weights = [1] * len(layouts)
        valid_qids = layouts.index.values.tolist()

    else:
        for key, value in layouts.iterrows():
            valid_qids.append(key)
            if module_weighting == 'answer_module_freq':
                weights.append(modules_weights[value['target'][-1]])
            if module_weighting in ['all_module_freq', 'module_param', 'inter_losses']:
                weights.append(sum([modules_weights[module] for module in value['target'][1:]]))
            if module_weighting == 'layouts_freq':
                weights.append(layouts_weights[tuple(value['target'])])
    print(len(weights), len(valid_qids))
    chosen = random.choices(valid_qids, weights=weights, k=data_size)
    print('chosen', len(chosen), 'data_size', data_size)
    df_chosen = pd.DataFrame({'question_id': chosen})
    data_sample = data.merge(df_chosen, on='question_id')
    print(len(data_sample))
    layouts_sample = layouts.loc[layouts.index.isin(chosen)]

    return data_sample, layouts_sample


def CL_scheduler(step, nb_iter=1):
    len_targets_train_all = {1: 14, 2: 14, 3: 13, 4: 14}
    # ind = int((step - 1) / nb_iter)
    ind = step - 1
    if not args.CL_no_len_layouts:
        nb_concepts_list = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 4]
        len_targets_list = [[2], [3], [4], [6], [5, 7, 8, 10, 12, 14], [4], [5], [6], [3, 7], [8, 9, 10, 12, 14], [6, 7, 8, 9, 10, 11, 12, 13], [12, 13, 14]]
        nb_concepts = nb_concepts_list[ind]
        len_layouts = len_targets_list[ind]
    else:
        nb_concepts_list = [1, 2, 3, 4]
        nb_concepts = nb_concepts_list[ind]
        len_layouts = 0

    return nb_concepts, len_layouts


class ExecutorDataset(Dataset):
    def __init__(self, split, data_path, tokenizer, wv, argument_type='lxmert', end_end='', data_size=None,
                 filter_layouts=False, max_seq_length=MAX_GQA_LENGTH,
                 loss_att=True, loss_bool=True, features_path=None,
                 use_CL=False, step=0, dont_forget='', dict_losses={}, functions=None, bertmodel=None):

        super().__init__()

        self.ans2label = json.load(open("%s/data/trainval_ans2label.json" % data_path))
        self.mappings = pd.read_hdf('%s/data/split_mapping.h5' % args.data_path)
        if split == 'train+val':
            frames = [pd.read_json('%s/data/train/train.json' % data_path, orient='records', dtype={'question_id': str, 'label':dict, 'sent': str, 'img_id': str}), pd.read_json('%s/data/val/val.json' % data_path, orient='records', dtype={'question_id': str, 'label':dict, 'sent': str, 'img_id': str})]
            self.data = pd.concat(frames, ignore_index=True)

            frames = [pd.read_json('%s/data/sceneGraphs/train_sceneGraphs.json' % data_path,
                                            orient=('index'), convert_axes=False),
                      pd.read_json('%s/data/sceneGraphs/val_sceneGraphs.json' % data_path,
                                            orient=('index'), convert_axes=False)]
            self.graphs = pd.concat(frames)

            frames = [pd.read_json('%s/data/train/layouts_train_words_ids.json' % data_path,
                                        orient=('index'), convert_axes=False),
                      pd.read_json('%s/data/val/layouts_val_words_ids.json' % data_path,
                                   orient=('index'), convert_axes=False)
                      ]
            self.layouts = pd.concat(frames)

            self.bboxes = pickle.load(open('%s/data/train/bboxes_train.p' % args.data_path, 'rb'))
            if not wv:
                train = pickle.load(open("%s/data/train_all/most_similar_to_given_train_all.p" % data_path, "rb"))
                val = pickle.load(open("%s/data/val_all/most_similar_to_given_val_all.p" % data_path , "rb"))
                self.wv_txt_ind = {**train, **val}
        elif split == 'submit':
            self.data = pd.read_json('%s/data/%s/%s.json' % (data_path, split, split), orient='records', dtype={'question_id': str, 'sent': str, 'img_id': str})

            with open('%s/data/train/bboxes_train.p' % args.data_path, 'rb') as f:
                self.bboxes = pickle.load(f)
            with open('%s/data/test_all/bboxes_test_.p' % args.data_path, 'rb') as f: #_: with img_id syncro
                self.bboxes.update(pickle.load(f))
            self.layouts = pd.read_pickle('models/generator4/layouts_submit_words_%s.p' % args.predictor_start)
            self.layouts = pd.DataFrame.from_dict(self.layouts, orient=('index'))

            print('using predictor_start arg with value %s' % args.predictor_start)
            if args.predictor_start != 4000000:
                 self.data = self.data[args.predictor_start:args.predictor_start + 1000000]
            else:
                self.data = self.data[args.predictor_start:]

        else:  # testdev, train_all
            if not wv:
                self.wv_txt_ind = pickle.load(open("%s/data/%s/most_similar_to_given_%s.p" % (data_path, split, split), "rb"))
            if 'testdev' in split:
                self.data = pd.read_hdf('%s/data/%s/%s.h5' % (data_path, split, split), 'data')
                self.layouts = pd.read_json('%s/data/%s/layouts_%s_words_ids.json' % (data_path, split, split), orient=('index'), convert_axes=False)#, dtype={"question": str, "semantic":list,  "target":list})

            if 'train' in split:  # train_all case
                self.bboxes = pickle.load(open('%s/data/train/bboxes_train.p' % args.data_path, 'rb'))
                self.graphs = pd.read_json('%s/data/sceneGraphs/train_sceneGraphs.json' % data_path, orient=('index'), convert_axes=False)

                if use_CL:
                    print('start_CL', step)
                    self.data = pd.read_hdf('%s/data/%s/%s.h5' % (data_path, split, split), 'data')
                    layouts_file = '%s/data/%s/layouts_%s_words_ids.h5' % (data_path, split, split)
                    if dont_forget:
                        if step > 1:

                            data_samples, layouts_samples = CL_sampler(self.data, layouts_file, int(data_size * (0.2/(step-1))),
                                                                       step=1, module_weighting='random',
                                                                       dict_losses=dict_losses)
                            for previous_step in range(2, step):
                                data_sample, layouts_sample = CL_sampler(self.data, layouts_file,
                                                                           int(data_size * (0.2/(step-1))),
                                                                           step=previous_step, module_weighting='random',
                                                                           dict_losses=dict_losses)
                                data_samples = pd.concat([data_samples, data_sample], ignore_index=True)
                                print(previous_step, 'data_sample', len(data_sample))
                                print(previous_step, 'data_samples', len(data_samples))
                                layouts_samples = pd.concat([layouts_samples, layouts_sample])

                            data_sample2, layouts_sample2 = CL_sampler(self.data, layouts_file, int(data_size * 0.8),
                                                                       step,
                                                                       module_weighting=args.module_weighting,
                                                                       dict_losses=dict_losses)
                            print('data_sample2', len(data_sample2))
                            print('data_samples', len(data_samples))
                            data_sample = pd.concat([data_samples, data_sample2], ignore_index=True)
                            layouts_sample = pd.concat([layouts_samples, layouts_sample2])

                    if not dont_forget or dont_forget and step == 1:
                        data_sample, layouts_sample = CL_sampler(self.data, layouts_file, data_size, step, module_weighting=args.module_weighting,
                                                             dict_losses=dict_losses)
                    print(dont_forget, step, len(data_sample))
                    self.data = data_sample
                    self.layouts = layouts_sample

                elif data_size == 3000000:
                    self.data = pd.read_json('%s/data/%s/%s_3M.json' % (data_path, split, split), orient='records',
                                             dtype={'question_id': str, 'label': dict, 'sent': str, 'img_id': str})
                    self.layouts = pd.read_json('%s/data/%s/layouts_%s_words_ids_3M.json' % (data_path, split, split),
                                                orient=('index'), convert_axes=False)
                elif data_size == 1000000 and not args.baseline_random:
                    self.data = pd.read_json('%s/data/%s/%s_1M.json' % (data_path, split, split), orient='records',
                                             dtype={'question_id': str, 'label': dict, 'sent': str, 'img_id': str})
                    self.layouts = pd.read_json('%s/data/%s/layouts_%s_words_ids_1M.json' % (data_path, split, split),
                                                orient=('index'), convert_axes=False)

                elif data_size and args.baseline_random == 'baseline1':
                    self.data = pd.read_hdf('%s/data/%s/%s.h5' % (data_path, split, split), 'data')
                    self.data = self.data.sample(n=data_size).reset_index(drop=True)
                    self.layouts = pd.read_json('%s/data/%s/layouts_%s_words_ids.json' % (data_path, split, split),
                                                orient=('index'), convert_axes=False)
                    self.layouts = self.layouts[self.layouts.index.isin(self.data["question_id"])]

                else:
                    self.data = pd.read_hdf('%s/data/%s/%s.h5' % (data_path, split, split), 'data')
                    self.layouts = pd.read_json('%s/data/%s/layouts_%s_words_ids.json' % (data_path, split, split),
                                                orient=('index'), convert_axes=False)

            else: # testdev case and testdev_all
                self.bboxes = pickle.load(open('%s/data/%s/bboxes_%s.p' % (args.data_path, split, split), 'rb'))

        self.bboxes = pd.DataFrame.from_dict(self.bboxes, orient='index')
        self.split = split
        self.data_path = data_path
        if features_path:
            self.features_path = features_path
        else:
            self.features_path = data_path
        self.argument_type = argument_type
        self.end_end = end_end
        self.tokenizer = tokenizer
        self.wv = wv
        self.loss_att = loss_att
        self.loss_bool = loss_bool
        self.has_id_layouts_att = HAS_ID_LAYOUTS_ATT
        self.max_seq_length = max_seq_length

        file = open('%s/data/objects_vocab.txt' % args.data_path, 'r')
        self.objects_vocab = file.read().split('\n')
        self.functions = functions

        if args.features == 'gqa':
            self.objects_info = json.load(open('%s/objects/gqa_objects_info.json' % args.features_path))

        if args.argument_type == 'bert':
            self.bert_model = bertmodel
            self.bert_tokenizer = tokenizer

    def __len__(self):
        print('use %s elements from dataset' % len(self.data))
        return len(self.data)

    def __getitem__(self, item: int):
        # print('stat getitem')
        cur_data = self.data.iloc[item]
        ques_id = cur_data['question_id']
        img_id = cur_data['img_id']
        # if self.split == 'train+val':

        if args.features == 'gqa':
            info = self.objects_info[img_id]
            file_id = info['file']
            file = h5py.File('%s/objects/gqa_objects_%s.h5' % (args.features_path, file_id), 'r')
            bboxes = file['bboxes'][info['idx']][:36, :]
            visual_feats = file['features'][info['idx']][:36, :]
            if args.argument_type == 'fasttext':
                lang_feats = torch.zeros(29, 768)

            if args.use_coordinates:
                normalized = bboxes.copy()
                normalized[:, (0, 2)] /= info['width']
                normalized[:, (1, 3)] /= info['height']
                visual_feats = torch.cat([torch.FloatTensor(visual_feats), torch.FloatTensor(normalized)], dim=1)

        else:
            if 'train_all' in self.split:
                # load = torch.load('%s/train_all/features/%s.pth' % (self.features_path, ques_id))
                load = torch.load('%s/train_all/%s.pth' % (self.features_path, ques_id))
            else:
                load = torch.load('%s/%s/%s.pth' % (self.features_path, self.mappings.loc[ques_id]['split'], ques_id))
                # load = torch.load('%s/%s/%s.pth' % (self.features_path, self.split, ques_id))

            lang_feats = load[:29]  # [29, 768]
            visual_feats = load[29:]  # [36, 768]

            if args.use_coordinates:
                img = self.bboxes.loc[img_id]
                coordinates_bboxes = img['boxes']
                normalized = coordinates_bboxes.copy()
                normalized[:, (0, 2)] /= img['img_w']
                normalized[:, (1, 3)] /= img['img_h']
                visual_feats = torch.cat([visual_feats, torch.FloatTensor(normalized)], dim=1)

        if args.split == 'submit':
            target = -1
            semantics = self.layouts.loc[ques_id]['semantic']
        else:
            ans = next(iter(cur_data['label']))
            if ans in self.ans2label.keys():
                target = self.ans2label[ans]
            else:
                target = -1
            semantics = self.layouts.loc[ques_id]['semantic']

        if args.argument_type in ['lxmert', 'fasttext']:
            tokens_a = self.tokenizer.tokenize(cur_data['sent'].strip())
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > self.max_seq_length - 2:
                tokens_a = tokens_a[:(self.max_seq_length - 2)]
            question = ["[CLS]"] + tokens_a + ["[SEP]"]
        else:
            encoded_question = self.bert_tokenizer(cur_data['sent'].strip(), padding='max_length', max_length=MAX_GQA_LENGTH, truncation=True, return_tensors='pt')
            question = self.bert_tokenizer.convert_ids_to_tokens(encoded_question.input_ids.squeeze())

            lang_feats = torch.load('%s/%s_bert/%s.pth' % (self.features_path, self.split, ques_id))

        if self.argument_type == 'lxmert':
            txts = torch.zeros(TARGET_SIZE, 768, dtype=torch.float)
        elif self.argument_type == 'fasttext':
            txts = torch.zeros(TARGET_SIZE, 300, dtype=torch.float)
        elif self.argument_type == 'bert':
            txts = torch.zeros(TARGET_SIZE, 768, dtype=torch.float)

        if args.tgt_att_type == 'soft':
            target_att = torch.full([TARGET_SIZE, 36], -1, dtype=torch.float)
        else:
            target_att = torch.full([TARGET_SIZE], -1, dtype=torch.long)

        target_bool = torch.full([TARGET_SIZE, 1], -1, dtype=torch.float)

        if args.split == 'submit' or self.end_end:
            for i, m in enumerate(semantics):
                # if m['operation'] == 'PAD':
                    # continue
                if m['operation'] in textual:
                    # TODO: lang_feats = self.wv[question.split()]
                    txt_attn = torch.FloatTensor(m['attn']).unsqueeze(0)
                    txts[i] = txt_attn.matmul(lang_feats[1:])

        else:
            for i, m in enumerate(semantics):
                if m['operation'] in textual and self.end_end == '':
                    if m['argument']:
                        txt_ind = [question.index(w) for w in self.tokenizer.tokenize(m['argument'].strip()) if w in question]
                        if not txt_ind:
                            if not self.wv:
                                # print(self.tokenizer.tokenize(m['argument'].strip()), question[1:-1])
                                txt_ind = self.wv_txt_ind[ques_id][m['argument']]
                            else:
                                txt_ind = [question.index(self.wv.most_similar_to_given(m['argument'], question[1:-1]))]
                        if self.argument_type == 'lxmert':
                            # txts[i] = lang_feats[[x + 1 for x in txt_ind]].unsqueeze(0).mean(dim=1)  # size (1, 768) #ncz i idded cls + tokens
                            txts[i] = lang_feats[[x for x in txt_ind]].unsqueeze(0).mean(dim=1)  # size (1, 768)
                        elif self.argument_type == 'bert':
                            txts[i] = lang_feats[[x for x in txt_ind]].unsqueeze(0).mean(dim=1)  # size (1, 768)
                        elif self.argument_type == 'fasttext':
                            txt = self.wv[' '.join([question[w] for w in txt_ind])].reshape(1,-1).copy() # 1,300
                            txts[i] = torch.from_numpy(txt)
                    else:
                        raise Exception(ques_id, m['operation'], ' needs an argumet' )

                if 'train' not in self.split: continue

                if self.loss_att and m['operation'] in self.has_id_layouts_att:
                    arg = m['argument']
                    if 'inter_id' not in m.keys(): raise Exception(ques_id, m['operation'], ' INTER_ID IS NONE')
                    inter_id = m['inter_id']
                    # print(ques_id, m['operation'], arg, inter_id)
                    if not inter_id: raise Exception(ques_id, m['operation'], ' INTER_ID IS NONE')
                    if inter_id.isdecimal():  # exceptions (id,id,id) for multiple obj, (-), scene
                        # img_id = self.questions[ques_id]['imageId']
                        coordinates_bboxes = self.bboxes.loc[img_id]['boxes']
                        graph_objects = self.graphs.loc[img_id]['objects']
                        name = [o['name'] for i, o in graph_objects.items() if i == inter_id]
                        if not name: continue
                        v = graph_objects[inter_id]
                        coordinates_graph = [[v['x'], v['y'], v['x'] + v['w'], v['y'] + v['h']]]
                        ious = torchvision.ops.box_iou(torch.Tensor(coordinates_graph),
                                                       torch.Tensor(coordinates_bboxes))

                        if args.tgt_att_type == 'soft':
                            ious = torch.where(ious > 0.5, torch.ones_like(ious), torch.zeros_like(ious))
                            if torch.any(ious.gt(0)):
                                target_att[i] = ious
                            else:
                                match_ind = ious.max(dim=1)[1]
                                target_att[i] = F.one_hot(match_ind, num_classes=36).float()
                        else: # args.tgt_att_type == 'hard'
                            match_ind = ious.max(dim=1)[1]
                            target_att[i] = match_ind

                if self.loss_att and 'filter' in m['operation']:
                    target_att[i] = target_att[i-1]

                if self.loss_bool:
                    if m['operation'] in ['verifyRelObj', 'verifyRelSub', 'exist', 'verifyAttr'] and 'and' not in self.layouts.loc[ques_id]['target'] and 'or' not in self.layouts.loc[ques_id]['target'] or m['operation'] in ['and', 'or', 'same', 'sameAll', 'different', 'differentAll', 'verifyPos']:
                        if ans == 'yes': target_bool[i] = 1
                        if ans == 'no': target_bool[i] = 0

                    elif m['operation'] in ['verifyRelObj', 'verifyRelSub', 'exist', 'verifyAttr'] and 'and' in self.layouts.loc[ques_id]['target'] and ans == 'yes':
                        target_bool[i] = 1

                    elif m['operation'] in ['verifyRelObj', 'verifyRelSub', 'exist', 'verifyAttr'] and 'or' in self.layouts.loc[ques_id]['target'] and ans == 'no':
                        target_bool[i] = 0

                    elif m['operation'] == 'exist':
                        if semantics[i-1]['operation'] in ['select', 'fusion'] and semantics[i-1]['inter_id'] == '-':
                            target_bool[i] = 0
                        elif 'filter' in semantics[i-1]['operation'] and semantics[i-2]['inter_id'] == '-':
                            target_bool[i] = 0

        return ques_id, lang_feats, txts, visual_feats, target, target_att, target_bool


class Executor(nn.Module):

    def __init__(self, argument_type='lxmert', end_end='', use_ae=False, loss_att=True, functions=None):
        super().__init__()

        self.functions = init_modules(functions_type=functions, argument_type=argument_type, use_ae=use_ae)

        for module in MODULES:
            if module in ['PAD', 'BOS', 'EOS']: continue
            self.add_module(module, self.functions[module])

        self.end_end = end_end
        self.has_id_layout = HAS_ID_LAYOUTS_ATT

    def forward(self, ques_ids, layouts, lang, txts, viz, loss_att=True, loss_bool=True, eps=0, targets_att=None, targets_bool=None):
        """
        :param lang: (N, 20, 768)
        :param layouts: [[{"operation": _ , "argument": _, "dependencies":[]}, ...], [], ...] # the semantic
        :param viz: tensor() with dim (N, nb_obj, 768))
        :return: answer: probs over answer vocab size (N, len(vocab_answers))
        """
        semantics = [layouts.loc[qid]['semantic'] for qid in ques_ids]
        batch_size = len(semantics)
        final_outputs = []
        outputs_att = torch.zeros(batch_size, TARGET_SIZE, 36, dtype=torch.float).to(device=DEVICE)
        outputs_bool = torch.zeros(batch_size, TARGET_SIZE, 1, dtype=torch.float).to(device=DEVICE)
        outputs_select = torch.zeros(batch_size, TARGET_SIZE, 36, dtype=torch.float).to(device=DEVICE)
        for n in range(batch_size):
            # print(ques_ids[n])
            output = torch.zeros(1, 36, dtype=torch.float).to(device=DEVICE)
            saved_att = torch.zeros(1, 36, dtype=torch.float).to(device=DEVICE)
            saved_bool = torch.zeros(1, 1, dtype=torch.float).to(device=DEVICE)
            out_att = torch.zeros(1, 36, dtype=torch.float).to(device=DEVICE)
            out_bool = torch.zeros(1, 1, dtype=torch.float).to(device=DEVICE)
            visual_feats = viz[n]  # size (36, 768)
            txt = txts[n]  # size(len(semantics), 768)

            saved_att_tgt = torch.zeros(1, 36, dtype=torch.float).to(device=DEVICE)
            saved_bool_tgt = torch.zeros(1, 1, dtype=torch.float).to(device=DEVICE)
            out_att_tgt = torch.zeros(1, 36, dtype=torch.float).to(device=DEVICE)
            out_bool_tgt = torch.zeros(1, 1, dtype=torch.float).to(device=DEVICE)

            for i, m in enumerate(semantics[n]):
                if eps > 0:
                    tf = True if random.random() < eps else False
                else: tf = False

                module = self.functions[m['operation']]
                if m['operation'] == 'select':
                    if i > 0 and outputs_type[semantics[n][i-1]['operation']] == 'att':
                        saved_att = out_att
                        saved_att_tgt = out_att_tgt

                    if i > 0 and outputs_type[semantics[n][i-1]['operation']] == 'bool':
                        saved_bool = out_bool
                        saved_bool_tgt = out_bool_tgt

                    if m['argument'] == 'scene':
                        out_att = torch.ones(1, 36, dtype=torch.float).to(device=DEVICE)
                    else:
                        out_att = module.forward(att1=out_att, att2=saved_att, txt=txt[i], vis=visual_feats)
                        if loss_att: outputs_att[n][i] = out_att
                        if args.tgt_att_type == 'soft':
                            out_att = torch.sigmoid(out_att)
                        else:
                            out_att = F.softmax(out_att, dim=-1)

                        if eps > 0 and args.tgt_att_type == 'soft':
                            out_att_tgt = targets_att[n][i:i + 1]

                        elif eps > 0 and args.tgt_att_type == 'hard' and targets_att[n, i] >= 0:
                             out_att_tgt = F.one_hot(targets_att[n][i:i + 1], num_classes=36).float()

                        elif eps > 0:
                            out_att_tgt = torch.full((1, 36), -1, dtype=torch.float)

                elif inputs_type[semantics[n][i]['operation']] == 'att':
                    if outputs_type[semantics[n][i]['operation']] == 'att':
                        if tf and torch.any(out_att_tgt.ge(0)) and (not nb_inputs[semantics[n][i]['operation']] == 2 or torch.any(saved_att_tgt.ge(0))):
                            out_att = module.forward(att1=out_att_tgt, att2=saved_att_tgt, txt=txt[i], vis=visual_feats)
                        else:
                            out_att = module.forward(att1=out_att, att2=saved_att, txt=txt[i], vis=visual_feats)
                        if loss_att: outputs_att[n][i] = out_att

                        if args.tgt_att_type == 'soft':
                            out_att = torch.sigmoid(out_att)
                        else:
                            out_att = F.softmax(out_att, dim=-1)

                        if eps > 0 and args.tgt_att_type == 'soft':
                            out_att_tgt = targets_att[n][i:i + 1]
                        elif eps > 0 and args.tgt_att_type == 'hard' and targets_att[n, i] >= 0:
                            out_att_tgt = F.one_hot(targets_att[n][i:i + 1], num_classes=36).float()
                        elif eps > 0:
                            out_att_tgt = torch.full((1, 36), -1, dtype=torch.float)

                    elif outputs_type[semantics[n][i]['operation']] == 'bool':
                        if tf and torch.any(out_att_tgt.ge(0)) and (not nb_inputs[semantics[n][i]['operation']] == 2 or torch.any(saved_att_tgt.ge(0))):
                            out_bool = module.forward(att1=out_att_tgt, att2=saved_att_tgt, txt=txt[i], vis=visual_feats)
                        else:
                            out_bool = module.forward(att1=out_att, att2=saved_att, txt=txt[i], vis=visual_feats)
                        if loss_bool:
                            if m['operation'] not in ['and', 'or']:
                                outputs_bool[n][i] = out_bool
                                out_bool = torch.sigmoid(out_bool)
                            else:
                                outputs_bool[n][i] = torch.logit(out_bool, eps=1e-7)

                        if eps > 0: out_bool_tgt = targets_bool[n][i:i + 1]

                    elif outputs_type[semantics[n][i]['operation']] == 'ans':
                        if tf and torch.any(out_att_tgt.ge(0)) and (not nb_inputs[semantics[n][i]['operation']] == 2 or torch.any(saved_att_tgt.ge(0))):
                            output = module.forward(att1=out_att_tgt, att2=saved_att_tgt, txt=txt[i], vis=visual_feats)
                        else:
                            output = module.forward(att1=out_att, att2=saved_att, txt=txt[i], vis=visual_feats)

                elif inputs_type[semantics[n][i]['operation']] == 'bool':
                    if outputs_type[semantics[n][i]['operation']] == 'bool':
                        if tf and torch.any(out_bool_tgt.ge(0)) and (not nb_inputs[semantics[n][i]['operation']] == 2 or torch.any(saved_bool_tgt.ge(0))):
                            out_bool = module.forward(att1=out_bool_tgt, att2=saved_bool_tgt, txt=txt[i], vis=visual_feats)
                        else:
                            out_bool = module.forward(att1=out_bool, att2=saved_bool, txt=txt[i], vis=visual_feats)
                        if loss_bool: outputs_bool[n][i] = out_bool
                        if m['operation'] not in ['and', 'or']: out_bool = torch.sigmoid(out_bool)

                        if eps > 0: out_bool_tgt = targets_bool[n][i:i + 1]

                    elif outputs_type[semantics[n][i]['operation']] == 'ans':
                        if tf and torch.any(out_bool_tgt.ge(0)) and (not nb_inputs[semantics[n][i]['operation']] == 2 or torch.any(saved_bool_tgt.ge(0))):
                            output = module.forward(att1=out_bool_tgt, att2=saved_bool_tgt, txt=txt[i], vis=visual_feats)
                        else:
                            output = module.forward(att1=out_bool, att2=saved_bool, txt=txt[i], vis=visual_feats)

            final_outputs.append(output)

        return torch.cat(final_outputs), outputs_att, outputs_bool  # [n, ans=1842] n,target_size,36a


def train_epoch(model, data_loader, layouts, optimizer, criterion_att, criterion_bool, criterion_ans, epoch,
                use_ae, ae_l, ae_v, loss_att, loss_bool, weight_losses, eps):
    model.train()
    total_loss, accuracy, total_loss1, total_loss2, total_loss3 = 0., 0., 0., 0., 0.
    lst_loss_bool, lst_loss_att = [], []
    dict_loss = {k: [] for k in MODULES[3:]} # if outputs_type[k] != 'ans'}
    start_time = time.time()
    print('start enumerate')
    for i, (ques_ids, lang_feats, txt, visual_feats, targets, targets_att, targets_bool) in enumerate(data_loader):
        lang_feats, txt, visual_feats, targets, targets_att, targets_bool = lang_feats.to(device=DEVICE), txt.to(device=DEVICE), visual_feats.to(device=DEVICE), targets.long().to(device=DEVICE), targets_att.to(device=DEVICE), targets_bool.to(device=DEVICE)

        if use_ae:
            txt = ae_l.encoder(txt)
            lang_feats = ae_l.encoder(lang_feats)
            visual_feats = ae_v.encoder(visual_feats)

        optimizer.zero_grad()
        outputs, outputs_att, outputs_bool = model.forward(ques_ids, layouts, lang_feats, txt, visual_feats, loss_att, loss_bool, eps, targets_att, targets_bool)

        _, prediction = outputs.max(1)

        accuracy += torch.sum((targets == prediction).float()).item()
        l1 = criterion_ans(outputs, targets)
        total_loss1 += l1.item()
        semantics = [layouts.loc[qid]['semantic'] for qid in ques_ids]
        # print(semantics)aa

        for n in range(len(ques_ids)): # for prints
            for k, m in enumerate(semantics[n]):
                if targets_att[n][k].sum() >= 0:
                    if criterion_att == 'soft':
                        mask = targets_att.ge(0)
                        l = criterion_att(torch.masked_select(outputs_att[n, k:k + 1, :]), torch.masked_select(targets_att[n, k:k + 1]))
                    else: l = criterion_att(outputs_att[n, k:k+1, :], targets_att[n, k:k+1])
                    dict_loss[m['operation']].append(l.mean().item())
                elif targets_bool[n][k].sum() >= 0:
                    l = criterion_bool(outputs_bool[n, k:k+1], targets_bool[n, k:k+1])
                    dict_loss[m['operation']].append(l.mean().item())
                elif outputs_type[m['operation']] == 'ans':  # answer module case
                    l = criterion_ans(outputs[n:n+1, :], targets[n:n+1])
                    dict_loss[m['operation']].append(l.item())

        if loss_att and not args.get_losses:

            if args.loss_reduction == 'none':

                if args.tgt_att_type == 'soft':
                    mask = torch.any(targets_att.ge(0), dim=-1, keepdim=True)
                else:
                    mask = targets_att.ge(0)

                modules_id = np.stack(layouts.loc[list(ques_ids)]['target'].apply(lambda x: [MODULES.index(i) for i in x] + [-1 for _ in range(TARGET_SIZE - len(x))]))
                l = modules_id * mask.squeeze(-1).detach().cpu().numpy()
                unique, counts = np.unique(l, return_counts=True)
                c = dict(zip(unique, counts))
                f = np.vectorize(lambda x: 1 / c[x])
                l = f(l) # bs * len(tgt)

                if args.tgt_att_type == 'soft':
                    outputs_att_masked = outputs_att.masked_select(mask)
                    targets_att_masked = targets_att.masked_select(mask)
                    l2 = criterion_att(outputs_att_masked, targets_att_masked).view(-1, 36).mean(dim=-1)
                    l = l.reshape(-1)[mask.detach().cpu().numpy().reshape(-1)]
                    weights_samples = torch.from_numpy(l).float().to(DEVICE)
                    l2 *= weights_samples
                else:
                    l2 = criterion_att(outputs_att.view(-1, 36), targets_att.view(-1))
                    weights_samples = torch.from_numpy(l.reshape(-1)).float().to(DEVICE)

                l2 *= weights_samples
                loss = weight_losses[0] * l1 + weight_losses[1] * l2.sum()
                total_loss2 += l2.sum().item()

            else:
                l2 = criterion_att(outputs_att.view(-1, 36), targets_att.view(-1))
                loss = weight_losses[0] * l1 + weight_losses[1] * l2 #/ len(ques_ids)
                total_loss2 += l2.item()

        if loss_bool and not args.get_losses:
            mask = targets_bool.ge(0)  # .view(-1)
            outputs_bool_masked = torch.masked_select(outputs_bool, mask)
            targets_bool_masked = torch.masked_select(targets_bool, mask)

            if torch.any(targets_bool.ge(0).view(-1)):
                l3 = criterion_bool(outputs_bool_masked, targets_bool_masked)

                if args.loss_reduction == 'none':
                    l = modules_id.reshape(-1)[mask.detach().cpu().numpy().reshape(-1)]
                    unique, counts = np.unique(l, return_counts=True)
                    c = dict(zip(unique, counts))
                    f = np.vectorize(lambda x: 1 / c[x])
                    l = f(l)
                    weights_samples = torch.from_numpy(l).float().to(DEVICE)
                    l3 *= weights_samples #[mask.reshape(-1)]
                    # print('batch boolean loss: ', l3.item())
                    loss += weight_losses[2] * l3.sum() #/ len(ques_ids)
                    total_loss3 += l3.sum().item()
                else:
                    loss += weight_losses[2] * l3 #/ len(ques_ids)
                    total_loss3 += l3.item()
        if not args.get_losses:
            if loss_att or loss_bool:
                loss.backward()
                total_loss += loss.item()
            else:
                l1.backward()
            # if np.random.binomial(n=1, p=0.09, size=1): plot_grad_flow(model.named_parameters(), args.executor_save, epoch, i)
            optimizer.step()

    for k, v in dict_loss.items():
        if v: dict_loss[k] = sum(v) / len(v)


    return (total_loss/(i + 1), total_loss1/(i+1), total_loss2/(i+1), total_loss3/(i+1)), accuracy / len(data_loader.dataset), dict_loss


def train(split, data_path, epochs, wv=None, log_interval=1000, lr=0.01, argument_type='lxmert', ft_path='/%s/'  % args.data_path,
          batch_size=10, start=0, save='', path='', data_size=None, use_scheduler=True, filter_layouts=False,
          use_ae=False, loss_att=True, loss_bool=True, weight_losses='1.,1.,1.', features_path= None, functions=None, use_tf=False):

    print('use inter_losses: ', loss_att, loss_bool)

    if use_tf:
        tf_epsilons = np.concatenate((np.arange(1., 0, -args.tf_step), np.zeros(epochs+1)))
    else:
        tf_epsilons = np.zeros(epochs+1)
    # print('start load ft')
    # start_time = time.time()
    # if not wv: print('USING NONE FOR FASTTEXT')
    # wv = fasttext.load_facebook_vectors(datapath("%s/wiki.en.bin" % ft_path))
    # print('end load ft in %s seconds' % (time.time() - start_time))
    model = Executor(argument_type=argument_type, use_ae=use_ae, loss_att=loss_att, functions=functions)
    # print(model.functions['select'].linear_txt.weight)
    if path:
        model.load_state_dict(torch.load(path)['model_state_dict'])
        if 'CL_step' in torch.load(path).keys():
            step = torch.load(path)['CL_step']
            print('STARTING FROM STEP: ', step)
        else:
            step = 1
            print('Using Path but step is 1')

    if use_ae:
        ae_l = Autoencoder([500, 200])
        ae_v = Autoencoder([500, 200])
        ae_l.load_state_dict(torch.load('/%s/ae5_l_pad/ae_50.pth' % args.data_path))
        ae_v.load_state_dict(torch.load('/%s/ae5_v/ae_50.pth'  % args.data_path))
        ae_l.to(device=args.device)
        ae_v.to(device=args.device)
        ae_l.eval()
        ae_v.eval()
        for p in ae_l.parameters():
            p.requires_grad = False
        for p in ae_v.parameters():
            p.requires_grad = False
    else:
        ae_l, ae_v = None, None

    '''if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
        #model = nn.parallel.DistributedDataParallel(model)'''
    model.to(device=DEVICE)
    model.train()
    if args.tgt_att_type == 'soft':
        criterion_att = nn.BCEWithLogitsLoss(reduction=args.loss_reduction)
    else:
        criterion_att = nn.CrossEntropyLoss(ignore_index=-1, reduction=args.loss_reduction)
    criterion_bool = nn.BCEWithLogitsLoss(reduction=args.loss_reduction)
    criterion_ans = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)  #, momentum=0.9, weight_decay=0.0001)
    if use_scheduler:
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', threshold=0.0001,
                                                               # patience=10, verbose=True, eps=0.00001, factor=0.5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', threshold=0.001, eps=1e-5, patience=2, verbose=True)

    total_loss, total_acc, total_acc_testdev = 0., 0., 0.
    total_loss1, total_loss2, total_loss3 = 0., 0., 0.

    if path and not args.CL_pretrain:
        lst_loss, lst_loss1, lst_loss2, lst_loss3 = pickle.load(open("%s/losses.p" % save, "rb"))
        lst_accuracy = pickle.load(open("%s/accuracy.p" % save, "rb"))
        lst_accuracy_testdev = pickle.load(open("%s/accuracy_testdev.p" % save, "rb"))
        if args.get_losses:
            dict_losses = {k: [] for k in MODULES[3:]}
        else:
            dict_losses = pickle.load(open("%s/dict_losses.p" % save, "rb"))

    else:
        if args.CL_pretrain: print('NOT LOADING ACC LOSS FILES .p !!!', start)
        lst_loss, lst_loss1, lst_loss2, lst_loss3 = [], [], [], []
        lst_accuracy, lst_accuracy_testdev = [], []
        dict_losses = {k: [] for k in MODULES[3:]} # if outputs_type[k] != 'ans'}
        step = 1

    nb_concepts, len_layout = 0, 0

    if args.argument_type == 'bert':
        '''bert_model = BertModelTranformers.from_pretrained("bert-base-uncased")
        for param in bert_model.parameters():
            param.requires_grad = False'''
        bert_model = None
        tokenizer = BertTokenizerTranformers.from_pretrained('bert-base-uncased', do_lower_case=True)
    else:
        bert_model = None
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    if not args.use_CL and not args.baseline_random:
        executor_data = ExecutorDataset(split, data_path, tokenizer, wv=wv, data_size=data_size,
                                        argument_type=argument_type,
                                        filter_layouts=filter_layouts,
                                        loss_att=loss_att, loss_bool=loss_bool,
                                        features_path=features_path, functions=functions, bertmodel=bert_model)

        data_loader = DataLoader(executor_data, batch_size=batch_size, shuffle=True, pin_memory=True,
                                 num_workers=2)

    for epoch in range(start + 1, epochs):
        print('EPOCH / EPOCHS', epoch, epochs)

        if args.use_CL or args.baseline_random:

            if args.CL_repeat:
                if not args.CL_pretrain and (step > 1 and lst_accuracy[-1] > (sum(lst_accuracy[:-1]) / len(lst_accuracy[:-1])) or step == 1 and epoch == 2):
                    step += 1
                elif args.CL_pretrain and (step == 1 and epoch == 4 or step > 1 and lst_accuracy[-1] > (sum(lst_accuracy[3:-1]) / len(lst_accuracy[3:-1]))):
                    step += 1
            else:
                if args.CL_pretrain:  # and epoch > 3:
                    step = epoch - 2
                else:  # if not args.CL_pretrain:
                    step = epoch
                step = int((step-1)/ args.CL_iter) + 1

            executor_data = ExecutorDataset(split, data_path, tokenizer, wv=wv, data_size=data_size,
                                            argument_type=argument_type,
                                            filter_layouts=filter_layouts,
                                            loss_att=loss_att, loss_bool=loss_bool,
                                            features_path=features_path,
                                            use_CL=args.use_CL, step=step,
                                            dict_losses=dict_losses, dont_forget=args.dont_forget, functions=functions)

            data_loader = DataLoader(executor_data, batch_size=batch_size, shuffle=True, pin_memory=True,
                                     num_workers=2)

        start_time = time.time()

        l, a, dict_loss = train_epoch(model, data_loader, executor_data.layouts, optimizer, criterion_att, criterion_bool, criterion_ans, epoch,
                           use_ae, ae_l, ae_v, loss_att, loss_bool, weight_losses, eps=tf_epsilons[epoch-1])

        with open('%s/dict_losses_%s.p' % (save, epoch), 'wb') as f:
            pickle.dump(dict_loss, f)

        if args.get_losses:
            print('FIN FORWARD EPOCH %s' % ( epoch))
            continue

        total_loss += l[0]
        total_loss1 += l[1]
        total_loss2 += l[2]
        total_loss3 += l[3]
        total_acc += a

        for k, v in dict_losses.items():
            if dict_loss[k]:
                dict_losses[k].append(dict_loss[k])
            elif dict_losses[k]:
                dict_losses[k].append(dict_losses[k][-1])
            else:
                dict_losses[k] = [0]

        # print(epoch, dict_losses)
        # print('l, a', l, a)
        # scheduler.step(l[1])
        if epoch % log_interval == 0:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'CL_step': step
                        }, '%s/executor_%s.pth' % (save, epoch))

            lst_loss.append(total_loss / log_interval)
            lst_loss1.append(total_loss1 / log_interval)
            lst_loss2.append(total_loss2 / log_interval)
            lst_loss3.append(total_loss3 / log_interval)
            lst_accuracy.append(total_acc / log_interval)
            total_loss, total_acc = 0., 0.
            total_loss1, total_loss2, total_loss3 = 0., 0., 0.

            plt.clf()
            plt.plot(range(len(lst_loss)), np.array(lst_loss), color='blue', label='loss total')
            plt.plot(range(len(lst_loss)), np.array(lst_loss1), color='red', label='loss answer')
            plt.plot(range(len(lst_loss)), np.array(lst_loss2), color='black', label='loss attention')
            plt.plot(range(len(lst_loss)), np.array(lst_loss3), color='green', label='loss boolean')
            plt.legend(loc='best')
            plt.savefig("%s/loss_%s.png" % (save, epoch))
            plt.clf()

            for k, v in dict_losses.items():
                if outputs_type[k] == 'bool':
                    plt.plot(range(len(lst_loss)), np.array(v),
                             c=(np.random.random(), np.random.random(), np.random.random()), label='loss %s' % k)
            plt.legend(loc='best')
            plt.savefig("%s/losses_bool_%s.png" % (save, epoch))
            plt.clf()

            for k, v in dict_losses.items():
                if outputs_type[k] == 'att':
                    plt.plot(range(len(lst_loss)), np.array(v),
                             c=(np.random.random(), np.random.random(), np.random.random()), label='loss %s' % k)
            plt.legend(loc='best')
            plt.savefig("%s/losses_att_%s.png" % (save, epoch))
            plt.clf()

            for k, v in dict_losses.items():
                if outputs_type[k] == 'ans':
                    plt.plot(range(len(lst_loss)), np.array(v),
                             c=(np.random.random(), np.random.random(), np.random.random()), label='loss %s' % k)
            plt.legend(loc='best')
            plt.savefig("%s/losses_ans_%s.png" % (save, epoch))
            plt.clf()
            with open('%s/losses.p' % save, 'wb') as f:
                pickle.dump((lst_loss, lst_loss1, lst_loss2, lst_loss3), f)
            with open('%s/accuracy.p' % save, 'wb') as f:
                pickle.dump(lst_accuracy, f)
            with open('%s/dict_losses.p' % save, 'wb') as f:
                pickle.dump(dict_losses, f)

            testdev_accuracy = evaluate(tokenizer=tokenizer, wv=wv, path=save + '/executor_%s.pth' % epoch,
                                        data_size=None, argument_type=args.argument_type, split='testdev_all',
                                        save=save, epoch=epoch, filter_layouts=filter_layouts, use_ae=use_ae, ae_l=ae_l,
                                        ae_v=ae_v, features_path=features_path, functions=functions, bert_model=bert_model)

            if use_scheduler: scheduler.step(testdev_accuracy)
            with open('%s/accuracy_testdev.p' % save, 'wb') as f:
                pickle.dump(lst_accuracy_testdev, f)

            lst_accuracy_testdev.append(testdev_accuracy)

            plt.plot(range(len(lst_accuracy)), np.array(lst_accuracy), color='blue', label='accuracy')
            plt.plot(range(len(lst_accuracy_testdev)), np.array(lst_accuracy_testdev), color='red', label='accuracy testdev')
            plt.legend(loc='best')
            plt.savefig("%s/accuracy_%s.png" % (save, epoch))
            plt.clf()

        print('end epoch for %s examples in %s seconds' % (len(executor_data.data), time.time() - start_time)) #args.data_size


def evaluate(path, data_size, tokenizer, wv, save, epoch, split, argument_type, filter_layouts=False,
             use_ae=False, ae_l=None, ae_v=None, features_path=None, functions=None, bert_model=None):

    model = Executor(argument_type=argument_type, use_ae=use_ae, functions=functions)
    model.eval()
    model.to(device=DEVICE)
    model.load_state_dict(torch.load(path)['model_state_dict'])
    # model.load_state_dict(torch.load(path))
    label2ans = json.load(open("%s/data/trainval_label2ans.json" % args.data_path))
    with torch.no_grad():
        correct = 0.
        results = []
        executor_data = ExecutorDataset(split=split, data_path=args.data_path, tokenizer=tokenizer, wv=wv,
                                        data_size=data_size, argument_type=argument_type, filter_layouts=filter_layouts,
                                        features_path=features_path, functions=functions, bertmodel=bert_model)
        data_loader = DataLoader(executor_data, batch_size=args.executor_bs, shuffle=False, pin_memory=True,
                                 num_workers=4)
        for i, (ques_ids, lang_feats, txt, visual_feats, targets, _, _) in enumerate(data_loader):
            # print(i)
            lang_feats, txt, visual_feats, targets = lang_feats.to(device=DEVICE), txt.to(
                device=DEVICE), visual_feats.to(device=DEVICE), targets.long().to(device=DEVICE)

            if use_ae:
                txt = ae_l.encoder(txt)
                lang_feats = ae_l.encoder(lang_feats)
                visual_feats = ae_v.encoder(visual_feats)

            if args.get_outputs:
                loss_att = True
                loss_bool = True
            else:
                loss_att = False
                loss_bool = False
            # print(ques_ids)

            outputs, outputs_att, outputs_bool = model(ques_ids, executor_data.layouts, lang_feats, txt, visual_feats, loss_att=loss_att, loss_bool=loss_bool, eps=0)
            if args.get_outputs:
                with open('%s_outputs_%s/batch_%s.pkl' % (save, args.executor_start, i), 'wb') as fp:
                    pickle.dump({'answers': outputs.cpu().numpy(), 'att': outputs_att.cpu().numpy(), 'bool': outputs_bool.cpu().numpy()}, fp)

            outputs = F.softmax(outputs, dim=1) # added 12 oct
            _, prediction = outputs.max(1)  # size [B]

            if args.split == 'submit':
                for n in range(len(prediction)):
                    results.append({'questionId': ques_ids[n], 'prediction': label2ans[prediction[n]]})
                    # print(results)
            else:
                a = torch.sum((targets == prediction).float()).item()
                correct += a
                # wrong_ids = [ e for e in qids if accuracy.tolist()[0][qids.index(e)]]

                for q, p, t in zip(ques_ids, prediction.tolist(), targets.tolist()):
                    results.append({'qid': q, 'prediction': p, 'targest': t})

        if args.split == 'submit':
            with open('%s/generator4_submit/resutls_submit_%s.json' % (save, args.predictor_start), 'w') as fp:  # indent if not submit
                json.dump(results, fp)
        else:
            with open('%s/results_%s_%s.p' % (save, split, epoch), 'wb') as f:
                pickle.dump(results, f)
            print(correct / len(executor_data))

            return correct / len(executor_data)


class Embeddings(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        embedding_output = self.embeddings(input_ids, token_type_ids)
        return embedding_output



if __name__ == "__main__":
    from src.param import args
    print(args.functions, args.use_coordinates)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    lst_accuracy_testdev = []
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    # embeddings = Embeddings.from_pretrained("bert-base-uncased")
    # embedding_output = embeddings(input_ids, token_type_ids)

    if args.split == 'submit':
        epoch = args.epoch
        evaluate(tokenizer=tokenizer, wv=None, path=args.save + '/executor_%s.pth' % epoch,
                 data_size=None,
                 argument_type=args.argument_type, split='submit', save=args.save, epoch=epoch,
                 features_path=args.features_path, functions=args.functions)

    elif args.split == 'testdev_all' and args.get_outputs:

        if args.argument_type == 'fasttext':
            wv = fasttext.load_facebook_vectors(datapath("%s/wiki.en.bin" % args.ft_path))
            print('end load fasttext')
        else:
            wv = None

        if args.argument_type == 'bert':
            '''bert_model = BertModelTranformers.from_pretrained("bert-base-uncased")
            for param in bert_model.parameters():
                param.requires_grad = False'''
            bert_model = None
            tokenizer = BertTokenizerTranformers.from_pretrained('bert-base-uncased', do_lower_case=True)
        else:
            bert_model = None
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

        path = os.path.join(args.executor_save, 'executor_%s.pth' % args.executor_start)

        testdev_accuracy = evaluate(tokenizer=tokenizer, wv=wv, path=args.executor_save + '/executor_%s.pth' % args.executor_start,
                                        data_size=None, argument_type=args.argument_type, split='testdev_all',
                                        save=args.executor_save, epoch=args.executor_start, filter_layouts=False, features_path=args.features_path, functions=args.functions, bert_model=bert_model)
        print(testdev_accuracy)

    elif args.split == 'testdev_all':
        for epoch in range(1, 51):
            testdev_accuracy = evaluate(tokenizer=tokenizer, wv=None, path=args.save + '/executor_%s.pth' % epoch,
                                    data_size=None,
                                    argument_type=args.argument_type, split='testdev_all', save=args.save, epoch=epoch,
                                    features_path=args.features_path, functions=args.functions)

            print(testdev_accuracy)
            lst_accuracy_testdev.append(testdev_accuracy)
            plt.plot(range(len(lst_accuracy_testdev)), np.array(lst_accuracy_testdev), color='red', label='accuracy testdev_all')
            plt.legend(loc='best')
            plt.savefig("%s/accuracy_testdev_all_%s.png" % (args.save, epoch))
            with open('%s/accuracy_testdev_all.p' % args.save, 'wb') as f:
                pickle.dump(lst_accuracy_testdev, f)

    elif 'train' in args.split:
        if args.executor_start > 0:
            path = os.path.join(args.executor_save, 'executor_%s.pth' % args.executor_start)
        else: path= ''
        if args.argument_type == 'fasttext':
            wv = fasttext.load_facebook_vectors(datapath("%s/wiki.en.bin" % args.ft_path))
            print('end load fasttext')
        else:
            wv = None
        train(epochs=101, log_interval=1, lr=args.executor_lr, wv=wv, save=args.executor_save, path=path, start=args.executor_start,
              split=args.split, data_path=args.data_path, ft_path=args.ft_path, argument_type=args.argument_type,
              data_size=args.data_size, batch_size=args.executor_bs,
              use_scheduler=False, filter_layouts=False, use_ae=False,
              loss_att=args.inter_sup, loss_bool=args.inter_sup,
              weight_losses=[float(x) for x in args.weighting.split(',')],
              features_path=args.features_path, functions=args.functions, use_tf=args.use_tf)


