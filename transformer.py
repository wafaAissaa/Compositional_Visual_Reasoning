import json, time, sys, pickle
import os, ast, math
import collections
import csv
# csv.field_size_limit(sys.maxsize)
import base64
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import pandas as pd
# from src.gqa_predicttrain import GQA, get_tuple
from src.tokenization import BertTokenizer
from tree import textual, MODULES
# from predictor import Predictor
from executor import MAX_GQA_LENGTH
from src.param import args

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


TARGET_SIZE = 16 # 14 + bos + eos


class TransformerDecoderLayer(nn.Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn_output_weights = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                                        key_padding_mask=memory_key_padding_mask)
        # print("attn_output_weights ", attn_output_weights.size())
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        if hasattr(self, "activation"):
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        else:  # for backward compatibility
            tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt, attn_output_weights


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)


class TransformerDataset(Dataset):
    def __init__(self, tokenizer, split, max_seq_length=MAX_GQA_LENGTH-1): # -1 bcz load[1:] so we already don't take CLS
        super().__init__()

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.split = split
        self.mappings = pd.read_hdf('%s/data/split_mapping.h5' % args.data_path)

        if split != 'submit':
            if split == 'train+val':
                self.data = json.load(open('%s/data/train/train.json' % args.data_path)) \
                            + json.load(open('%s/data/val/val.json' % args.data_path))

                layouts_train = json.load(open('%s/data/train/layouts_train_words_ids.json' % args.data_path))
                layouts_val = json.load(open('%s/data/val/layouts_val_words_ids.json' % args.data_path))
                self.layouts = {**layouts_train, **layouts_val}
                # self.mappings = pd.read_hdf('%s/data/split_mapping_train+val.h5' % args.data_path)
                wv_train = pickle.load(
                    open("%s/data/train_all/most_similar_to_given_train_all.p" % args.data_path, "rb"))
                wv_val = pickle.load(
                    open("%s/data/val_all/most_similar_to_given_val_all.p" % args.data_path, "rb"))
                self.wv_txt_ind = {**wv_train, **wv_val}
            else:
                self.data = json.load(open('%s/data/%s/%s.json' % (args.data_path, split, split)))
                self.layouts = json.load(open('%s/data/%s/layouts_%s_words_ids.json' % (args.data_path, split, split)))
                self.wv_txt_ind = pickle.load(
                    open("%s/data/%s/most_similar_to_given_%s.p" % (args.data_path, split, split), "rb"))

            # self.path = '%s/data/%s/' % (args.data_path, split)
            self.targets = {key: value['target'] for key, value in self.layouts.items()}

        else: #submit
            with open('%s/data/%s/%s.json' % (args.data_path, split, split)) as f:
                self.data = json.load(f)
            if args.predictor_start != 4000000:
                self.data = self.data[args.predictor_start:args.predictor_start+1000000]
            else:
                self.data = self.data[args.predictor_start:]


        # only use available features
        '''file = open('%s/data/train/features/filenames.txt' % args.data_path, 'r')
        filenames = file.read().split('.pth\n')
        small_data = []
        for item in self.data:
            if item['question_id'] in filenames:
                small_data.append(item)
        self.data = small_data'''

        print("Use %d data in transformer dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        # feats = self.data[item]
        # ques_id = self.qids[item]
        cur_data = self.data[item]
        sent = cur_data['sent']
        tokens = self.tokenizer.tokenize(sent.strip())
        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]
        # tokens = tokens_a + ["[SEP]"] not same eos embeddings for all questions
        memory_mask = [0] * len(tokens) # non-zero positions are not allowed to attend
        padding = [1] * (self.max_seq_length - len(tokens)) # 28 = max_seq_length
        memory_mask += padding
        memory_mask = torch.tensor(memory_mask, dtype=torch.bool)
        ques_id = cur_data['question_id']

        if self.split not in ['submit', 'test_all', 'test', 'challenge_all', 'challenge']:
            if self.split == 'train+val':
                # load = torch.load('%s/train/features/%s.pth' % (args.features_path, ques_id))
                load = torch.load(
                    '%s/%s/%s.pth' % (args.features_path, self.mappings.loc[ques_id]['split'], ques_id))
            else:
                load = torch.load('%s/%s/%s.pth' % (args.features_path, self.split, ques_id))
            lang_feats = load[1:29]  # 0 is pooled question
            modules = self.targets[ques_id] + ['EOS']
            inputs = torch.zeros(TARGET_SIZE).long()
            inputs[0] = MODULES.index('BOS')
            target = torch.zeros(TARGET_SIZE).long()
            if len(modules) >= 16:
                raise Exception(ques_id)
            for t in range(len(modules)):
                inputs[t + 1] = MODULES.index(modules[t])
                target[t] = MODULES.index(modules[t])

            # create target for arguments
            target_args = torch.zeros((TARGET_SIZE, self.max_seq_length))
            semantic = self.layouts[ques_id]['semantic']
            for i, m in enumerate(semantic):
                if m['operation'] in textual:
                    txt_ind = [tokens.index(w) for w in self.tokenizer.tokenize(m['argument'].strip()) if
                               w in tokens]
                    if not txt_ind:
                        txt_ind = self.wv_txt_ind[ques_id][m['argument']]
                        txt_ind = [t-1 for t in txt_ind] # -1 bcz the sim matrix was created for the executor and executor sentences start with ["CLS"]
                    target_args[i].index_fill_(0, torch.tensor(txt_ind), 1)

                    # target_args[i] = torch.tensor(same_name + [0] * (28 - len(tokens)))
            return ques_id, lang_feats, memory_mask, inputs, target, target_args
        else:  # submit case
            try:
                load = torch.load('%s/%s/%s.pth' % (args.features_path, self.mappings.loc[ques_id]['split'], ques_id))
                lang_feats = load[1:29]  # 0 is pooled question
                return ques_id, lang_feats, memory_mask, 0, 0, 0
            except Exception as e:
                print('HEREEEE EXEPCTIONNNN')
                print(ques_id)

        # print(ques_id)
        # print(lang_feats.size())
        # print(memory_mask.size())
        # print(inputs.size())
        # print(target.size())


class ProgramGenerator(nn.Module):

    def __init__(self, ntoken, nhid, nhead, dropout=0.5):
        super(ProgramGenerator, self).__init__()
        self.ntoken = ntoken
        self.embeddings = nn.Embedding(ntoken, nhid)
        self.embeddings.weight.data.uniform_(-0.001, 0.001)
        # print(self.embeddings.weight)a
        self.decoder_layer = TransformerDecoderLayer(d_model=nhid, nhead=nhead)
        self.output = nn.Linear(nhid, ntoken)
        self.output.weight.data.uniform_(-0.001, 0.001)

    def forward(self, tgt, memory, memory_mask, tgt_mask, tgt_key_padding_mask=None):
        # print(tgt)
        tgt = self.embeddings(tgt)
        # print(tgt.transpose(0,1).size(), memory.transpose(0,1).size(), tgt_mask.size())
        # print(tgt.type(), memory.type(), tgt_mask.type())
        predict, attn_output_weights = self.decoder_layer(tgt=tgt.transpose(0, 1), memory=memory.transpose(0, 1),
                                                          memory_key_padding_mask=memory_mask, tgt_mask=tgt_mask,
                                                          tgt_key_padding_mask=tgt_key_padding_mask)
        out = self.output(predict) # torch.Size([12, bs, 32])
        # print('att', attn_output_weights.size()) torch.Size([bs, 12, 28])
        # print('memory', memory.size()) torch.Size([bs, 28, 768])
        txt = torch.bmm(attn_output_weights, memory)  # torch.Size([bs, 12, 768])

        return txt, attn_output_weights, out

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by
            the attention. If a ByteTensor is provided, the non-zero positions will be ignored while the zero
            positions will be unchanged
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device=args.device)


def train_epoch(model, data_loader, optimizer, criterion):

    model.to(device=args.device)
    model.train()  # Turn on the train mode
    ntokens = len(MODULES)
    total_loss, total_loss_tokens, total_loss_args = 0, 0, 0
    target_pad = MODULES.index('PAD')
    for i, (ques_id, feats, memory_mask, inputs, target, target_args) in enumerate(data_loader):
        # print('tgt_args ',target_args.size()) size bs,12,28
        optimizer.zero_grad()
        """
        [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked
        positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
        while the zero positions will be unchanged.
        """
        target_mask = (inputs == target_pad).squeeze(1).to(device=args.device)
        # print('size target mask', target_msk.size())
        # print(memory_mask.size(), memory_mask)
        # print(target_mask.size(), target_mask)
        inputs, feats, memory_mask, target, target_args = inputs.to(device=args.device), feats.to(device=args.device), \
                                             memory_mask.to(device=args.device), target.to(device=args.device), \
                                             target_args.to(device=args.device)
        _, attn_output_weights, output = model(tgt=inputs, memory=feats, memory_mask=memory_mask,
                          tgt_mask=model.generate_square_subsequent_mask(TARGET_SIZE), tgt_key_padding_mask=target_mask)
        # attn_output_weights.size() bs, 12, 28
        # print('output',output.size())
        # _, prediction = output.max(2) # recheck this
        # print(prediction)
        # batch_size = prediction.size(1)
        """ for b in range(batch_size):
            #print('INPUT: ', [MODULES[p] for p in inputs[b]])
            print('TARGET: ', [MODULES[p] for p in target[b]])
            print('PREDICTION GT: ', [MODULES[p] for p in prediction.transpose(0, 1)[b]]) """
        kl = nn.KLDivLoss(reduction='batchmean')
        loss_args = kl(torch.log(attn_output_weights+1e-7), target_args)
        loss_tokens = criterion(output.transpose(0, 1).reshape(-1, model.ntoken), target.view(-1))
        # print(loss_args, loss_tokens)
        loss = 0.5 * loss_args + loss_tokens
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        # print('iteration: %s, loss: %s, loss_tokens: %s, loss_args: %s' %(i, loss.item(), loss_tokens.item(), loss_args.item()))
        total_loss += loss.item()
        total_loss_tokens += loss_tokens.item()
        total_loss_args += loss_args.item()

        '''predictor = Predictor(model, beam_size=5)

        for b in range(batch_size):
            seq = predictor.translate_sentence(feats[b].unsqueeze(0))
            print('PREDICTION: ' , [MODULES[p] for p in seq])'''

    # print(total_loss/(i+1))
    return total_loss/(i+1), total_loss_tokens/(i+1), total_loss_args/(i+1)


def train(split, epochs, log_interval, lr, batch_size, start, save='', path=''):

    model = ProgramGenerator(len(MODULES), 768, 4)

    if path:
        model.train()
        model.load_state_dict(torch.load(path))

    model.to(device=args.device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    total_loss, total_loss_tokens, total_loss_args = 0, 0, 0
    lst_losses, lst_loss_tokens, lst_loss_args = [], [], []
    lst_acc_testdev, lst_acc_train = [],  []

    if path:
        lst_losses, lst_loss_tokens, lst_loss_args = pickle.load(open("%s/losses.p" % save, "rb"))
        lst_acc_testdev = pickle.load(open("%s/accuracy_testdev.p" % save, "rb"))
        lst_acc_train = pickle.load(open("%s/accuracy_train.p" % save, "rb"))

    for epoch in range(start+1, epochs + 1):
        print('epoch:  ', epoch)

        transformerData = TransformerDataset(tokenizer, split)
        data_loader = DataLoader(transformerData, batch_size=batch_size, shuffle=True, pin_memory=True,
                                 num_workers=2)
        loss, loss_tokens, loss_args = train_epoch(model, data_loader, optimizer, criterion)
        torch.save(model.state_dict(), '%s/transformer_%s.pth' % (save, epoch))
        accuracy_train = evaluate(path=save + '/transformer_%s.pth' % epoch, split='train+val')
        lst_acc_train.append(accuracy_train)
        accuracy_testdev = evaluate(path=save + '/transformer_%s.pth' % epoch, split='testdev_all')
        lst_acc_testdev.append(accuracy_testdev)

        lst_losses.append(loss)
        lst_loss_tokens.append(loss_tokens)
        lst_loss_args.append(loss_args)

        plt.clf()
        plt.plot(range(len(lst_losses)), np.array(lst_losses), color='blue', label='loss_train')
        plt.plot(range(len(lst_losses)), np.array(lst_loss_tokens), color='green', label='loss_tokens')
        plt.plot(range(len(lst_losses)), np.array(lst_loss_args), color='red', label='loss_args')
        plt.legend(loc='best')
        plt.savefig("%s/losses.png" % save)
        plt.clf()
        plt.plot(range(len(lst_acc_testdev)), np.array(lst_acc_testdev), color='pink', label='accuracy_testdev')
        plt.plot(range(len(lst_acc_train)), np.array(lst_acc_train), color='purple', label='accuracy_train')
        plt.legend(loc='best')
        plt.savefig("%s/accuracies.png" % save)
        plt.clf()

        with open('%s/losses.p' % save, 'wb') as f:
            pickle.dump((lst_losses, lst_loss_tokens, lst_loss_args), f)
        with open('%s/accuracy_testdev.p' % save, 'wb') as f:
            pickle.dump(lst_acc_testdev, f)
        with open('%s/accuracy_train.p' % save, 'wb') as f:
            pickle.dump(lst_acc_train, f)


def evaluate(path, split='testdev_all'):

    model = ProgramGenerator(len(MODULES), 768, 4)
    model.to(device=args.device)
    model.load_state_dict(torch.load(path))
    model.eval()

    target_pad = MODULES.index('PAD')
    # criterion = nn.CrossEntropyLoss(ignore_index=0)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    results = {}
    total_loss = 0.
    lst_loss = []
    correct, wrong = 0., 0.

    with torch.no_grad():
        transformerData = TransformerDataset(tokenizer, split)
        # print(len(transformerData))
        data_loader = DataLoader(transformerData, batch_size=args.transformer_bs, shuffle=False, pin_memory=True,
                                 num_workers=2)
        for i, (ques_id, feats, memory_mask, inputs, target, _) in enumerate(data_loader):
            target_mask = (target == target_pad).squeeze(1).to(device=args.device)
            # print('size target mask', target_msk.size())
            # print(target_msk)
            inputs, feats, memory_mask, target = inputs.to(device=args.device), feats.to(device=args.device), \
                                                 memory_mask.to(device=args.device), target.to(device=args.device)
            _, _, output = model(tgt=inputs, memory=feats, memory_mask=memory_mask,
                        tgt_mask=model.generate_square_subsequent_mask(TARGET_SIZE), tgt_key_padding_mask=target_mask)
            # print('output',output.size())
            _, prediction = output.max(2)  # size= 12, 1024
            batch_size = prediction.size(1)
            compare = (1 - target_mask.float()) * target == (1 - target_mask.float()) * prediction.transpose(0, 1)
            accurate = compare.all(dim=1).int().sum().item()
            correct += accurate
            '''for e, q in enumerate(ques_id):
                pred = prediction[:, e].tolist()
                if 2 in pred: pred = pred[:pred.index(2)]
                results[q] = int(pred == target[e].tolist()[:target[e].tolist().index(2)])'''

        print('accuracy %s = ' % split, correct/len(transformerData))
        return correct/len(transformerData)  #, results

if __name__ == "__main__":

    from src.param import args

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train(split=args.train, epochs=10, log_interval=1, lr=args.transformer_lr, batch_size=args.transformer_bs, start=0,
         save=args.transformer_save)

