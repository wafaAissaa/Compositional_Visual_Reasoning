# coding=utf-8
# Copyleft 2019 project LXRT.
import json
import os
import collections
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import csv
import base64
import subprocess
import paramiko

from  gqa_model import GQAModel
from  gqa_data import GQADataset, GQATorchDataset, GQAEvaluator

from  param import args


DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def put_file(machinename, username, dirname, filename, data):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(machinename, username=username)
    sftp = ssh.open_sftp()
    '''try:
        sftp.mkdir(dirname)
    except IOError:
        pass'''
    f = sftp.open(dirname + '/' + filename, 'wb')
    # f.write(data)
    # pickle.dump(data, f)
    torch.save(data, f)
    f.close()
    ssh.close()


def get_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = GQADataset(splits)
    tset = GQATorchDataset(dset)
    evaluator = GQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=10,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class GQA:
    def __init__(self):
        self.train_tuple = get_tuple(
            args.train, bs=args.data_bs, shuffle=False, drop_last=False
        )

        '''if args.valid != "":
            valid_bsize = 2048 if args.multiGPU else 512
            self.valid_tuple = get_tuple(
                args.valid, bs=valid_bsize,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None'''

        self.model = GQAModel(self.train_tuple.dataset.num_answers)

        # GPU options
        self.model = self.model.cuda()

        '''if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()'''

        #OUTPUT
        '''self.output = args.output
        os.makedirs(self.output, exist_ok=True)'''

    def predict(self, eval_tuple: DataTuple, dump=None):
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            # ques_id, feats, boxes, sent = datum_tuple[:4]   # avoid handling target
            ques_id, img_id, label, feats, boxes, objects_ids, attrs_ids, sent = datum_tuple[:8]
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                feat_seq, pooled_output, logit = self.model(feats, boxes, sent)
                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid] = ans
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def predict_dumpFeatures(self, eval_tuple: DataTuple, dump=None):
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        #FIELDNAMES = ["ques_id", "img_id", "lang_feats", "visn_feats", "pooled_output", "boxes", "objects_id",
        #              "attrs_id", "label"]
        #FIELDNAMES = ["ques_id", "img_id", "lang_feats", "boxes", "objects_id", "attrs_id", "label"]
        FIELDNAMES = ["ques_id", "lang_feats"]
        with open('./save/feats.tsv', 'w') as tsvfile:
            writer = csv.DictWriter(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)
            for i, datum_tuple in enumerate(loader):
                ques_id, img_id, label, feats, boxes, objects_ids, attrs_ids, sent = datum_tuple[:8] # avoid handling target
                with torch.no_grad():
                    feats, boxes = feats.cuda(), boxes.cuda()
                    feat_seq, pooled_output, logit = self.model(feats, boxes, sent)
                    score, label = logit.max(1)
                    j = 0
                    for qid, l in zip(ques_id, label.cpu().numpy()):
                        ans = dset.label2ans[l]
                        #quesid2ans[qid] = ans
                        #tmp = {"ques_id": int(qid), "img_id": int(img_id[j]), "lang_feats": feat_seq[0][j].tolist(),
                        # "visn_feats": feat_seq[1][j].tolist(),
                        #                   "pooled_output": pooled_output[j].tolist(), "boxes": boxes[j].tolist(),
                        #                   "objects_id": objects_ids[j].tolist(), "attrs_id": attrs_ids[j].tolist(),
                        #                   "label": label[j].tolist()}
                        #tmp = {"ques_id": int(qid), "img_id": int(img_id[j]), "lang_feats": feat_seq[0][j].tolist(),
                        #                   "boxes": boxes[j].tolist(), "objects_id": objects_ids[j].tolist(),
                        #                   "attrs_id": attrs_ids[j].tolist(), "label": label[j].tolist()}
                        tmp = {"ques_id": qid, "lang_feats": feat_seq[0][j].tolist()}
                        writer.writerow(tmp)
                        j += 1
                break

        return quesid2ans

    def get_Features(self, eval_tuple: DataTuple, dump=None):
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        FIELDNAMES = ["ques_id", "lang_feats"]
        #feats = torch.zeros([1,36, 786], dtype=torch.float32)
        #ids = torch.zeros([1])
        for i, datum_tuple in enumerate(loader):
            ques_id, img_id, label, feats, boxes, objects_ids, attrs_ids, sent = datum_tuple[:8] # avoid handling target
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                feat_seq, pooled_output, logit = self.model(feats, boxes, sent)
                if i == 0:
                    visual_feats = feat_seq[0].cpu()  # 0 if lang_feat else 1 for visual feats
                    ids = ques_id
                else:
                    visual_feats = torch.cat((visual_feats, feat_seq[0].cpu()), 0) # 0 if lang_feat else 1 for visual feats
                    ids += ques_id
            #CONCAT HERE TEST SIZE BEFORE !!
        return ids, visual_feats

    def feature_files(self, eval_tuple: DataTuple):

        #ssh = paramiko.SSHClient()
        #ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        #ssh.connect('192.168.1.109', username='wafa')
        #sftp = ssh.open_sftp()

        self.model.eval()
        dset, loader, evaluator = eval_tuple
        print('start extracting features')
        for i, datum_tuple in enumerate(loader):
            # ques_id, img_id, label, feats, boxes, objects_ids, attrs_ids, sent = datum_tuple[:8]
            ques_id, feats, boxes, sent = datum_tuple  # avoid unk ans label error
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                feat_seq, pooled_output, logit = self.model(feats, boxes, sent)
                lang, viz = feat_seq
                for n, id in enumerate(ques_id):
                    save = torch.cat((lang[n].cpu(), viz[n].cpu()), 0) # torch.Size([65, 768]) 29 + 36
                    # torch.save(save, '/media/wafa/wafa_disk/submit/features/%s.pth' % id)
                    # if args.train =='train_all':
                    # put_file('192.168.1.109', 'wafa', '/media/wafa/wafa_disk/train_all/features', '%s.pth' % id, save)
                    #f = sftp.open('/media/wafa/wafa_disk/train_all/features/%s.pth' % id, 'wb')
                    #torch.save(save, f)
                    #f.close()
                    torch.save(save, '/media/Data2/wafa/challenge_all/features/%s.pth' % id)
                        # print('saved', '/media/Data2/wafa/train_all/features/%s.pth' % id )
                    # torch.save(save, '%s/%s/features/%s.pth' % (args.data_path, args.train, id))
        #ssh.close()
        print('done saving files')


    def evaluate(self, eval_tuple: DataTuple, dump=None):
        dset, loader, evaluator = eval_tuple
        quesid2ans = self.predict(eval_tuple, dump)
        return evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load viz from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        for key in list(state_dict.keys()):
            if '.module' in key:
                state_dict[key.replace('.module', '')] = state_dict.pop(key)
        self.model.load_state_dict(state_dict, strict=False)







'''if __name__ == "__main__":
    print('MAINING')
    # Build Class
    gqa = GQA()

    # Load Model
    if args.load is not None:
        gqa.load(args.load)

    # Test or Train
    print(args.test)
    if args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test
        if 'submit' in args.test:
            gqa.predict(
                get_tuple(args.test, bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'submit_predict.json')
            )
        if 'testdev' in args.test:
            result = gqa.evaluate(
                get_tuple('testdev', bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'testdev_predict.json')
            )
            print(result)
        if 'predicttrain' in args.test :
            gqa.predict_dumpFeatures(
                get_tuple('train', bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'feats.tsv')
            )

    else:
        # print("Train Oracle: %0.2f" % (gqa.oracle_score(gqa.train_tuple) * 100))
        print('Splits in Train data:', gqa.train_tuple.dataset.splits)
        if gqa.valid_tuple is not None:
            print('Splits in Valid data:', gqa.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (gqa.oracle_score(gqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        gqa.train(gqa.train_tuple, gqa.valid_tuple)


'''
