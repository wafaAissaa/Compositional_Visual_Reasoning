import torch
import pickle
from param import args
from gqa_predicttrain import GQA

# remember to change 0 if lang_feat else 1 for visual feats twice
# VALID LAYOUTS CHANGE TO avoid CUDA out of memory


def save_feats():

    gqa = GQA()

    # Load Model
    gqa.load('models/BEST')
    '''
    ques_id, lang_feats = gqa.get_Features(gqa.train_tuple)
    print(lang_feats.size())
    torch.save(lang_feats, 'data/train/lang_feats_%s_29.pth' % args.data[-1])
    with open('data/train/lang_ids_%s_29.p' % args.data[-1], 'wb') as f:
        pickle.dump(ques_id, f)'''
    gqa.feature_files(gqa.train_tuple)

    print('DONE SAVING FEATS %s' % args.train)

    del gqa


if __name__ == "__main__":
    save_feats()

