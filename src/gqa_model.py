# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn

#from param import args
from  entry import LXRTEncoder
from  modeling import BertLayerNorm, GeLU

# Max length including <bos> and <eos>
#MAX_GQA_LENGTH = 20
MAX_GQA_LENGTH = 29

class GQAModel(nn.Module):
    def __init__(self, num_answers):
        super().__init__()
        self.lxrt_encoder = LXRTEncoder(
            max_seq_length=MAX_GQA_LENGTH,
            mode='lxr'
        )
        hid_dim = self.lxrt_encoder.dim
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        feat_seq, pooled_output = self.lxrt_encoder(sent, (feat, pos))
        logit = self.logit_fc(pooled_output)

        return feat_seq, pooled_output, logit


