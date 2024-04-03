import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle


MODULES = ['PAD', 'BOS', 'EOS', 'select', 'fusion', 'relateObj', 'relateSub', 'relateAttr', 'filterAttr', 'filterNot', 'filterPos',
           'exist', 'verifyRelObj', 'verifyRelSub', 'verifyAttr', 'verifyPos', 'and', 'or', 'different',
           'differentAll', 'same', 'sameAll',
           'chooseName', 'chooseRel', 'chooseAttr', 'choosePos', 'queryName', 'queryAttr', 'queryPos', 'common',
           'answerLogic', 'compare']

nb_inputs = {'select': 0,
             'fusion': 2,
             'relateAttr': 1,
             'relateObj': 1,
             'relateSub': 1,
             'filterAttr': 1,
             'filterNot': 1,
             'filterPos': 1,
             'exist': 1,
             'verifyRelObj': 2,
             'verifyRelSub': 2,
             'verifyAttr': 1,
             'verifyPos': 1,
             'and': 2,
             'or': 2,
             'different': 2,
             'differentAll': 1,
             'same': 2,
             'sameAll': 1,
             'chooseName': 1,
             'chooseRel': 2,
             'chooseAttr': 1,
             'compare' :2,
             'choosePos': 1,
             'queryName': 1,
             'queryAttr': 1,
             'queryPos': 1,
             'common': 2,
             'answerLogic': 1}

textual = ['same', 'sameAll', 'different', 'differentAll', 'verifyRelSub', 'verifyRelObj', 'verifyAttr', 'verifyPos',
           'select', 'filterAttr', 'filterPos', 'filterNot', 'relateSub', 'relateObj', 'relateAttr', 'relateAttr', 'chooseName', 'chooseAttr',
           'choosePos', 'chooseRel', 'compare']

inputs_type = {'and': 'bool',
               'or': 'bool',
               'same': 'att',
               'sameAll': 'att',
               'different': 'att',
               'differentAll': 'att',
               'exist': 'att',
               'verifyRelSub': 'att',
               'verifyRelObj': 'att',
               'verifyAttr': 'att',
               'verifyPos': 'att',
               'select': 'None',
               'fusion': 'att',
               'relateAttr': 'att',
               'filterAttr': 'att',
               'filterPos': 'att',
               'filterNot': 'att',
               'relateSub': 'att',
               'relateObj': 'att',
               'chooseName': 'att',
               'chooseAttr': 'att',
               'compare': 'att',
               'choosePos': 'att',
               'chooseRel': 'att',
               'common': 'att',
               'queryName': 'att',
               'queryAttr': 'att',
               'queryPos': 'att',
               'answerLogic': 'bool'
               }

outputs_type = {'and': 'bool',  # binary ans was bool
                'or': 'bool',  # binary ans was bool
                'same': 'bool',  # binary ans was bool
                'sameAll': 'bool',  # binary ans was bool
                'different': 'bool',
                'differentAll': 'bool',
                'exist': 'bool',
                'verifyRelSub': 'bool',
                'verifyRelObj': 'bool',
                'verifyAttr': 'bool',
                'verifyPos': 'bool',
                'select': 'att',
                'fusion': 'att',
                'relateAttr': 'att',
                'filterAttr': 'att',
                'filterPos': 'att',
                'filterNot': 'att',
                'relateSub': 'att',
                'relateObj': 'att',
                'chooseName': 'ans',
                'chooseAttr': 'ans',
                'compare': 'ans',
                'choosePos': 'ans',
                'chooseRel': 'ans',
                'common': 'ans',
                'queryName': 'ans',
                'queryAttr': 'ans',
                'queryPos': 'ans',
                'answerLogic': 'ans'
                }

GT = {'select': [0],
      'verifyAttr': [1],
      'filterAttr': [1],
      'relateSub': [1],
      'queryName': [1],
      'chooseAttr': [1, 2],
      'relateObj': [1],
      'verifyRel': [1],
      'verifyPos': [1],
      'filterPos': [1],
      'chooseRel': [1],
      'queryAttr': [1],
      'exist': [1],
      'or': [2],
      'and': [2],
      'queryPos': [1],
      'choosePos': [1],
      'different': [2, 1],
      'filterNot': [1],
      'same': [2, 1],
      'chooseName': [1],
      'common': [2]}


def build_validity_mat(module_names):
    state_size = 4  # [nb_att, nb_bool, nb_ans, T-remain]
    nb_modules = len(MODULES)
    nb_constraints = 4
    P = np.zeros((nb_modules, state_size), np.int32)  # transition matrix
    W = np.zeros((state_size, nb_modules, nb_constraints), np.int32)
    b = np.zeros((nb_modules, nb_constraints), np.int32)

    nb_att_in = np.zeros(nb_modules)
    nb_att_out = np.zeros(nb_modules)
    nb_bool_in = np.zeros(nb_modules)
    nb_bool_out = np.zeros(nb_modules)
    nb_ans_out = np.zeros(nb_modules)

    for n, m in enumerate(MODULES):
        if m not in ['PAD', 'BOS', 'EOS']:
            nb_att_in[n] = (inputs_type[m] == 'att') * nb_inputs[m]
            nb_att_out[n] = outputs_type[m] == 'att'
            nb_bool_in[n] = (inputs_type[m] == 'bool') * nb_inputs[m]
            nb_bool_out[n] = outputs_type[m] == 'bool'
            nb_ans_out[n] = outputs_type[m] == 'ans'

    # transition matrix
    for n, m in enumerate(MODULES):
        P[n, 0] = nb_att_out[n] - nb_att_in[n]  # change in #att = out - in
        P[n, 1] = nb_bool_out[n] - nb_bool_in[n]  # change in #bool = out - in
        P[n, 2] = nb_ans_out[n]  # change in answer
        P[n, 3] = -1  # time-1

    for n, m in enumerate(module_names):
        # x*W - b >= 0
        if m not in ['PAD', 'BOS', 'EOS']: #m != 'EOS':
            # ct 0: we can predict a non eos iff nb_att >= nb_att_in or nb_bool >= nb_bool_in
            if inputs_type[m] == 'att':
                W[0, n, 0] = 1
                b[n, 0] = nb_att_in[n]
            elif inputs_type[m] == 'bool':
                W[1, n, 0] = 1
                b[n, 0] = nb_bool_in[n]
            # ct 1: no outputs left other than the needed inputs if ans, nb_att <= nb_att_in or (and) nb_bool <= nb_bool_in
            if outputs_type[m] == 'ans':
                if inputs_type[m] == 'att':
                    W[0, n, 1] = -1
                    b[n, 1] = -nb_att_in[n]
                    # check if no bool is left if input_type=att nb_bool <= 0
                    W[1, n, 1] = -1
                    # b[n, 1] = -nb_bool_in[n]

                if inputs_type[m] == 'bool':
                    W[1, n, 1] = -1
                    b[n, 1] = -nb_bool_in[n]
                    # check if no att is left if input_type=bool nb_att <= 0
                    W[0, n, 1] = -1
                    # b[n, 1] = -nb_att_in[n]
            if outputs_type[m] in ['att', 'bool']:  # if not ans, T_remain >= 3, enough time for this, ans and eos
                W[3, n, 1] = 1
                b[n, 1] = 3

            # ct2: no answer module has been already predicted nb_ans <= 0
            W[2, n, 2] = -1
            # b[n, 2] = 0 already 0

            # ct3: enough time left to use all att bool ans eos in state
            if outputs_type[m] in ['att', 'bool']:
                if inputs_type[m] == 'att':
                    W[0, n, 3] = -1
                    W[3, n, 3] = 2  # max absorbed att for non ans modules
                    b[n, 3] = 4 + nb_att_in[n] - nb_att_out[n]
                if inputs_type[m] == 'bool':
                    W[1, n, 3] = -1
                    W[3, n, 3] = 2
                    b[n, 3] = 4 + nb_bool_in[n] - nb_bool_out[n]

        elif m in ['EOS', 'PAD']:  # 'EOS' # ans >=1
            W[2, n, 0] = 1
            b[n, 0] = 1

        else:  #PAD BOS
            b[n, 0] = 1

    return torch.FloatTensor(P), torch.FloatTensor(W), torch.FloatTensor(b)


def get_valid_tokens(X, W, b):
    # print('W', W[:, 30, :])
    # print('dot', torch.tensordot(X, W, dims=1)[30,:])
    # print('b', b[30, :])
    constraints_validity = torch.ge(torch.tensordot(X, W, dims=1) - b, 0)
    # print(constraints_validity[30,:])
    # print(X, W)
    # print(torch.tensordot(X, W, dims=1).size()) #torch.Size([1, 30, 4])
    # print('np.tensordot(X, W, axes=1).shape', np.tensordot(X, W, axes=1).shape) #(26,4)
    token_validity = torch.all(constraints_validity, dim=-1)
    # print(token_validity[30])
    # print(token_validity) 26
    return token_validity


def update_decoding_state(X, s, P):

    X = X + P[s]  # X = X + S P  s.t s is modules indice

    return X


if __name__ == "__main__":
    with open('/share/homes/aissa_w/models/generator4/layouts_submit_woPad.p', 'rb') as f:
        train = pickle.load(f)
    print(len(train))

    P, W, b = build_validity_mat(MODULES)
    for qId, values in train.items():
        X = torch.FloatTensor([0, 0, 0, 16])
        for n, l in enumerate(values['target']):
            #print(l)
            if not get_valid_tokens(X, W, b)[l]: print(qId, values['target'], MODULES[l], 'unvalid token')  # qId, values['target'],
            X = update_decoding_state(X, l, P)


