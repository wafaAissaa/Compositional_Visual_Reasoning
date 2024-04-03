import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.param import args

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# DEVICE = torch.device('cpu')

labels = json.load(open("%s/data/trainval_ans2label.json" %args.data_path))
yes = labels['yes']
no = labels['no']

class Select_2(nn.Module):

    def __init__(self, dim_txt, dim_vis, dim=768,  nb_obj=36):
        super().__init__()
        self.linear_txt = nn.Linear(dim_txt, dim)
        self.linear_vis = nn.Linear(dim_vis, dim)
        self.linear_out = nn.Linear(dim, 1)
        nn.init.kaiming_normal_(self.linear_txt.weight)
        nn.init.kaiming_normal_(self.linear_vis.weight)
        nn.init.kaiming_normal_(self.linear_out.weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, att1, att2, txt, vis):
        txt_dim = F.relu(self.linear_txt(txt))
        vis = F.relu(self.linear_vis(vis))
        eltwise_mul = torch.mul(txt_dim, vis)  # 36, 768
        a = self.linear_out(eltwise_mul).transpose(0, 1)  # 1, 36
        return a


# Boolean Modules

class And(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, att1, att2, txt, vis):
        out = att1 * att2 # out = a * b
        return out


class Or(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, att1, att2, txt, vis):
        out = att1 + att2 - att1 * att2  # out = a + b - a * b
        return out


class Same(nn.Module):

    def __init__(self, dim_txt, dim_vis, dim=768):
        super().__init__()
        self.linear_att = nn.Linear(dim_vis, dim)
        self.linear_txt = nn.Linear(dim_txt, dim)
        self.linear_out = nn.Linear(dim, 1)
        nn.init.kaiming_normal_(self.linear_att.weight)
        nn.init.kaiming_normal_(self.linear_txt.weight)
        nn.init.kaiming_normal_(self.linear_out.weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, att1, att2, txt, vis):
        a = F.relu(self.linear_att(torch.mm(att1, vis)))
        b = F.relu(self.linear_att(torch.mm(att2, vis)))
        c = F.relu(self.linear_txt(txt))
        d = self.linear_out(a * b * c)
        # return self.sigmoid(d)
        return d


class SameAll(nn.Module):
    # SameAll(same.linear1, same.linear2)
    def __init__(self, linear_att, linear_txt, dim=768):
        super().__init__()
        self.linear_att = linear_att
        self.linear_txt = linear_txt
        self.linear_out = nn.Linear(dim, 1)
        nn.init.kaiming_normal_(self.linear_out.weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, att1, att2, txt, vis):
        a = F.relu(self.linear_att(torch.mm(att1, vis)))
        b = F.relu(self.linear_txt(txt))
        c = self.linear_out(a * b)
        # return self.sigmoid(c)
        return c


class Different(nn.Module):
    # different(same.linear1, same.linear2, same.linear3)
    def __init__(self, linear_att, linear_txt, linear_out):
        super().__init__()
        self.linear_att = linear_att
        self.linear_txt = linear_txt
        self.linear_out = linear_out
        self.sigmoid = nn.Sigmoid()

    def forward(self, att1, att2, txt, vis):
        a = F.relu(self.linear_att(torch.mm(att1, vis)))
        b = F.relu(self.linear_att(torch.mm(att2, vis)))
        c = F.relu(self.linear_txt(txt))
        d = self.linear_out(a * b * c)
        # return 1 - self.sigmoid(d)
        return 1 - d


class DifferentAll(nn.Module):
  # differentAll(sameAll.linear1, sameAll.linear2, sameAll.linear3)
    def __init__(self, linear_att, linear_txt, linear_out):
        super().__init__()
        self.linear_att = linear_att
        self.linear_txt = linear_txt
        self.linear_out = linear_out
        self.sigmoid = nn.Sigmoid()

    def forward(self, att1, att2, txt, vis):
        a = F.relu(self.linear_att(torch.mm(att1, vis)))
        b = F.relu(self.linear_txt(txt))
        c = self.linear_out(a * b)
        # return 1 - self.sigmoid(c)
        return 1 - c


class Exist(nn.Module):

    def __init__(self, dim=36 + 3):
        super().__init__()
        self.linear = nn.Linear(dim, 1)
        nn.init.kaiming_normal_(self.linear.weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, att1, att2, txt, vis):
        vec = torch.cat((att1, torch.max(att1, dim=-1)[0].unsqueeze(1), torch.min(att1, dim=-1)[0].unsqueeze(1),
                         torch.mean(att1, dim=-1).unsqueeze(1)), 1)
        # return self.sigmoid(self.linear(vec))
        return self.linear(vec)


class VerifyRelSub(nn.Module):

    def __init__(self, dim_txt, dim_vis, dim=768):
        super().__init__()
        self.linear_att = nn.Linear(dim_vis, dim)
        self.linear_txt = nn.Linear(dim_txt, dim)
        self.linear_out = nn.Linear(dim, 1)
        nn.init.kaiming_normal_(self.linear_att.weight)
        nn.init.kaiming_normal_(self.linear_out.weight)
        nn.init.kaiming_normal_(self.linear_txt.weight)

    def forward(self, att1, att2, txt, vis):
        a = F.relu(self.linear_att(torch.mm(att1, vis)))
        b = F.relu(self.linear_att(torch.mm(att2, vis)))
        c = F.relu(self.linear_txt(txt))
        # return F.softmax(self.linear_out(a * b * c), dim=-1)
        return self.linear_out(a * b * c)


class VerifyRelObj(nn.Module):
    # VerifyRelObj(verifyRelSub.linear1, verifyRelSub.linear2)
    def __init__(self, linear_att, linear_txt, dim=768):
        super().__init__()
        self.linear_txt = linear_txt
        self.linear_att = linear_att
        self.linear_out = nn.Linear(dim, 1)
        nn.init.kaiming_normal_(self.linear_out.weight)

    def forward(self, att1, att2, txt, vis):
        a = F.relu(self.linear_att(torch.mm(att1, vis)))
        b = F.relu(self.linear_att(torch.mm(att2, vis)))
        c = F.relu(self.linear_txt(txt))
        # return F.softmax(self.linear_out(a * b * c), dim=-1)
        return self.linear_out(a * b * c)


class VerifyPos(nn.Module):
    #VerifyPos(verifyRelSub.linear1)
    def __init__(self, linear_att, dim_txt, dim=768):
        super().__init__()
        self.linear_att = linear_att
        self.linear_txt = nn.Linear(dim_txt, dim)
        self.linear_out = nn.Linear(dim, 1)
        nn.init.kaiming_normal_(self.linear_txt.weight)
        nn.init.kaiming_normal_(self.linear_out.weight)

    def forward(self, att1, att2, txt, vis):
        a = F.relu(self.linear_att(torch.mm(att1, vis)))
        b = F.relu(self.linear_txt(txt))
        #return F.softmax(self.linear_out(a * b), dim=-1)
        return self.linear_out(a * b)


class VerifyAttr(nn.Module):
    # VerifyAttr(verifyRelSub.linear1)
    def __init__(self, linear_att, dim_txt, dim=768):
        super().__init__()
        self.linear_att = linear_att
        self.linear_txt = nn.Linear(dim_txt, dim)
        self.linear_out = nn.Linear(dim, 1)
        nn.init.kaiming_normal_(self.linear_txt.weight)
        nn.init.kaiming_normal_(self.linear_out.weight)

    def forward(self, att1, att2, txt, vis):
        a = F.relu(self.linear_att(torch.mm(att1, vis)))
        b = F.relu(self.linear_txt(txt))
        # return F.softmax(self.linear_out(a * b), dim=-1)
        return self.linear_out(a * b)


# Attention Modules

'''class Select(nn.Module):

    def __init__(self, dim=768, dim_in=300, nb_obj=36):
        super().__init__()
        self.linear_txt = nn.Linear(dim_in, dim)
        self.linear_out = nn.Linear(dim, 1)
        nn.init.kaiming_normal_(self.linear_txt.weight)
        nn.init.kaiming_normal_(self.linear_out.weight)

    def forward(self, att1, att2, txt, vis):
        a = F.relu(self.linear_txt(txt))
        b = torch.mul(a, vis)  # 36, 768
        c = self.linear_out(b).transpose(0, 1)
        return F.sigmoid(c)'''

class Select(nn.Module):

    def __init__(self, dim=768, nb_obj=36):
        super().__init__()
        self.linear = nn.Linear(dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.linear.weight.data.uniform_(-0.001, 0.001)
        self.dim = dim

    def forward(self, att1, att2, txt, vis):
        eltwise_mul = torch.mul(txt, vis) # 36, 768
        a = self.linear(eltwise_mul)
        return self.sigmoid(a.transpose(0, 1))

class Select_fasttext(nn.Module):

    def __init__(self, dim=768, dim_in=300, nb_obj=36):
        super().__init__()
        self.linear_input = nn.Linear(dim_in, dim)
        self.linear = nn.Linear(dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.linear_input.weight.data.uniform_(-0.001, 0.001)
        self.linear.weight.data.uniform_(-0.001, 0.001)

    def forward(self, att1, att2, txt, vis):
        txt_dim = self.linear_input(txt)
        eltwise_mul = torch.mul(txt_dim, vis) # 36, 768
        a = self.linear(eltwise_mul).transpose(0, 1)
        return self.sigmoid(a)


class FilterAttr(nn.Module):
    # FilterAttr(verifyAttr.linear2)
    def __init__(self, linear_txt, linear_vis, dim=768):
        super().__init__()
        self.linear_txt = linear_txt
        self.linear_vis = linear_vis
        self.linear_out = nn.Linear(dim, 1)
        nn.init.kaiming_normal_(self.linear_out.weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, att1, att2, txt, vis):
        # return self.softmax(torch.min(att1, self.select(att1, att2, txt, vis)))
        a = F.relu(self.linear_txt(txt))
        vis = F.relu(self.linear_vis(vis))
        b = self.sigmoid(self.linear_out(a * vis).transpose(0, 1))
        # return F.softmax(torch.min(att1, b), dim=-1)
        return torch.min(att1, b)


class FilterNot(nn.Module):
    # FilterNot(filterAttr.linear1)
    def __init__(self, linear_txt, linear_vis, linear_out, dim=768):
        super().__init__()
        self.linear_out = linear_out
        self.linear_vis = linear_vis
        self.linear_txt = linear_txt
        self.sigmoid = nn.Sigmoid()

    def forward(self, att1, att2, txt, vis):
        a = F.relu(self.linear_txt(txt))
        vis = F.relu(self.linear_vis(vis))
        b = self.sigmoid(self.linear_out(a * vis).transpose(0, 1))
        #return F.softmax(torch.min(att1, 1 - b), dim=-1)
        return torch.min(att1, 1 - b)


class FilterPos(nn.Module):
    # FilterPos(verifyPos.linear2)
    def __init__(self, linear_txt, linear_vis, dim=768):
        super().__init__()
        self.linear_txt = linear_txt
        self.linear_vis = linear_vis
        self.linear_out = nn.Linear(dim, 1)
        nn.init.kaiming_normal_(self.linear_out.weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, att1, att2, txt, vis):
        a = F.relu(self.linear_txt(txt))
        vis = F.relu(self.linear_vis(vis))
        b = self.sigmoid(self.linear_out(a * vis).transpose(0, 1))
        #return F.softmax(torch.min(att1, b), dim=-1)
        return torch.min(att1, b)


class RelateSub(nn.Module):
    # RelateSub(verifyRelSub.linear_att, verifyRelSub.linear_txt)
    def __init__(self, linear_txt, linear_vis, dim_vis, dim=768, nb_obj=36):
        super().__init__()
        self.linear_txt = linear_txt
        self.linear_att = nn.Linear(dim_vis, dim)
        self.linear_vis = linear_vis
        # self.linear_vis = nn.Linear(nb_obj, 1)
        # self.linear_out = nn.Linear(dim, nb_obj)
        self.linear_out = nn.Linear(dim, 1)
        nn.init.kaiming_normal_(self.linear_att.weight)
        # nn.init.kaiming_normal_(self.linear_vis.weight)
        nn.init.kaiming_normal_(self.linear_out.weight)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, att1, att2, txt, vis):
        a = F.relu(self.linear_att(torch.mm(att1, vis)))
        # b = F.relu(self.linear_vis(vis.transpose(0, 1)).transpose(0, 1))
        b = F.relu(self.linear_txt(txt))
        # c = torch.mul(a * b , vis)  # 36, 768
        vis = self.linear_vis(vis)
        d = self.linear_out(torch.mul(a * b, vis)).transpose(0, 1)
        # return self.softmax(d)
        return d


class RelateObj(nn.Module):
    # RelateSub(verifyRelSub.linear_att, verifyRelSub.linear_txt, relateSub.linear_vis)
    def __init__(self, linear_att, linear_txt, linear_vis, dim=768, nb_obj=36):
        super().__init__()
        self.linear_txt = linear_txt
        self.linear_att = linear_att
        self.linear_vis = linear_vis
        # self.linear_out = nn.Linear(dim, nb_obj)
        self.linear_out = nn.Linear(dim, 1)
        nn.init.kaiming_normal_(self.linear_out.weight)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, att1, att2, txt, vis):
        a = F.relu(self.linear_att(torch.mm(att1, vis)))
        # b = F.relu(self.linear_vis(vis.transpose(0, 1)).transpose(0, 1))
        b = F.relu(self.linear_txt(txt))
        vis = self.linear_vis(vis)
        d = self.linear_out(torch.mul(a * b, vis)).transpose(0, 1)
        # return self.softmax(d)
        return d


class RelateAttr(nn.Module):
    # RelateAttr( verifyRelSub.linear_att, verifyAttr.linear_txt, relateSub.linear_vis )
    def __init__(self, linear_att, linear_txt, linear_vis, dim=768, nb_obj=36):
        super().__init__()
        self.linear_txt = linear_txt
        self.linear_att = linear_att
        self.linear_vis = linear_vis
        # self.linear_out = nn.Linear(dim, nb_obj)
        self.linear_out = nn.Linear(dim, 1)
        nn.init.kaiming_normal_(self.linear_out.weight)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, att1, att2, txt, vis):
        a = F.relu(self.linear_att(torch.mm(att1, vis)))
        # b = F.relu(self.linear_vis(vis.transpose(0, 1)).transpose(0, 1))
        b = F.relu(self.linear_txt(txt))
        vis = self.linear_vis(vis)
        d = self.linear_out(torch.mul(a * b , vis)).transpose(0, 1)
        # return self.softmax(d)
        return d


class Fusion(nn.Module):

    def __init__(self, dim=768, nb_obj=36):
        super().__init__()

    def forward(self, att1, att2, txt, vis):
        return torch.min(att1, att2)


# Answer Modules

class AnswerLogic(nn.Module):
    # the input is between 0 and 1
    def __init__(self, ans=1842):
        super().__init__()
        self.ans = ans

    def forward(self, att1, att2, txt, vis):
        output = torch.full((1, 1842), -10000, dtype=torch.float).to(device=DEVICE)
        output[0, yes] = torch.logit(att1, eps=1e-7)
        output[0, no] = torch.logit(1 - att1, eps=1e-7)
        return output


class ChooseName(nn.Module):

    def __init__(self, dim_txt, dim_vis, dim=768, ans=1842):
        super().__init__()
        self.linear_att = nn.Linear(dim_vis, dim)
        self.linear_txt = nn.Linear(dim_txt, dim)
        self.linear_out = nn.Linear(dim, ans)
        nn.init.kaiming_normal_(self.linear_att.weight)
        nn.init.kaiming_normal_(self.linear_out.weight)
        nn.init.kaiming_normal_(self.linear_txt.weight)

    def forward(self, att1, att2, txt, vis):
        a = F.relu(self.linear_att(torch.mm(att1, vis)))
        b = F.relu(self.linear_txt(txt))
        return self.linear_out(a * b)


class ChooseAttr(nn.Module):
    # chooseAttr(
    def __init__(self, linear_att, linear_txt, dim=768, ans=1842):
        super().__init__()
        self.linear_att = linear_att
        # self.linear_txt = nn.Linear(dim_txt, dim)
        self.linear_txt = linear_txt
        self.linear_out = nn.Linear(dim, ans)
        nn.init.kaiming_normal_(self.linear_out.weight)
        #nn.init.kaiming_normal_(self.linear_txt.weight)

    def forward(self, att1, att2, txt, vis):
        a = F.relu(self.linear_att(torch.mm(att1, vis)))
        b = F.relu(self.linear_txt(txt))
        return self.linear_out(a * b)


class Compare(nn.Module):

    def __init__(self, linear_att, linear_txt, dim=768, ans=1842):
        super().__init__()
        self.linear_att = linear_att
        # self.linear_txt = nn.Linear(dim_txt, dim)
        self.linear_txt = linear_txt
        self.linear_out = nn.Linear(dim, ans)
        #nn.init.kaiming_normal_(self.linear_txt.weight)
        nn.init.kaiming_normal_(self.linear_out.weight)

    def forward(self, att1, att2, txt, vis):
        a = F.relu(self.linear_att(torch.mm(att1, vis)))
        b = F.relu(self.linear_att(torch.mm(att2, vis)))
        c = F.relu(self.linear_txt(txt))
        return self.linear_out(a * b * c)


class ChoosePos(nn.Module):

    def __init__(self, linear_att, linear_txt, dim=768, ans=1842):
        super().__init__()
        self.linear_att = linear_att
        # self.linear_txt = nn.Linear(dim_txt, dim)
        self.linear_txt = linear_txt
        self.linear_out = nn.Linear(dim, ans)
        nn.init.kaiming_normal_(self.linear_out.weight)
        # nn.init.kaiming_normal_(self.linear_txt.weight)

    def forward(self, att1, att2, txt, vis):
        a = F.relu(self.linear_att(torch.mm(att1, vis)))
        b = F.relu(self.linear_txt(txt))
        return self.linear_out(a * b)


class ChooseRel(nn.Module):

    def __init__(self, linear_att, linear_txt, dim=768, ans=1842):
        super().__init__()
        self.linear_att = linear_att
        # self.linear_txt = nn.Linear(dim_txt, dim)
        self.linear_txt = linear_txt
        self.linear_out = nn.Linear(dim, ans)
        nn.init.kaiming_normal_(self.linear_out.weight)
        #nn.init.kaiming_normal_(self.linear_txt.weight)

    def forward(self, att1, att2, txt, vis):
        a = F.relu(self.linear_att(torch.mm(att1, vis)))
        b = F.relu(self.linear_att(torch.mm(att2, vis)))
        c = F.relu(self.linear_txt(txt))
        return self.linear_out(a * b * c)


class Common(nn.Module):

    def __init__(self, linear_att, dim=768, ans=1842):
        super().__init__()
        self.linear_att = linear_att
        self.linear_out = nn.Linear(dim, ans)
        nn.init.kaiming_normal_(self.linear_out.weight)

    def forward(self, att1, att2, txt, vis):
        a = F.relu(self.linear_att(torch.mm(att1, vis)))
        b = F.relu(self.linear_att(torch.mm(att2, vis)))
        return self.linear_out(a * b)


class QueryName(nn.Module):

    def __init__(self, dim_vis, dim=768, ans=1842):
        super().__init__()
        self.linear_att = nn.Linear(dim_vis, dim)
        self.linear_out = nn.Linear(dim, ans)
        nn.init.kaiming_normal_(self.linear_att.weight)
        nn.init.kaiming_normal_(self.linear_out.weight)

    def forward(self, att1, att2, txt, vis):
        a = F.relu(self.linear_att(torch.mm(att1, vis)))
        return self.linear_out(a)


class QueryAttr(nn.Module):

    def __init__(self, linear_att, dim=768, ans=1842):
        super().__init__()
        self.linear_att = linear_att
        self.linear_out = nn.Linear(dim, ans)
        nn.init.kaiming_normal_(self.linear_out.weight)

    def forward(self, att1, att2, txt, vis):
        a = F.relu(self.linear_att(torch.mm(att1, vis)))
        return self.linear_out(a)


class QueryPos(nn.Module):

    def __init__(self, linear_att, dim=768, ans=1842):
        super().__init__()
        self.linear_att = linear_att
        self.linear_out = nn.Linear(dim, ans)
        nn.init.kaiming_normal_(self.linear_out.weight)

    def forward(self, att1, att2, txt, vis):
        a = F.relu(self.linear_att(torch.mm(att1, vis)))
        return self.linear_out(a)

