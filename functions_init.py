from src.param import args

if args.functions == 'original':
    from modules import *
elif args.functions == '2':
    from modules_2 import *
elif args.functions == 'kaiming':
    from modules_kaiming import *
elif args.functions == 'qn':
    from modules_qn import *
elif args.functions == 'share':
    from modules_share import *
elif args.functions == 'new':
    from modules_new import *
elif args.functions == 'shareAll':
    from modules_shareAll import *
elif args.functions == 'dropout':
    from modules_dropout import *
elif args.functions == 'new_relate':
    from modules_new_relate import *
elif args.functions == 'coordinates':
    from modules_coordinates import *


def init_modules(functions_type, argument_type='lxmert', use_ae='False'):
    
    if argument_type == 'lxmert':
        dim_in = dim_txt = 768
    elif argument_type == 'fasttext':
        dim_in = dim_txt = 300
        
    if functions_type in ['original', '2', 'qn', 'kaiming']:
        functions = {'relateObj': RelateObj(dim_in), 'relateSub': RelateSub(dim_in),
                          'filterAttr': FilterAttr(dim_in), 'filterNot': filterNot(dim_in), 'filterPos': FilterPos(dim_in),
                          'exist': Exist(), 'verifyRelObj': VerifyRelObj(dim_in), 'verifyRelSub': VerifyRelSub(dim_in),
                          'verifyAttr': VerifyAttr(dim_in), 'verifyPos': VerifyPos(dim_in), 'and': And(), 'or': Or(),
                          'different': Different(dim_in), 'differentAll': DifferentAll(dim_in), 'same': Same(dim_in), 'sameAll': SameAll(dim_in),
                          'chooseName': ChooseName(dim_in), 'chooseRel': ChooseRel(dim_in), 'chooseAttr': ChooseAttr(dim_in),
                          'choosePos': ChoosePos(dim_in), 'queryName': QueryName(), 'queryAttr': QueryAttr(),
                          'queryPos': QueryPos(), 'common': Common(dim_in), 'answerLogic': AnswerLogic(), 'compare': Compare(dim_in),
                          'fusion': Fusion(), 'relateAttr': RelateAttr(dim_in)}
    
        if args.select_2 == 'True':
            functions['select'] = Select_2(dim_in=200)
        else:
            if argument_type == 'lxmert':
                functions['select'] = Select()
            elif argument_type == 'fasttext':
                functions['select'] = Select_fasttext()
    
    elif functions_type in ['share', 'new']:
    
        if use_ae:
            dim_in = 200
            dim = 200
        else:
            dim_in = 768
            dim = 768
    
        functions = {'and': And(), 'or': Or(), 'exist': Exist(), 'fusion': Fusion(), 'same': Same(dim_in)}
        functions['sameAll'] = SameAll(functions['same'].linear_att, functions['same'].linear_txt)
        functions['different'] = Different(functions['same'].linear_att, functions['same'].linear_txt, functions['same'].linear_out)
        functions['differentAll'] = DifferentAll(functions['same'].linear_att, functions['same'].linear_txt, functions['sameAll'].linear_out)
    
        functions['verifyRelSub'] = VerifyRelSub(dim_in)
        functions['verifyRelObj'] = VerifyRelObj(functions['verifyRelSub'].linear_att, functions['verifyRelSub'].linear_txt)
        functions['verifyPos'] = VerifyPos(functions['verifyRelSub'].linear_att, dim_in)
        functions['verifyAttr'] = VerifyAttr(functions['verifyRelSub'].linear_att, dim_in)
    
        functions['filterAttr'] = FilterAttr(functions['verifyAttr'].linear_txt)
        functions['filterNot'] = FilterNot(functions['verifyAttr'].linear_txt, functions['filterAttr'].linear_out)
        functions['filterPos'] = FilterPos(functions['verifyPos'].linear_txt)
    
        functions['relateSub'] = RelateSub(functions['verifyRelSub'].linear_txt)
        functions['relateObj'] = RelateObj(functions['relateSub'].linear_att, functions['verifyRelSub'].linear_txt, functions['relateSub'].linear_vis)
        functions['relateAttr'] = RelateAttr(functions['relateSub'].linear_att, functions['verifyRelSub'].linear_txt, functions['relateSub'].linear_vis)
    
        functions['chooseName'] = ChooseName(dim_in)
        functions['chooseAttr'] = ChooseAttr(functions['chooseName'].linear_att, dim_in)
        functions['chooseRel'] = ChooseRel(functions['chooseName'].linear_att, dim_in)
        functions['choosePos'] = ChoosePos(functions['chooseName'].linear_att, dim_in)
        functions['compare'] = Compare(functions['chooseName'].linear_att, dim_in)
        functions['common'] = Common(functions['chooseName'].linear_att, dim_in)
        functions['queryName'] = QueryName(dim=dim)
        functions['queryAttr'] = QueryAttr(functions['queryName'].linear_att)
        functions['queryPos'] = QueryPos(functions['queryName'].linear_att)
        functions['answerLogic'] = AnswerLogic()
    
        if args.select_2 == 'True':
            functions['select'] = Select_2(dim_in=dim_in, dim=dim)
        else:
            if argument_type == 'lxmert':
                functions['select'] = Select()
            elif argument_type == 'fasttext':
                functions['select'] = Select_fasttext()
    
    elif functions_type in ['new_relate']:
    
        if use_ae:
            dim_in = 200
            dim = 200
    
        functions = {'and': And(), 'or': Or(), 'exist': Exist(), 'fusion': Fusion(), 'same': Same(dim_in)}
        functions['sameAll'] = SameAll(functions['same'].linear_att, functions['same'].linear_txt)
        functions['different'] = Different(functions['same'].linear_att, functions['same'].linear_txt, functions['same'].linear_out)
        functions['differentAll'] = DifferentAll(functions['same'].linear_att, functions['same'].linear_txt, functions['sameAll'].linear_out)
    
        functions['verifyRelSub'] = VerifyRelSub(dim_in)
        functions['verifyRelObj'] = VerifyRelObj(functions['verifyRelSub'].linear_att, functions['verifyRelSub'].linear_txt)
        functions['verifyPos'] = VerifyPos(functions['verifyRelSub'].linear_att, dim_in)
        functions['verifyAttr'] = VerifyAttr(functions['verifyRelSub'].linear_att, dim_in)
    
        functions['filterAttr'] = FilterAttr(functions['verifyAttr'].linear_txt)
        functions['filterNot'] = FilterNot(functions['verifyAttr'].linear_txt, functions['filterAttr'].linear_out)
        functions['filterPos'] = FilterPos(functions['verifyPos'].linear_txt)
    
        functions['relateSub'] = RelateSub(functions['verifyRelSub'].linear_txt)
        functions['relateObj'] = RelateObj(functions['relateSub'].linear_att, functions['verifyRelSub'].linear_txt)
        functions['relateAttr'] = RelateAttr(functions['relateSub'].linear_att, functions['verifyRelSub'].linear_txt)
    
        functions['chooseName'] = ChooseName(dim_in)
        functions['chooseAttr'] = ChooseAttr(functions['chooseName'].linear_att, dim_in)
        functions['chooseRel'] = ChooseRel(functions['chooseName'].linear_att, dim_in)
        functions['choosePos'] = ChoosePos(functions['chooseName'].linear_att, dim_in)
        functions['compare'] = Compare(functions['chooseName'].linear_att, dim_in)
        functions['common'] = Common(functions['chooseName'].linear_att, dim_in)
        functions['queryName'] = QueryName()
        functions['queryAttr'] = QueryAttr(functions['queryName'].linear_att)
        functions['queryPos'] = QueryPos(functions['queryName'].linear_att)
        functions['answerLogic'] = AnswerLogic()
    
        if args.select_2 == 'True':
            functions['select'] = Select_2(dim_in)
        else:
            if argument_type == 'lxmert':
                functions['select'] = Select()
            elif argument_type == 'fasttext':
                functions['select'] = Select_fasttext()
    
    elif functions_type in ['shareAll', 'dropout']:
    
        functions = {'and': And(), 'or': Or(), 'exist': Exist(), 'fusion': Fusion()}
    
        if args.select_2 == 'True':
            functions['select'] = Select_2(dim_in)
        else:
            if argument_type == 'lxmert':
                functions['select'] = Select()
            elif argument_type == 'fasttext':
                functions['select'] = Select_fasttext()
    
        functions['same' ]= Same(functions['select'].linear_txt)
        functions['sameAll'] = SameAll(functions['same'].linear_att, functions['select'].linear_txt, functions['same'].linear_out)
        functions['different'] = Different(functions['same'].linear_att, functions['select'].linear_txt,
                                                functions['same'].linear_out)
        functions['differentAll'] = DifferentAll(functions['same'].linear_att,
                                                      functions['select'].linear_txt,
                                                      functions['same'].linear_out)
        functions['verifyRelSub'] = VerifyRelSub(functions['same'].linear_att, functions['select'].linear_txt)
        functions['verifyRelObj'] = VerifyRelObj(functions['same'].linear_att, functions['select'].linear_txt,
                                                      functions['verifyRelSub'].linear_out)
        functions['verifyPos'] = VerifyPos(functions['same'].linear_att, functions['select'].linear_txt,
                                                functions['verifyRelSub'].linear_out)
        functions['verifyAttr'] = VerifyAttr(functions['same'].linear_att, functions['select'].linear_txt,
                                                  functions['verifyRelSub'].linear_out)
        functions['filterAttr'] = FilterAttr(functions['select'].linear_txt)
        functions['filterNot'] = FilterNot(functions['select'].linear_txt,
                                                functions['filterAttr'].linear_out)
        functions['filterPos'] = FilterPos(functions['select'].linear_txt, functions['filterAttr'].linear_out)
    
        functions['relateSub'] = RelateSub(functions['select'].linear_txt, functions['same'].linear_att)
        functions['relateObj'] = RelateObj(functions['select'].linear_txt, functions['same'].linear_att,
                                                functions['relateSub'].linear_vis,
                                                functions['relateSub'].linear_out)
        functions['relateAttr'] = RelateAttr(functions['select'].linear_txt, functions['same'].linear_att,
                                                  functions['relateSub'].linear_vis,
                                                  functions['relateSub'].linear_out)
        functions['chooseName'] = ChooseName(functions['same'].linear_att, dim_in)
        functions['chooseAttr'] = ChooseAttr(functions['chooseName'].linear_txt, functions['same'].linear_att, functions['chooseName'].linear_out)
        functions['chooseRel'] = ChooseRel(functions['chooseName'].linear_txt, functions['same'].linear_att, functions['chooseName'].linear_out)
        functions['choosePos'] = ChoosePos(functions['chooseName'].linear_txt, functions['same'].linear_att, functions['chooseName'].linear_out)
        functions['compare'] = Compare(functions['chooseName'].linear_txt, functions['same'].linear_att, functions['chooseName'].linear_out)
        functions['common'] = Common(functions['same'].linear_att, functions['chooseName'].linear_out)
        functions['queryName'] = QueryName(functions['same'].linear_att)
        functions['queryAttr'] = QueryAttr(functions['same'].linear_att)
        functions['queryPos'] = QueryAttr(functions['same'].linear_att)
        functions['answerLogic'] = AnswerLogic()
        
    elif functions_type == 'coordinates':
        
        if use_ae:
            dim_txt = 200
            dim = 200
        else:
            if argument_type == 'lxmert' or argument_type == 'bert':
                dim_in = dim_txt = 768
            elif argument_type == 'fasttext':
                dim_in = dim_txt = 300
            if args.features == 'gqa':
                dim_vis = 2048
            else:
                dim_vis = 768
            if args.use_coordinates:
                dim_vis += 4

            dim = min(dim_txt, dim_vis)
        print('here', dim_txt, dim_vis, dim)
        functions = {'and': And(), 'or': Or(), 'exist': Exist(), 'fusion': Fusion(), 'same': Same(dim_txt, dim_vis, dim)}

        if args.select_2 == 'True':
            functions['select'] = Select_2(dim_txt=dim_txt, dim_vis=dim_vis, dim=dim)

        functions['sameAll'] = SameAll(functions['same'].linear_att, functions['same'].linear_txt, dim)
        functions['different'] = Different(functions['same'].linear_att, functions['same'].linear_txt,
                                           functions['same'].linear_out)
        functions['differentAll'] = DifferentAll(functions['same'].linear_att, functions['same'].linear_txt,
                                                 functions['sameAll'].linear_out)
        functions['answerLogic'] = AnswerLogic()
        functions['verifyRelSub'] = VerifyRelSub(dim_txt, dim_vis, dim)
        functions['verifyRelObj'] = VerifyRelObj(functions['verifyRelSub'].linear_att,
                                                 functions['verifyRelSub'].linear_txt, dim)
        functions['verifyPos'] = VerifyPos(functions['verifyRelSub'].linear_att, dim_txt, dim)
        functions['verifyAttr'] = VerifyAttr(functions['verifyRelSub'].linear_att, dim_txt, dim)

        functions['filterAttr'] = FilterAttr(functions['verifyAttr'].linear_txt, functions['select'].linear_vis, dim)
        functions['filterNot'] = FilterNot(functions['verifyAttr'].linear_txt, functions['select'].linear_vis, functions['filterAttr'].linear_out, dim)
        functions['filterPos'] = FilterPos(functions['verifyPos'].linear_txt, functions['select'].linear_vis, dim)

        functions['relateSub'] = RelateSub(functions['verifyRelSub'].linear_txt, functions['select'].linear_vis, dim_vis, dim)
        functions['relateObj'] = RelateObj(functions['relateSub'].linear_att, functions['verifyRelSub'].linear_txt, functions['select'].linear_vis, dim)
        functions['relateAttr'] = RelateAttr(functions['relateSub'].linear_att, functions['verifyRelSub'].linear_txt, functions['select'].linear_vis, dim)

        functions['chooseName'] = ChooseName(dim_txt, dim_vis, dim)
        functions['chooseAttr'] = ChooseAttr(functions['chooseName'].linear_att, functions['chooseName'].linear_txt, dim)
        functions['chooseRel'] = ChooseRel(functions['chooseName'].linear_att, functions['chooseName'].linear_txt, dim)
        functions['choosePos'] = ChoosePos(functions['chooseName'].linear_att, functions['chooseName'].linear_txt, dim)
        functions['compare'] = Compare(functions['chooseName'].linear_att, functions['chooseName'].linear_txt, dim)
        functions['common'] = Common(functions['chooseName'].linear_att, dim)
        functions['queryName'] = QueryName(dim_vis, dim)
        functions['queryAttr'] = QueryAttr(functions['queryName'].linear_att, dim)
        functions['queryPos'] = QueryPos(functions['queryName'].linear_att, dim)

    return functions
