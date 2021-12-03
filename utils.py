import torch
import random
import numpy as np
import os
import logging
import datetime
import math

def set_seed(seed):
    if seed is None:
        seed = 0
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_experiments(args, prefix=""):
    base_dir = os.path.join('./experiments', "exp_"+prefix, args.exp_name)
    os.makedirs(base_dir,exist_ok=True)
    checkpoint_dir = os.path.join(base_dir, 'checkpoints')
    os.makedirs(checkpoint_dir,exist_ok=True)
    code_dir = os.path.join(base_dir, 'code_backup')
    os.makedirs(code_dir, exist_ok=True)
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    ###Code Backup##
    ##To do ########
    ################
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.addHandler(logging.FileHandler(base_dir+'/logs.txt'))
    logger.addHandler(logging.StreamHandler())
    logger.info(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    return logger, checkpoint_dir, results_dir

def pkd_initialization(teacher, student):
    assert len(teacher.bert.encoder.layer) > len(student.bert.encoder.layer)
    student_dict = student.state_dict()
    pretrained_dict = {}
    for k, v in teacher.state_dict().items():
        if 'qa_outputs' in k:
            continue
        if k in student_dict:
            pretrained_dict[k] = v
    student_dict.update(pretrained_dict)
    student.load_state_dict(student_dict)

def matching_alignment(t_embed, s_embed, matching_strategy, only_cls=False):
    def compute_gcd(x, y):
        while y != 0:
            (x, y) = (y, x % y)
        return x
    assert matching_strategy in ["skip","last", "final", "att_skip", "emb+final", "emb", "emb+skip", "triple"]
    if matching_strategy == "att_skip":
        assert (len(t_embed)) % (len(s_embed)) == 0
        interval = int((len(t_embed))/(len(s_embed)))
        t_embed = torch.stack(t_embed[0::interval], dim=1)
        s_embed = torch.stack(s_embed[0:], dim=1)

    elif matching_strategy == "skip":
        gcd = compute_gcd(len(t_embed)-1, len(s_embed)-1)
        step_t = (len(t_embed)-1) / gcd
        step_s = (len(s_embed)-1) / gcd
        t_embed = torch.stack(t_embed[step_t::step_t], dim=1)
        s_embed = torch.stack(s_embed[step_s::step_s], dim=1)

    elif matching_strategy == "emb+skip":
        gcd = compute_gcd(len(t_embed)-1, len(s_embed)-1)
        step_t = int((len(t_embed)-1) / gcd)
        step_s = int((len(s_embed)-1) / gcd)
        t_embed = torch.cat((t_embed[0].unsqueeze(1),torch.stack(t_embed[step_t::step_t], dim=1)), dim=1)
        s_embed = torch.cat((s_embed[0].unsqueeze(1),torch.stack(s_embed[step_s::step_s], dim=1)), dim=1)

    elif matching_strategy == "last":
        start = len(t_embed) - len(s_embed)
        t_embed = torch.stack(t_embed[7:], dim=1)
        s_embed = torch.stack(s_embed[1:], dim=1)

    elif matching_strategy == "final":
        t_embed = t_embed[-1].unsqueeze(1)
        s_embed = s_embed[-1].unsqueeze(1)

    elif matching_strategy == "emb":
        t_embed = t_embed[0].unsqueeze(1)
        s_embed = s_embed[0].unsqueeze(1)

    elif matching_strategy == "emb+final":
        t_embed = torch.stack((t_embed[0], t_embed[-1]), dim=1)
        s_embed = torch.stack((s_embed[0], s_embed[-1]), dim=1)

    elif matching_strategy == "triple":
        t_middle = int((len(t_embed)-1)/2)
        s_middle = int((len(s_embed)-1)/2)
        t_embed = torch.stack((t_embed[0], t_embed[t_middle], t_embed[-1]), dim=1)
        s_embed = torch.stack((s_embed[0], s_embed[s_middle], s_embed[-1]), dim=1)   

    else:
        raise NotImplementedError
    if only_cls:
        return t_embed[:,:,0,:], s_embed[:,:,0,:]
    return t_embed, s_embed