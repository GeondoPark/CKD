import os

##6layer-512
class COLA:
    alpha=0.7
    temperature=3.0
    wrdist_w=1
    wtrdist_w=1
    wrangle_w=10
    wtrangle_w=10
    learning_rate=5e-5
    num_train_epochs=4
    eval_steps=100

class SST2:
    alpha=0.9
    temperature=4.0
    wrdist_w=100
    wtrdist_w=100
    wrangle_w=1000
    wtrangle_w=1000
    learning_rate=4e-5
    num_train_epochs=4
    eval_steps=500

class RTE:
    alpha=0.9
    temperature=3.0
    wrdist_w=1
    wtrdist_w=1
    wrangle_w=10
    wtrangle_w=10
    learning_rate=5e-5
    num_train_epochs=6
    eval_steps=500

class QNLI:
    alpha=0.9
    temperature=3.0
    wrdist_w=10
    wtrdist_w=10
    wrangle_w=1
    wtrangle_w=1
    learning_rate=4e-5
    num_train_epochs=4
    eval_steps=2000

class MRPC:
    alpha=0.7
    temperature=4.0
    wrdist_w=1
    wtrdist_w=1
    wrangle_w=1
    wtrangle_w=1
    learning_rate=3e-5
    num_train_epochs=4
    eval_steps=500

class STSB:
    alpha=0.9
    temperature=1.0
    wrdist_w=0.1
    wtrdist_w=0.1
    wrangle_w=0.1
    wtrangle_w=0.1
    learning_rate=5e-5
    num_train_epochs=4
    eval_steps=100

class MNLI:
    alpha=0.7
    temperature=4.0
    wrdist_w=100
    wtrdist_w=100
    wrangle_w=1
    wtrangle_w=1
    learning_rate=4e-5
    num_train_epochs=4
    eval_steps=2000

class QQP:
    alpha=0.9
    temperature=2.0
    wrdist_w=10
    wtrdist_w=10
    wrangle_w=10
    wtrangle_w=10
    learning_rate=5e-5
    num_train_epochs=4
    eval_steps=2000

def change_args(args):
    if args.task_name=='cola':
        config=COLA
    elif args.task_name=='sst-2':
        config=SST2
    elif args.task_name=='rte':
        config=RTE
    elif args.task_name=='qnli':
        config=QNLI
    elif args.task_name=='mrpc':
        config=MRPC
    elif args.task_name=='sts-b':
        config=STSB
    elif args.task_name=='mnli':
        config=MNLI
    elif args.task_name=='qqp':
        config=QQP
    else:
        raise NotImplementedError

    args.alpha=config.alpha
    args.temperature=config.temperature
    args.wrdist_w=config.wrdist_w
    args.wtrdist_w=config.wtrdist_w
    args.wrangle_w=config.wrangle_w
    args.wtrangle_w=config.wtrangle_w
    args.learning_rate=config.learning_rate
    args.num_train_epochs=config.num_train_epochs
    args.eval_steps=config.eval_steps
    return args