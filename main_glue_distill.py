import argparse
import os
import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from transformers import glue_tasks_num_labels, glue_output_modes
from dataset import GlueDataset
from utils import set_seed, set_experiments, pkd_initialization
from glue_train import train, evaluate, predict_test, distill_train
import transformers
from transformers import AutoModel
from distil_config import change_args

def parse_args():
    parser = argparse.ArgumentParser()
    ##Custom setting
    parser.add_argument('--exp_name', type=str, default='distillation',
                        help='Name of the experiment')
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", 
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true", 
                        help="Whether to run eval on the test set.")
    parser.add_argument("--test_output_dir", type=str, default="",
                        help="Test output directory")

    ##Distillation Options
    parser.add_argument("--student_init", type=str, default="pretrained",
                        help="Experiments Setting Type Paper! [upper | pretrained]")
    parser.add_argument("--distil_loss",type=str, default="kd",
                        help="Distillation Loss type! \
                        [kd] \
                        [pkd]] \
                        [tiny_hidden, tiny_attention, tiny_embed] \
                        [wrdist, wrdist_window, wrangle, wrangle_window, ltrdist, ltrangle] \
                        Separate the distillation loss to be used with +. \
                        Ex)kd+wrdist+wrangle_window+ltrdist+ltrangle")
    
    ##For HintonKD or DistilBERT
    parser.add_argument("--alpha", default=0.5, type=float,
                        help="Distillation loss linear weight. Only for distillation.")
    parser.add_argument("--temperature", default=2.0, type=float, 
                        help="Distillation temperature. Only for distillation.")
    ##For PKD
    parser.add_argument("--beta", default=10.0, type=float,
                        help="Distillation loss linear weight. Only for distillation.")

    ##For TinyBERT
    parser.add_argument("--hidden_w", default=1.0, type=float,
                        help="Distillation loss linear weight. Only for distillation.")
    parser.add_argument("--attention_w", default=1.0, type=float,
                        help="Distillation loss linear weight. Only for distillation.")
    parser.add_argument("--embedding_w", default=1.0, type=float,
                        help="Distillation loss linear weight. Only for distillation.")
    parser.add_argument("--matching_strategy", type=str, default="emb+skip",
                        help="Layer alignment for PKD, TinyBERT! [skip|last|once]")
    
    ##Function configuration for CKD
    parser.add_argument("--window_size", type=int, default=21,
                        help ="Window Size For Angle")
    parser.add_argument("--distance", type=str, default="cos",
                        help="Definition of distance, cos or l2.")
    parser.add_argument("--distnorm", action="store_true", 
                        help="Whether to do normalize the distance matrix.")
    parser.add_argument("--lossfunc_d", type=str, default="kldiv",
                        help="kldiv, l1loss, l2loss or smoothl1.")
    parser.add_argument("--lossfunc_a", type=str, default="l2loss")

    ##Coefficient for CKD
    parser.add_argument("--wrdist_w", default=1.0, type=float,
                        help="Distillation loss linear weight. Coefficent for distance in word relation.")
    parser.add_argument("--wrangle_w", default=1.0, type=float,
                        help="Distillation loss linear weight. Coefficent for angle in word relation.")
    parser.add_argument("--ltrdist_w", default=1.0, type=float,
                        help="Distillation loss linear weight. Coefficent for distance in transforming relation.")
    parser.add_argument("--ltrangle_w", default=1.0, type=float,
                        help="Distillation loss linear weight. Coefficent for angle in transforming relation.")

    ##Matching Alignment for CKD
    parser.add_argument("--matching_strategy_wr", type=str, default="emb+skip",
                        help="Layer alignment for WR. Please refer matching_alignment function in utils.py")
    parser.add_argument("--matching_strategy_ltr", type=str, default="emb+skip",
                        help="Layer alignment for LTR. Please refer matching_alignment function in utils.py")

    ##Model Setting
    parser.add_argument("--teacher_type", default="bert", type=str, required=True,
                        help="Teacher type selected in the list")
    parser.add_argument("--student_type", default=None, type=str, required=True,
                        help="Student type selected in the list")
    parser.add_argument("--teacher_path", default=None, type=str,
                        help="Path to the already SQuAD fine-tuned teacher model. Only for distillation.")
    parser.add_argument("--student_path", default=None, type=str,
                        help="Path to the already SQuAD fine-tuned student model. Target Network")
    parser.add_argument("--tokenizer_name", default="", type=str, 
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--do_lower_case", action="store_true", 
                        help="Set this flag if you are using an uncased model.")

    ##Dataset Setting
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="Specify the task name in glue for training")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                        "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--data_dir", default="./datas/glue", type=str, 
                        help="Saved Data`s directory,"
                        "download from the huggingface transformer utils/download_glue_data.py")
    parser.add_argument("--augdata", action="store_true",
                        help="Use Augmented dataset")

    ##Training Setting
    parser.add_argument("--per_gpu_batch_size", default=1, type=int, 
                    help="Batch size per GPU/CPU for training.")
    parser.add_argument("--num_train_epochs", default=4.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--warmup_steps", default=0, type=int, 
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--weight_decay", default=0.0, type=float, 
                        help="Weight decay if we apply some.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, 
                        help="Max gradient norm.")

    ##Experiments Setting
    parser.add_argument("--local_rank", type=int, default=-1, 
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed for initialization")
    parser.add_argument("--resume", type=str, default=None,
                        help="Continue Training, insert the stopped model name")
    parser.add_argument("--logging_steps", type=int, default=100, 
                        help="Log every X updates steps.")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluation step for early stop")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    return parser.parse_args()

args = parse_args()
#We included specific hyperparameters of CKD in the BERT 6/768 setting.
args = change_args(args)

logger, args.checkpoint_dir, args.result_dir = set_experiments(args, "distil_{}".format(args.task_name))

if args.local_rank == -1:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.n_gpu = 1

args.device = device

logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        args.device,
        args.n_gpu,
        bool(args.local_rank != -1))

set_seed(args.seed)

if args.local_rank not in [-1, 0]:
    torch.distributed.barrier()
try:
    args.num_labels = glue_tasks_num_labels[args.task_name]
except KeyError:
    raise ValueError("Task not found: %s" % (args.task_name))

##Path for task-specifically pre-trained teacher for each tasks
args.teacher_path = os.path.join(args.teacher_path, args.task_name, "checkpoints/best_checkpoint")

##Path for dataset / if you want to use augmented dataset like TinyBERT
if args.augdata:
    args.data_dir = os.path.join(args.data_dir, (args.task_name+"_aug").upper())
else:
    args.data_dir = os.path.join(args.data_dir, args.task_name.upper())

tokenizer = BertTokenizer.from_pretrained(args.teacher_path,
                                    do_lower_case=args.do_lower_case)

if args.teacher_type == 'bert':
    teacher_config = BertConfig.from_pretrained(args.teacher_path)
    teacher = BertForSequenceClassification.from_pretrained(args.teacher_path, config=teacher_config)
else:
    raise NotImplementedError

##Download pre-trained small BERTs or pretrain with task-agnostic distillation
##Then input the path for specific smallBERTs

if args.resume == None:

    if args.student_type =="bert_small":
        student_config = BertConfig.from_pretrained(args.student_path,
                                                    num_labels=args.num_labels,
                                                    finetuning_task=args.task_name)

        student = BertForSequenceClassification.from_pretrained(args.student_path, 
                                                    config=student_config)

    ##If you perform the task-specific distillation on pre-trained tinyBERT
    elif args.student_type == "tiny_bert":
        student = BertForSequenceClassification.from_pretrained("nreimers/TinyBERT_L-4_H-312_v2",
                                                                num_labels=args.num_labels,
                                                                finetuning_task=args.task_name)
    else:
        NotImplementedError

    ##Initialize the student with the first layers of the teacher like PKD
    if args.student_init=="pkd":
        pkd_initialization(teacher, student)
    elif args.student_init=="pretrained":
        pass    ##Should correctly specify the path for pre-trained small BERT

else:
    student_config = BertConfig.from_pretrained(args.resume,
                                                num_labels=args.num_labels,
                                                finetuning_task=args.task_name)

    student = BertForSequenceClassification.from_pretrained(args.resume, 
                                                config=student_config)



args.s_hidden_size = student_config.hidden_size
args.t_hidden_size = teacher_config.hidden_size

if args.local_rank == 0:
    # Make sure only the first process in distributed training will download model & vocab
    torch.distributed.barrier()

teacher.to(args.device)
student.to(args.device)
logger.info("Training/Evaluation Parameters : ")
for attr, value in sorted(args.__dict__.items()):
    logger.info("\t{}={}".format(attr.upper(), value))

if args.do_train:
    train_dataset = GlueDataset(args, tokenizer=tokenizer, logger=logger, mode="train")
    dev_dataset = GlueDataset(args, tokenizer=tokenizer, logger=logger, mode="dev")
    global_step, train_loss, metric_name, eval_best = distill_train(args, train_dataset, dev_dataset, student, teacher, tokenizer, logger)
    logger.info(" global_step = %s, average loss = %s", global_step, train_loss)
    logger.info(" End Training ")
    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        logger.info("Saving model checkpoint to %s", args.result_dir)
        last_output_dir = os.path.join(args.checkpoint_dir, "last_checkpoint")
        model_to_save = student.module if hasattr(student, "module") else student
        model_to_save.save_pretrained(last_output_dir)
        tokenizer.save_pretrained(last_output_dir)
        torch.save(args, os.path.join(last_output_dir, "training_args.bin"))
        os.makedirs("./results",exist_ok=True)
        with open("./results/dev_results_{}_distil.txt".format(args.task_name), "a") as f:
            f.write("{}\t{}\t{}\n".format(args.exp_name, metric_name, eval_best))
        f.close()

if args.do_eval and args.local_rank in [-1, 0]:
    dev_dataset = GlueDataset(args, tokenizer=tokenizer, logger=logger, mode="dev")
    logger.info("Loading checkpoint %s for evaluation", args.model_path)
    logger.info("Evaluate the following checkpoints: %s", args.model_path)
    if args.student_type == "distil_bert":
        config = BertConfig.from_pretrained(os.path.join(args.checkpoint_dir, "best_checkpoint")+'/config.json')
        model = DistilBertForSequenceClassification.from_pretrained(os.path.join(args.checkpoint_dir, "best_checkpoint"), config=config)
    else:
        config = BertConfig.from_pretrained(os.path.join(args.checkpoint_dir, "best_checkpoint")+'/config.json')
        model = BertForSequenceClassification.from_pretrained(os.path.join(args.checkpoint_dir, "best_checkpoint"), config=config)
    model.to(args.device)
    # Evaluate
    results = evaluate(args, model, dev_dataset, logger)
    logging.info("Results : {}".format(results))

if args.do_test and args.local_rank in [-1, 0]:
    if args.student_type == "distil_bert":
        config = BertConfig.from_pretrained(os.path.join(args.model_path, "best_checkpoint")+'/config.json')
        model = DistilBertForSequenceClassification.from_pretrained(args.model_path, config=config)
    else:
        config = BertConfig.from_pretrained(os.path.join(args.model_path, "best_checkpoint")+'/config.json')
        model = BertForSequenceClassification.from_pretrained(args.model_path, config=config)

    model.to(args.device)
    output_mode = glue_output_modes[args.task_name]
    logger.info("Test Trained Network in \n {}".format(args.model_path))
    test_dataset = GlueDataset(args, tokenizer=tokenizer, logger=logger, mode="test")
    prediction = predict_test(args, model, test_dataset, logger)

    output_test_file = os.path.join(args.test_output_dir, 
                                    "{}.tsv".format(args.task_name.upper()))

    with open(output_test_file, "w") as f:
        f.write("index\tprediction\n")
        for index, item in enumerate(prediction):
            if output_mode == "regression":
                f.write("%d\t%3.3f\n" % (index, item))
            else:
                item = test_dataset.get_labels()[item]
                f.write("%d\t%s\n" % (index, item))
        f.close()
