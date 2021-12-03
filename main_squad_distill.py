import argparse
import torch
import torch.nn as nn
import os
from transformers import BertConfig, BertForQuestionAnswering, BertTokenizer
from transformers import DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer
#Fror Training 
from utils import set_seed, set_experiments, pkd_initialization
from dataset import load_squad_dataset
from squad_train import distill_train, evaluate

#MODEL_CLASSES = {
#    "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
#    "distilbert": (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer)
#}
##########Todo :  Different Architecture Setting
#   "roberta": (RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer),
#   "xlm": (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
#   "xlnet": (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
##5117
def parse_args():
    parser = argparse.ArgumentParser()

    ##Custom setting
    parser.add_argument('--exp_name', type=str, default='test',
                        help='Name of the experiment')
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", 
                        help="Whether to run eval on the dev set.")
    ##Distillation Options
    parser.add_argument("--exp_type", type=str, default="google",
                        help="Experiments Setting Type Paper! [pkd/distilbert]")
    parser.add_argument("--distil_loss",type=str, default="kd",
                        help="Distillation Loss type! [kd, pkd, structure]")
    parser.add_argument("--alpha", default=0.5, type=float,
                        help="Distillation loss linear weight. Only for distillation.")
    parser.add_argument("--temperature", default=2.0, type=float, 
                        help="Distillation temperature. Only for distillation.")

    parser.add_argument("--beta", default=10.0, type=float,
                        help="Distillation loss linear weight. Only for distillation.")

    parser.add_argument("--hidden_w", default=1.0, type=float,
                        help="Distillation loss linear weight. Only for distillation.")
    parser.add_argument("--attention_w", default=1.0, type=float,
                        help="Distillation loss linear weight. Only for distillation.")
    parser.add_argument("--embedding_w", default=1.0, type=float,
                        help="Distillation loss linear weight. Only for distillation.")

    parser.add_argument("--wrdist_w", default=1.0, type=float,
                        help="Distillation loss linear weight. Only for distillation.")
    parser.add_argument("--wrangle_w", default=1.0, type=float,
                        help="Distillation loss linear weight. Only for distillation.")
    parser.add_argument("--wtrdist_w", default=1.0, type=float,
                        help="Distillation loss linear weight. Only for distillation.")
    parser.add_argument("--wtrangle_w", default=1.0, type=float,
                        help="Distillation loss linear weight. Only for distillation.")
    parser.add_argument("--lossfunc", type=str, default="smooth",
                        help="l1loss or l2loss or smooth_l1loss.")
    parser.add_argument("--matching_strategy", type=str, default="emb+skip",
                        help="Experiments Setting Type Paper! [skip|last|once]")
    parser.add_argument("--window_size", type=int, default=5,
                        help ="Window Size For Angle")
    parser.add_argument("--distnorm", action="store_true", 
                        help="Whether to do normalize the distance matrix.")
    ##DataSet Setting
    parser.add_argument("--version_2", action="store_true",
                        help="If true, the SQuAD examples contain some that do not have an answer.")
    parser.add_argument("--data_dir", default="./datas/squad", type=str,
                        help="The input data dir. Should contain the .json files for the task."
                        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.")
    parser.add_argument("--train_file", default="train-v1.1.json", type=str,
                        help="The input training file. If a data dir is specified, will look for the file there"
                        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.")
    parser.add_argument("--predict_file", default="dev-v1.1.json", type=str,
                        help="The input evaluation file. If a data dir is specified, will look for the file there"
                        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                        "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                        "be truncated to this length.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start"
                        "and end predictions are not conditioned on one another.")
    parser.add_argument("--null_score_diff_threshold", type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")

    ##Option for dataloading
    parser.add_argument("--threads", type=int, default=1, 
                        help="multiple threads for converting example to features")
    
    ##Model Setting
    parser.add_argument("--teacher_type", default=None, type=str, required=True,
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

    ##Training Setting TODO: parallel the GPUs
    parser.add_argument("--per_gpu_batch_size", default=1, type=int, 
                        help="Batch size per GPU/CPU for training.")
    #parser.add_argument("--max_steps", default=-1, type=int,
    #                    help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
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
    parser.add_argument("--no_cuda", action="store_true", 
                        help="Whether not to use CUDA when available")
    parser.add_argument("--seed", type=int, default=42, 
                        help="random seed for initialization")
    parser.add_argument("--resume", type=str, default=None,
                        help="Continue Training, insert the stopped model name")
    parser.add_argument("--logging_steps", type=int, default=500, 
                        help="Log every X updates steps.")
    parser.add_argument("--eval_steps", type=int, default=500, 
                        help="Evaluation step for early stop")
    parser.add_argument("--save_steps", type=int, default=10000,
                        help="Save checkpoint every X updates steps.")
    return parser.parse_args()

args = parse_args()

logger, args.checkpoint_dir, args.result_dir = set_experiments(args, "squad-distill")
# Setup CUDA, GPU & distributed training
if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
else:  
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.n_gpu = 1

args.device = device

logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1))

set_seed(args.seed)

###TODO: Need to search
# Load pretrained model and tokenizer
if args.local_rank not in [-1, 0]:
    torch.distributed.barrier()

#Teacher and Student Model Setup
##TODO: GENERALIZE ON THE OTHER NETWORKS
##TODO:Extension To other Network
args.student_type = args.student_type.lower()
args.teacher_type = args.teacher_type.lower()

##TODO: Assertion
tokenizer = BertTokenizer.from_pretrained(args.teacher_path,
                                    do_lower_case=args.do_lower_case)

if args.teacher_type == 'bert':
    teacher_config = BertConfig.from_pretrained(args.teacher_path)
    teacher = BertForQuestionAnswering.from_pretrained(args.teacher_path, config=teacher_config)
else:
    raise NotImplementedError

if args.student_type == "distil_bert":
    student_config = BertConfig.from_pretrained(args.student_path+'/config.json')
    student = DistilBertForQuestionAnswering.from_pretrained(args.student_path, config=student_config)
elif args.student_type =="bert6":
    student_config = BertConfig.from_pretrained(args.student_path)
    student = BertForQuestionAnswering(config=student_config)

elif args.student_type =="bert_small":
    if args.student_path == "./download_models/mobile-bert":
        student_config = MobileBertConfig.from_pretrained(args.student_path)
        student = MobileBertForQuestionAnswering.from_pretrained(args.student_path, 
                                                    config=student_config)

    else:
        student_config = BertConfig.from_pretrained(args.student_path)
        student = BertForQuestionAnswering.from_pretrained(args.student_path, 
                                                    config=student_config)   
elif args.student_type == "tiny-bert":
    student = BertForQuestionAnswering.from_pretrained("nreimers/TinyBERT_L-4_H-312_v2")
    
else:
    NotImplementedError



if args.exp_type == "pkd":
    pkd_initialization(teacher, student)

###TODO: Need to search
if args.local_rank == 0:
    # Make sure only the first process in distributed training will download model & vocab
    torch.distributed.barrier()

teacher.to(args.device)
student.to(args.device)

logger.info("Training/Evaluation Parameters : ")
for attr, value in sorted(args.__dict__.items()):
    logger.info("\t{}={}".format(attr.upper(), value))

##TODO: Apex fp16
if args.do_train:
    train_dataset = load_squad_dataset(args, tokenizer, logger, evaluate=False, output_examples=False)
    global_step, train_loss, exact, f1  = distill_train(args, train_dataset, student, tokenizer, teacher, logger)
    logger.info(" global_step = %s, average loss = %s", global_step, train_loss)
    logger.info(" End Training ")
    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        logger.info("Saving last checkpoint to %s", args.result_dir)
        model_to_save = (student.module if hasattr(student, "module") else student)  
        model_to_save.save_pretrained(args.result_dir)
        torch.save(args, os.path.join(args.result_dir, "training_args.bin"))
        with open("results_squad_distil.txt", "a") as f:
            f.write("{}\t{}\t{}\n".format(args.exp_name, exact, f1))
        f.close()

if args.do_eval:
    results = {}
    # Load a trained model and vocabulary that you have fine-tuned
    logger.info("Evaluation Best Checkpoint")
    if args.student_type == "distil_bert":
        model = DistilBertForQuestionAnswering.from_pretrained(os.path.join(args.checkpoint_dir, "best_checkpoint"))
    elif args.student_type =="bert6":
        model = BertForQuestionAnswering.from_pretrained(os.path.join(args.checkpoint_dir, "best_checkpoint"))
    tokenizer = BertTokenizer.from_pretrained(os.path.join(args.checkpoint_dir, "best_checkpoint"), do_lower_case=args.do_lower_case)
    model.to(args.device)
    result = evaluate(args, model, tokenizer, logger)
    logger.info("Results : {}".format(results))
