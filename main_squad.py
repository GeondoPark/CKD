import os
import argparse
import torch
import torch.nn as nn
from transformers import BertConfig, BertForQuestionAnswering, BertTokenizer
from dataset import load_squad_dataset
from utils import set_seed, set_experiments
from squad_train import train

def parse_args():
    parser = argparse.ArgumentParser()
    ##Custom setting
    parser.add_argument('--exp_name', type=str, default='test',
                        help='Name of the experiment')
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", 
                        help="Whether to run eval on the dev set.")

    ##DataSet Setting
    parser.add_argument("--version_2", action="store_true",
                        help="If true, the SQuAD examples contain some that do not have an answer.")
    parser.add_argument("--data_dir", default="./datas", type=str,
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
    parser.add_argument("--threads", type=int, default=8, 
                        help="multiple threads for converting example to features")
    
    ##Model Setting
    parser.add_argument("--model_type", default="bert", type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--model_path", default=None, type=str,
                        help="Path to the already SQuAD fine-tuned student model. Target Network")
    parser.add_argument("--tokenizer_name", default="", type=str, 
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--do_lower_case", action="store_true", 
                        help="Set this flag if you are using an uncased model.")

    ##Training Setting TODO: parallel the GPUs
    parser.add_argument("--per_gpu_batch_size", default=32, type=int, 
                        help="Batch size per GPU/CPU for training.")
    #parser.add_argument("--max_steps", default=-1, type=int,
    #                    help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, 
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
    parser.add_argument("--seed", type=int, default=42, 
                        help="random seed for initialization")
    parser.add_argument("--resume", type=str, default=None,
                        help="Continue Training, insert the stopped model name")
    parser.add_argument("--logging_steps", type=int, default=500, 
                        help="Log every X updates steps.")
    parser.add_argument("--eval_steps", type=int, default=500, 
                        help="Evaluation step for early stop")
    parser.add_argument("--save_steps", type=int, default=1000000,
                        help="Save checkpoint every X updates steps.")

    return parser.parse_args()

args = parse_args()

logger, args.checkpoint_dir, args.result_dir = set_experiments(args, "squad")

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
        device,
        args.n_gpu,
        bool(args.local_rank != -1))

set_seed(args.seed)

    # Make sure only the first process in distributed training will download model & vocab
if args.local_rank not in [-1, 0]:
    torch.distributed.barrier()

#Extend to other networks
if args.model_type.lower() == "bert":
    config = BertConfig.from_pretrained(args.model_path)
    tokenizer = BertTokenizer.from_pretrained(args.model_path,
                                            do_lower_case=args.do_lower_case)
    model = BertForQuestionAnswering.from_pretrained(args.model_path,
                                            config=config)

if args.local_rank == 0:
    # Make sure only the first process in distributed training will download model & vocab
    torch.distributed.barrier()

model.to(args.device)
logger.info("Training/Evaluation Parameters : ")
for attr, value in sorted(args.__dict__.items()):
    logger.info("\t{}={}".format(attr.upper(), value))

#TODO : Apex fp16?

if args.do_train:
    train_dataset = load_squad_dataset(args, tokenizer, logger, evaluate=False, output_examples=False)
    global_step, train_loss = train(args, train_dataset, model, tokenizer, logger)
    logger.info(" global_step = %s, average loss = %s", global_step, train_loss)
    logger.info(" End Training ")
    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        logger.info("Saving model checkpoint to %s", args.result_dir)
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.result_dir)
        tokenizer.save_pretrained(args.result_dir)
        torch.save(args, os.path.join(args.result_dir, "training_args.bin"))

if args.do_eval:
    results = {}
    # Load a trained model and vocabulary that you have fine-tuned
    logger.info("Evaluation Best Checkpoint")
    model = DistilBertForQuestionAnswering.from_pretrained(os.path.join(args.checkpoint_dir, "best_checkpoint"))
    tokenizer = BertTokenizer.from_pretrained(os.path.join(args.checkpoint_dir, "best_checkpoint"), do_lower_case=args.do_lower_case)
    model.to(args.device)
    result = evaluate(args, model, tokenizer, logger)
    logger.info("Results : {}".format(results))