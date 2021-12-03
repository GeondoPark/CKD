import os
import time
import torch
from transformers import squad_convert_examples_to_features 
from transformers import glue_convert_examples_to_features
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
from transformers.data.processors.glue import glue_processors, glue_output_modes
from torch.utils.data.dataset import Dataset
from transformers import XLMRobertaTokenizer
from transformers import BartTokenizer, BartTokenizerFast
from transformers import RobertaTokenizer, RobertaTokenizerFast
from transformers import PreTrainedTokenizer

def load_squad_dataset(args, tokenizer, logger, evaluate=False, output_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    input_file = args.predict_file if evaluate else args.train_file
    processed_data_path = os.path.join(args.data_dir, "preprocessed_{}_bert-base-uncased_{}".format(
                                                    "dev" if evaluate else "train",
                                                    str(args.max_seq_length)))

    if os.path.exists(processed_data_path):
        logger.info("Loading features from cached file %s", processed_data_path)
        features_and_dataset = torch.load(processed_data_path)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_file)
        processor = SquadV2Processor() if args.version_2 else SquadV1Processor()
        if evaluate:
            examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file)
        else:
            examples = processor.get_train_examples(args.data_dir, filename=args.train_file)

        ##TODO: Need to know if this returns the feature from BERT? with PDB
        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=args.threads,
            )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into prerocessed file %s", processed_data_path)
            torch.save({"features": features, "dataset": dataset, "examples": examples}, processed_data_path)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset

class GlueDataset(Dataset):
    """
    From Hugging Face
    """
    def __init__(self, args, tokenizer, logger, mode="train"):
        self.args = args
        self.processor = glue_processors[args.task_name]()
        self.output_mode = glue_output_modes[args.task_name]
        label_list = self.processor.get_labels()

        if args.task_name in ["mnli", "mnli-mm"] and tokenizer.__class__ in (
            RobertaTokenizer,
            RobertaTokenizerFast,
            XLMRobertaTokenizer,
            BartTokenizer,
            BartTokenizerFast,
        ):
            label_list[1], label_list[2] = label_list[2], label_list[1]

        self.label_list = label_list

        # Load data features from cache or dataset file
        preprocessed_features_file = os.path.join(args.data_dir,"preprocessed_{}_{}_{}_{}".format(
                mode, tokenizer.__class__.__name__, str(args.max_seq_length), args.task_name))


        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.

        if os.path.exists(preprocessed_features_file):
            start = time.time()
            self.features = torch.load(preprocessed_features_file)
            logger.info(f"Loading features from cached file {preprocessed_features_file} [took %.3f s]", time.time() - start)
        else:
            logger.info(f"Creating features from dataset file at {args.data_dir}")
            if mode == "train":
                examples = self.processor.get_train_examples(args.data_dir)
            elif mode == "dev":
                examples = self.processor.get_dev_examples(args.data_dir)
            else:
                examples = self.processor.get_test_examples(args.data_dir)

            self.features = glue_convert_examples_to_features(
                examples,
                tokenizer,
                task=args.task_name,
                max_length=args.max_seq_length,
                label_list=label_list,
                output_mode=self.output_mode)

            start = time.time()
            torch.save(self.features, preprocessed_features_file)
            logger.info("Saving features into cached file %s [took %.3f s]", preprocessed_features_file, time.time() - start)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]

    def get_labels(self):
        return self.label_list