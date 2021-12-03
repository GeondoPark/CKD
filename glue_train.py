import os
from platform import dist
import timeit
import numpy as np
import torch
import torch.nn as nn
from dataset import GlueDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from loss import *
from utils import matching_alignment
from tensorboardX import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup, glue_compute_metrics, glue_output_modes
from transformers.data.data_collator import default_data_collator
from transformers.trainer_utils import EvalPrediction, PredictionOutput
import time

def get_eval_metric(task_name, results):
    if task_name == "cola":
        metric_name = 'mcc'
    if task_name == "mnli":
        metric_name = 'mnli/acc'
    if task_name == "mnli-mm":
        metric_name = 'mnli-mm/acc'
    if task_name == "mrpc":
        metric_name = 'f1'
    if task_name == "qqp":
        metric_name = 'acc'
    if task_name in ["qnli", "rte", "sst-2", "wnli"]:
        metric_name = 'acc'
    if task_name == "sts-b":
        metric_name = 'spearmanr'
    current_eval = results[metric_name]
    return current_eval, metric_name

def build_compute_metrics_fn(task_name: str):
    output_mode = glue_output_modes[task_name]
    def compute_metrics_fn(p: EvalPrediction):
        if output_mode == "classification":
            preds = np.argmax(p.predictions, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(p.predictions)
        return glue_compute_metrics(task_name, preds, p.label_ids)
    return compute_metrics_fn

def train(args, train_dataset, dev_dataset, model, tokenizer, logger):
    ##Vanilla Training Code 
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join("./tb_logs/", args.task_name, args.exp_name))

    args.train_batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, 
                                sampler=train_sampler, 
                                batch_size=args.train_batch_size,
                                collate_fn=default_data_collator)

    t_total = int(len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0}]
    
    optimizer = AdamW(optimizer_grouped_parameters, 
                        lr=args.learning_rate, 
                        eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                num_warmup_steps=args.warmup_steps, 
                                num_training_steps=t_total)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                    device_ids=[args.local_rank], 
                                    output_device=args.local_rank, 
                                    find_unused_parameters=True)

    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    # Train!
    logger.info("*****Start Training*****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_batch_size)
    
    epoch = 0
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    if args.resume is not None:
        if os.path.exists(args.model_name_or_path):
            try:
                optimizer.load_state_dict(torch.load(os.path.join(args.resume, "optimizer.pt")))
                scheduler.load_state_dict(torch.load(os.path.join(args.resume, "scheduler.pt")))
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                logger.info("  Starting fine-tuning.")

    train_loss = 0.0
    logging_loss = 0.0
    best_eval = 0.0
    model.zero_grad()

    train_iterator = trange(epochs_trained, int(args.num_train_epochs), 
                        desc="Epoch", 
                        disable=args.local_rank not in [-1, 0])
    loss_log = {}
    loss_log["ce"] = 0.0

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, inputs in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            ##When resuming,may be needed
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(args.device)
        
            outputs = model(**inputs)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            train_loss += loss.item()
            loss_log["ce"] = train_loss

            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                len(epoch_iterator)<=args.gradient_accumulation_steps 
                and (step+1)==len(epoch_iterator)):

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logger.info("[TRAIN] Current [Epoch/Iter] : [{}|{}/{}|{}] | LR - {:.7f}"
                                                .format(epoch, args.num_train_epochs, step, len(epoch_iterator),scheduler.get_lr()[0]))
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    train_text = "[TRAIN] Train Loss : "
                    for name, value in loss_log.items():
                        tb_writer.add_scalar("loss_{}".format(name), (value / args.logging_steps), global_step)
                        train_text += "{} : {:.8f} |".format(name.upper(), (value / args.logging_steps))
                        loss_log[name] = 0.0
                    logger.info(train_text)

                if args.local_rank == -1 and args.eval_steps > 0 and ((global_step % args.eval_steps == 0) or (global_step == t_total)):
                    results = evaluate(args, model, dev_dataset, tokenizer, logger)

                    if args.task_name == "mnli":
                        current_eval_list = []
                        metric_name_list = []
                        for (i, result) in enumerate(results):
                            for key, value in result.items():
                                tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                            if i == 0:
                                args.task_name = "mnli"
                            else:
                                args.task_name = "mnli-mm"
                            eval_temp, metric_temp = get_eval_metric(args.task_name, result)
                            current_eval_list.append(eval_temp)
                            metric_name_list.append(metric_temp)
                        args.task_name = "mnli"

                    else:
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        current_eval, metric_name = get_eval_metric(args.task_name, results)
                    if args.task_name == "mnli":
                        current_eval = sum(current_eval_list)
            
                    if current_eval > best_eval:
                        if args.task_name == "mnli":
                            best_eval_list[0] = current_eval_list[0]
                            best_eval_list[1] = current_eval_list[1]
                        best_eval = current_eval
                        best_output_dir = os.path.join(args.checkpoint_dir, "best_checkpoint")
                        model_to_save = (model.module if hasattr(model, "module") else model)
                        model_to_save.save_pretrained(best_output_dir)
                        tokenizer.save_pretrained(best_output_dir)
                        torch.save(args, os.path.join(best_output_dir, "training_args.bin"))
                        torch.save(optimizer.state_dict(), os.path.join(best_output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(best_output_dir, "scheduler.pt"))
                        logger.info("Best Model Saving optimizer and scheduler states to %s", best_output_dir)
                    if args.task_name == "mnli":
                        for (j, value) in enumerate(current_eval_list):
                            logger.info("Current Results : {} - {}".format(metric_name_list[j], value))                    
                            logger.info("Best Results : {} - {}".format(metric_name_list[j], best_eval_list[j]))
                    else:
                        logger.info("Current Results : {} - {}".format(metric_name, current_eval))                    
                        logger.info("Best Results : {} - {}".format(metric_name, best_eval))

                # Save model checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.checkpoint_dir, "checkpoint-{}".format(global_step))
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    if args.task_name == "mnli":
        logger.info("Fianl Best Results : {} - {}".format(metric_name_list, best_eval_list))
        return global_step, loss_log, metric_name_list, best_eval_list
    else:
        logger.info("Fianl Best Results : {} - {}".format(metric_name, best_eval))
        return global_step, loss_log, metric_name, best_eval

def distill_train(args, train_dataset, dev_dataset, student, teacher, tokenizer, logger):
    ##Distillation Training
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join("./tb_logs/", args.task_name, args.exp_name))

    args.train_batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, 
                                sampler=train_sampler, 
                                batch_size=args.train_batch_size,
                                collate_fn=default_data_collator)

    t_total = int(len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs)

    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {"params": [p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay},
        {"params": [p for n, p in student.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0}]
    
    optimizer = AdamW(optimizer_grouped_parameters, 
                        lr=args.learning_rate, 
                        eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                num_warmup_steps=args.warmup_steps, 
                                num_training_steps=t_total)

    if args.n_gpu > 1:
        teacher = nn.DataParallel(teacher)
        student = nn.DataParallel(student)

    if args.local_rank != -1:
        teacher = torch.nn.parallel.DistributedDataParallel(
            teacher, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        student = torch.nn.parallel.DistributedDataParallel(
            student, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)


    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    # Train!
    logger.info("*****Start Training*****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_batch_size)
    
    epoch = 0
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    if args.resume is not None:
        if os.path.exists(args.model_name_or_path):
            try:
                optimizer.load_state_dict(torch.load(os.path.join(args.resume, "optimizer.pt")))
                scheduler.load_state_dict(torch.load(os.path.join(args.resume, "scheduler.pt")))
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                logger.info("  Starting fine-tuning.")

    loss_log = {}
    loss_log["ce"] = 0.0
    for name in args.distil_loss.split('+'):
        loss_log[name] = 0.0
    
    best_eval = 0.0
    #For MNLI task, MNLI-m and MNLI-mm
    best_eval_list = [0.0, 0.0]

    student.zero_grad()
    teacher.eval()

    train_iterator = trange(epochs_trained, int(args.num_train_epochs), 
                        desc="Epoch", 
                        disable=args.local_rank not in [-1, 0])

    if "kd" in args.distil_loss.split('+'):
        kd_function = DistillKL(T=args.temperature)

    if "pkd" in args.distil_loss.split('+'):
        ##pkd loss`s input :  batch_size, n_layer, hidden_size
        pkd_function = PKD_loss(p=2, normalize=True)

    if "wrdist" in args.distil_loss.split('+'):
        wrdist_function = WR_Dist()

    if "wrdist_window" in args.distil_loss.split('+'):
        wrdist_function = WR_Dist_window()

    if "wrangle" in args.distil_loss.split('+'):
        wrang_function = WR_Angle()
        if args.n_gpu > 1:
            wrang_function = nn.DataParallel(wrang_function)

    if "wrangle_window" in args.distil_loss.split('+'):
        wrang_function = WR_Angle_window()
        if args.n_gpu > 1:
            wrang_function = nn.DataParallel(wrang_function)

    if "ltrdist" in args.distil_loss.split('+'):
        ltrdist_function = LTR_Dist()

    if "ltrangle" in args.distil_loss.split('+'):
        ltrang_function = LTR_Angle()
        if args.n_gpu > 1:
            ltrang_function = nn.DataParallel(ltrang_function)

    if "tiny_hidden" in args.distil_loss.split('+'):
        hidden_function = Hidden_mse(args.s_hidden_size, args.t_hidden_size).to(args.device)

    if "tiny_attention" in args.distil_loss.split('+'):
        att_function = Attention_mse()

    if "tiny_embed" in args.distil_loss.split('+'):
        embed_function = Embedding_mse(args.s_hidden_size, args.t_hidden_size).to(args.device)

    for epoch in train_iterator:

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        start_time= time.time()
        for step, inputs in enumerate(epoch_iterator):
            ## Skip past any already trained steps if resuming training
            ## When resuming,may be needed
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            student.train()
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(args.device)

            inputs["output_attentions"] = True
            inputs["output_hidden_states"] = True

            with torch.no_grad():
                _, teacher_logits, t_hidden, t_atts = teacher(**inputs)
            loss, student_logits, s_hidden, s_atts= student(**inputs)

            loss_log["ce"] += loss.mean().item()

            if "kd" in args.distil_loss.split('+'):
                if args.task_name == "sts-b":
                    loss_kd = kd_function(student_logits, teacher_logits, mode="regression")
                else:
                    loss_kd = kd_function(student_logits, teacher_logits)
                #loss = args.alpha * loss_kd + (1-args.alpha) * loss
                loss = loss_kd
                loss_log["kd"] += loss_kd.item()

            if "pkd" in args.distil_loss.split('+'):
                ##pkd loss`s input :  batch_size, n_layer, hidden_size
                t_patience, s_patience = matching_alignment(t_hidden, s_hidden, 
                                                            matching_strategy=args.matching_strategy,
                                                            only_cls=True)
                pkd = pkd_function(t_patience, s_patience)
                loss = loss + args.beta * pkd
                loss_log["pkd"] += pkd.item()
    
            if "pkdall" in args.distil_loss.split('+'):
                ##pkd loss`s input :  batch_size, n_layer, hidden_size
                t_patience, s_patience = matching_alignment(t_hidden, s_hidden, 
                                                            matching_strategy=args.matching_strategy,
                                                            only_cls=False)
                pkd = pkd_function(t_patience, s_patience)
                loss = loss + args.beta * pkd
                loss_log["pkdall"] += pkd.item()

            if "tiny_hidden" in args.distil_loss.split('+'):
                t_hid, s_hid = matching_alignment(t_hidden, s_hidden,
                                                        matching_strategy=args.matching_strategy,
                                                        only_cls=False)

                hidden_function.train()
                hidden_loss = hidden_function(s_hid, t_hid)
                loss = loss + args.hidden_w * hidden_loss
                loss_log["tiny_hidden"] += hidden_loss.item()

            #This could not be used if the matching_startegy in tiny_hidden is emb+skip
            if "tiny_embed" in args.distil_loss.split('+'):
                t_embed, s_embed = matching_alignment(t_hidden, s_hidden,
                                                        matching_strategy="emb",
                                                        only_cls=False)
                embed_function.train()
                embed_loss = embed_function(s_embed, t_embed)
                loss = loss + args.embedding_w * embed_loss
                loss_log["tiny_embed"] += embed_loss.item()

            if "tiny_attention" in args.distil_loss.split('+'):

                t_atts, s_atts = matching_alignment(t_atts, s_atts,
                                                    matching_strategy="att_skip",
                                                    only_cls=False)
                att_loss = att_function(s_atts, t_atts)
                loss = loss + args.attention_w * att_loss
                loss_log["tiny_attention"] += att_loss.item()

            if "wrdist" in args.distil_loss.split('+'):
                    t_embed, s_embed = matching_alignment(t_hidden, s_hidden, 
                                                            matching_strategy=args.matching_strategy_wr,
                                                            only_cls=False)

                    wrdist_loss = wrdist_function(t_embed, s_embed, inputs["attention_mask"], 
                                                distance=args.distance,
                                                lossfunc=args.lossfunc_d, 
                                                normalize=args.distnorm)

                    loss = loss + args.wrdist_w * wrdist_loss
                    loss_log["wrdist"] += wrdist_loss.item()

            if "wrangle" in args.distil_loss.split('+'):
                t_embed, s_embed = matching_alignment(t_hidden, s_hidden, 
                                                        matching_strategy=args.matching_strategy_wr,
                                                        only_cls=False)
                wrang_loss = wrang_function(t_embed, s_embed, inputs["attention_mask"], 
                                        lossfunc=args.lossfunc_a)
                loss = loss + args.wrangle_w * wrang_loss
                loss_log["wrangle"] += wrang_loss.mean().item()

            if "wrangle_window" in args.distil_loss.split('+'):
                t_embed, s_embed = matching_alignment(t_hidden, s_hidden, 
                                                        matching_strategy=args.matching_strategy_wr,
                                                        only_cls=False)
                wrang_loss = wrang_function(t_embed, s_embed, inputs["attention_mask"], 
                                        lossfunc=args.lossfunc_a,
                                        window=args.window_size)

                loss = loss + args.wrangle_w * wrang_loss
                loss_log["wrangle_window"] += wrang_loss.mean().item()

            if "ltrdist" in args.distil_loss.split('+'):
                t_embed, s_embed = matching_alignment(t_hidden, s_hidden,
                                                        matching_strategy=args.matching_strategy_ltr,
                                                        only_cls=False)
                ltrdist_loss = ltrdist_function(t_embed, s_embed, inputs["attention_mask"],
                                            distance=args.distance,
                                            lossfunc=args.lossfunc_d,
                                            normalize=args.distnorm)
                                    
                loss = loss + args.ltrdist_w * ltrdist_loss
                loss_log["ltrdist"] += ltrdist_loss.item()

            if "ltrangle" in args.distil_loss.split('+'):
                t_embed, s_embed = matching_alignment(t_hidden, s_hidden, 
                                                        matching_strategy=args.matching_strategy_ltr,
                                                        only_cls=False)
                ltrang_loss = ltrang_function(t_embed, s_embed, inputs["attention_mask"],
                                            lossfunc=args.lossfunc_a)            
                loss = loss + args.ltrangle_w * ltrang_loss
                loss_log["ltrangle"] += ltrang_loss.mean().item()

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                len(epoch_iterator)<=args.gradient_accumulation_steps 
                and (step+1)==len(epoch_iterator)):

                torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                student.zero_grad()
                global_step += 1
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logger.info("[TRAIN] Current [Epoch/Iter] : [{}|{}/{}|{}] | LR - {:.7f}"
                                                .format(epoch, args.num_train_epochs, step, len(epoch_iterator),scheduler.get_lr()[0]))
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    train_text = "[TRAIN] Train Loss : "
                    for name, value in loss_log.items():
                        tb_writer.add_scalar("loss_{}".format(name), (value / args.logging_steps), global_step)
                        train_text += "{} : {:.8f} |".format(name.upper(), (value / args.logging_steps))
                        loss_log[name] = 0.0
                    logger.info(train_text)

                if args.local_rank == -1 and args.eval_steps > 0 and ((global_step % args.eval_steps == 0) or (global_step == t_total)):
                    results = evaluate(args, student, dev_dataset, tokenizer, logger)

                    if args.task_name == "mnli":
                        current_eval_list = []
                        metric_name_list = []
                        for (i, result) in enumerate(results):
                            for key, value in result.items():
                                tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                            if i == 0:
                                args.task_name = "mnli"
                            else:
                                args.task_name = "mnli-mm"
                            eval_temp, metric_temp = get_eval_metric(args.task_name, result)
                            current_eval_list.append(eval_temp)
                            metric_name_list.append(metric_temp)
                        args.task_name = "mnli"

                    else:
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        current_eval, metric_name = get_eval_metric(args.task_name, results)
                    
                    if args.task_name == "mnli":
                        current_eval = sum(current_eval_list)
            
                    if current_eval > best_eval:
                        if args.task_name == "mnli":
                            best_eval_list[0] = current_eval_list[0]
                            best_eval_list[1] = current_eval_list[1]
                        best_eval = current_eval
                        best_output_dir = os.path.join(args.checkpoint_dir, "best_checkpoint")
                        model_to_save = (student.module if hasattr(student, "module") else student)
                        model_to_save.save_pretrained(best_output_dir)
                        tokenizer.save_pretrained(best_output_dir)
                        torch.save(args, os.path.join(best_output_dir, "training_args.bin"))
                        torch.save(optimizer.state_dict(), os.path.join(best_output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(best_output_dir, "scheduler.pt"))
                        logger.info("Best Model Saving optimizer and scheduler states to %s", best_output_dir)
                    if args.task_name == "mnli":
                        for (j, value) in enumerate(current_eval_list):
                            logger.info("Current Results : {} - {}".format(metric_name_list[j], value))                    
                            logger.info("Best Results : {} - {}".format(metric_name_list[j], best_eval_list[j]))
                    else:
                        logger.info("Current Results : {} - {}".format(metric_name, current_eval))                    
                        logger.info("Best Results : {} - {}".format(metric_name, best_eval))

                # Save model checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.checkpoint_dir, "checkpoint-{}".format(global_step))
                    model_to_save = student.module if hasattr(student, "module") else student
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    if args.task_name == "mnli":
        logger.info("Fianl Best Results : {} - {}".format(metric_name_list, best_eval_list))
        return global_step, loss_log, metric_name_list, best_eval_list
    else:
        logger.info("Fianl Best Results : {} - {}".format(metric_name, best_eval))
        return global_step, loss_log, metric_name, best_eval

def evaluate(args, model, dataset, tokenizer, logger, prefix=""):

    args.eval_batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)

    if args.task_name == "mnli":
        output = []
        datasets = []
        datasets.append(dataset)
        args.task_name = "mnli-mm"
        datasets.append(GlueDataset(args, tokenizer=tokenizer, logger=logger, mode="dev"))
        for i, dataset in enumerate(datasets):
            eval_sampler = SequentialSampler(dataset)
            eval_dataloader = DataLoader(dataset, 
                                sampler=eval_sampler, 
                                batch_size=args.eval_batch_size,
                                collate_fn=default_data_collator)
            if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
                model = torch.nn.DataParallel(model)

            # Eval!
            logger.info("***** Running evaluation {} *****".format(prefix))
            logger.info("  Num examples = %d", len(dataset))
            logger.info("  Batch size = %d", args.eval_batch_size)
            model.eval()
            eval_losses = []
            preds = []
            labels = []
            start_time = timeit.default_timer()
            for inputs in tqdm(eval_dataloader, desc="Evaluating"):
                has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(args.device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    if has_labels:
                        loss, logits = outputs[:2]
                        loss = loss.mean().item()
                        labels.append(inputs.get("labels").detach())
                        preds.append(logits)
                        eval_losses.append(loss)
                    else:
                        loss = None
                        logits = outputs[0]
                        labels=None
            preds = torch.cat(preds, dim=0).cpu().numpy()
            labels = torch.cat(labels, dim=0).cpu().numpy()
            if i == 0:
                args.task_name = "mnli"
            else:
                args.task_name = "mnli-mm"
            metrics  = build_compute_metrics_fn(args.task_name)(EvalPrediction(predictions=preds, label_ids=labels))
            metrics["loss"] = np.mean(eval_losses)
            output.append(PredictionOutput(predictions=preds, label_ids=labels, metrics=metrics).metrics)
        args.task_name ="mnli"

    else:
        eval_sampler = SequentialSampler(dataset)

        eval_dataloader = DataLoader(dataset, 
                            sampler=eval_sampler, 
                            batch_size=args.eval_batch_size,
                            collate_fn=default_data_collator)

        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        model.eval()
        eval_losses = []
        preds = []
        labels = []
        start_time = timeit.default_timer()

        for inputs in tqdm(eval_dataloader, desc="Evaluating"):
            has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(args.device)

            with torch.no_grad():
                outputs = model(**inputs)
                if has_labels:
                    loss, logits = outputs[:2]
                    loss = loss.mean().item()
                    labels.append(inputs.get("labels").detach())
                    preds.append(logits)
                    eval_losses.append(loss)
                else:
                    loss = None
                    logits = outputs[0]
                    labels=None

        preds = torch.cat(preds, dim=0).cpu().numpy()
        labels = torch.cat(labels, dim=0).cpu().numpy()
        metrics  = build_compute_metrics_fn(args.task_name)(EvalPrediction(predictions=preds, label_ids=labels))
        metrics["loss"] = np.mean(eval_losses)
        output = PredictionOutput(predictions=preds, label_ids=labels, metrics=metrics).metrics
    return output

def predict_test(args, model, test_dataset, logger, prefix=""):

    output_mode = glue_output_modes[args.task_name]
    args.eval_batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(test_dataset)
    eval_dataloader = DataLoader(test_dataset,
                        sampler=eval_sampler, 
                        batch_size=args.eval_batch_size,
                        collate_fn=default_data_collator)

    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Make Prediction File {} *****".format(prefix))
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    preds = []
    start_time = timeit.default_timer()

    for inputs in tqdm(eval_dataloader, desc="Prediction"):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(args.device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs[0]

        preds.append(logits)

    preds = torch.cat(preds, dim=0).cpu().numpy()
    prediction = PredictionOutput(predictions=preds, label_ids=None, metrics=None).predictions
    if output_mode =="classification":
        prediction = np.argmax(prediction, axis=1)
    return prediction
