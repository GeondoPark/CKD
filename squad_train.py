import os
import timeit
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm, trange
from dataset import load_squad_dataset
from loss import *
from utils import matching_alignment
from tensorboardX import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.data.metrics.squad_metrics import compute_predictions_log_probs, compute_predictions_logits, squad_evaluate
from transformers.data.processors.squad import SquadResult

def train(args, train_dataset, model, tokenizer, logger):
    ##Vanilla Training Code 
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join("./tb_logs/", "squad", args.exp_name))

    args.train_batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    """
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
    """
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
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
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    # Train!
    logger.info("*****Start Training*****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_batch_size)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
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
    best_exact = 0.0
    best_f1 = 0.0

    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), 
                        desc="Epoch", 
                        disable=args.local_rank not in [-1, 0])
    # Added here for reproductibility
    #Why? one more time?
    #set_seed(args)

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            ##When resuming,may be needed
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            outputs = model(input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            token_type_ids=inputs["token_type_ids"],
                            start_positions=inputs["start_positions"],
                            end_positions=inputs["end_positions"])

            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            train_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (train_loss - logging_loss) / args.logging_steps, global_step)
                    logger.info("[TRAIN] Current [Epoch/Iter] : [{}/{}]".format(epoch, step))
                    logger.info("[TRAIN] Current Learning_Rate : LR - {}".format(scheduler.get_lr()[0]))
                    logger.info("[TRAIN] Current Train Loss : LOSS - {}".format((train_loss - logging_loss) / args.logging_steps))
                    logging_loss = train_loss

                if args.local_rank == -1 and args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    results = evaluate(args, model, tokenizer, logger)
                    for key, value in results.items():
                        tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    sum_eval = results["exact"] + results["f1"]
                    if sum_eval > best_eval:
                        best_eval = sum_eval
                        best_f1 = results["f1"]
                        best_exact = results["exact"]
                        best_output_dir = os.path.join(args.checkpoint_dir, "best_checkpoint")
                        model_to_save = (model.module if hasattr(model, "module") else model)
                        model_to_save.save_pretrained(best_output_dir)
                        tokenizer.save_pretrained(best_output_dir)
                        torch.save(args, os.path.join(best_output_dir, "training_args.bin"))
                        torch.save(optimizer.state_dict(), os.path.join(best_output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(best_output_dir, "scheduler.pt"))
                        logger.info("Best Model Saving optimizer and scheduler states to %s", best_output_dir)                        
                    logger.info("[EVAL] Current [Epoch/Iter] : [{}/{}]".format(epoch, step))
                    logger.info("[EVAL] Current Results : Exact - {} F1 - {}".format(results["exact"], results["f1"]))
                    logger.info("[EVAL] Best Results : Exact - {} F1 - {}".format(best_exact, best_f1))

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

    return global_step, train_loss / global_step

def distill_train(args, train_dataset, student, tokenizer, teacher, logger):
    ##Training Code
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join("./tb_logs/", args.exp_name))

    args.train_batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, drop_last=True)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
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

    ##TDOO:Apex model setting
    if args.n_gpu > 1:
        teacher = nn.DataParallel(teacher)
        student = nn.DataParallel(student)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        teacher = torch.nn.parallel.DistributedDataParallel(
            teacher, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        student = torch.nn.parallel.DistributedDataParallel(
            student, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    logger.info("*****Start Training*****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_batch_size)

    global_step = 1
    current_epoch = 0
    steps_trained_in_current_epoch = 0

    ##TODO: Set the notataion
    if args.resume is not None:
        if os.path.exist(args.student_name_or_path):
            try:
                optimizer.load_state_dict(torch.load(os.path.join(args.resume, "optimizer.pt")))
                scheduler.load_state_dict(torch.load(os.path.join(args.resume, "scheduler.pt")))
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = args.resume.split("-")[-1].split("/")[0]
                global_step = int(checkpoint_suffix)
                current_epoch = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
                logger.info(" Continuing training from checkpoint, will skip to saved global_step")
                logger.info(" Continuing training from epoch %d", epochs_trained)
                logger.info("Continuing training from global step %d", global_step)
                logger.info("Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                logger.info("  File doesn`t exist, Start fine-tuning from scratch")

    loss_log = {}
    loss_log["ce"] = 0.0
    for name in args.distil_loss.split('+'):
        loss_log[name] = 0.0

    best_eval = 0.0
    student.zero_grad()
    train_iterator = trange(current_epoch, int(args.num_train_epochs), 
                            desc='Epoch', 
                            disable=args.local_rank not in [-1, 0])
    teacher.eval()

    if "kd" in args.distil_loss.split('+'):
        kd_function = DistillKL(T=args.temperature)

    if "pkd" in args.distil_loss.split('+'):
        ##pkd loss`s input :  batch_size, n_layer, hidden_size
        pkd_function = PKD_loss(p=2, normalize=True)

    if "pkdall" in args.distil_loss.split('+'):
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

    if "wtrdist" in args.distil_loss.split('+'):
        wtrdist_function = WTR_Dist()

    if "wtrangle" in args.distil_loss.split('+'):
        wtrang_function = WTR_Angle()
        if args.n_gpu > 1:
            wtrang_function = nn.DataParallel(wtrang_function)

    if "tiny_hidden" in args.distil_loss.split('+'):
        hidden_function = Hidden_mse(args.s_hidden_size, args.t_hidden_size).to(args.device)

    if "tiny_attention" in args.distil_loss.split('+'):
        att_function = Attention_mse()

    if "tiny_embed" in args.distil_loss.split('+'):
        embed_function = Embedding_mse(args.s_hidden_size, args.t_hidden_size).to(args.device)


    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            student.train()
            if args.exp_type == "distilbert":
                inputs = {"input_ids": batch[0].to(args.device),
                        "attention_mask": batch[1].to(args.device),
                        "start_positions": batch[3].to(args.device),
                        "end_positions": batch[4].to(args.device)}
                        
            else:
                inputs = {"input_ids": batch[0].to(args.device),
                        "attention_mask": batch[1].to(args.device),
                        "token_type_ids": batch[2].to(args.device),
                        "start_positions": batch[3].to(args.device),
                        "end_positions": batch[4].to(args.device),
                        "output_hidden_states":True,
                        "output_attentions":True}

    ##TODO For Structure Distill
    ##Why distillBerT embedding don`t need type token
    ##                "output_hidden_states"=True}

            outputs = student(**inputs)
            loss, student_start_logits, student_end_logits, s_hidden, s_atts = outputs
            if args.exp_type == "distilbert":
                inputs["token_type_ids"] = batch[2].to(args.device)

            with torch.no_grad():
                teacher_start_logits, teacher_end_logits, t_hidden, t_atts = teacher(input_ids=inputs["input_ids"],
                                                token_type_ids=inputs["token_type_ids"],
                                                attention_mask=inputs["attention_mask"],
                                                output_hidden_states=True,
                                                output_attentions=True)

            loss_log["ce"] += loss.mean().item()

            if "kd" in args.distil_loss.split('+'):
                loss_start_kd = kd_function(student_start_logits, teacher_start_logits)
                loss_end_kd = kd_function(student_end_logits, teacher_end_logits)
                loss_kd = (loss_start_kd + loss_end_kd) / 2.0
                loss = args.alpha * loss_kd + (1-args.alpha) * loss
                loss_log["kd"] += loss_kd.item()

            if "pkd" in args.distil_loss.split('+'):
                ##pkd loss`s input :  batch_size, n_layer, hidden_size
                t_patience, s_patience = matching_alignement(t_hidden, s_hidden, 
                                                            matching_strategy=args.matching_strategy,
                                                            only_cls=True)
                pkd = pkd_function(t_patience, s_patience)
                loss = loss + args.beta * pkd
                loss_log["pkd"] += pkd.item()

            if "pkdall" in args.distil_loss.split('+'):
                ##pkd loss`s input :  batch_size, n_layer, hidden_size
                t_patience, s_patience = matching_alignement(t_hidden, s_hidden, 
                                                            matching_strategy=args.matching_strategy,
                                                            only_cls=False)
                pkd = pkd_function(t_patience, s_patience)
                loss = loss + args.beta * pkd
                loss_log["pkdall"] += pkd.item()

            if "tiny_hidden" in args.distil_loss.split('+'):
                t_embed, s_embed = matching_alignement(t_hidden, s_hidden,
                                                        matching_strategy=args.matching_strategy,
                                                        only_cls=False)
                hidden_loss = hidden_mse(s_embed, t_embed)
                loss = loss + args.hidden_w * hidden_loss
                loss_log["hidden_mse"] += hidden_loss.item()
                
            if "tiny_embed" in args.distil_loss.split('+'):
                t_embed, s_embed = matching_alignement(t_hidden, s_hidden,
                                                        matching_strategy=args.matching_strategy,
                                                        only_cls=False)
                hidden_loss = hidden_mse(s_embed, t_embed)
                loss = loss + args.embed_w * hidden_loss
                loss_log["embed_mse"] += hidden_loss.item()

            if "tiny_attention" in args.distil_loss.split('+'):
                t_embed, s_embed = matching_alignement(t_hidden, s_hidden,
                                                        matching_strategy=args.matching_strategy,
                                                        only_cls=False)
                hidden_loss = hidden_mse(s_embed, t_embed)
                loss = loss + args.att_w * hidden_loss
                loss_log["attention_mse"] += hidden_loss.item()

            if "wrdist" in args.distil_loss.split('+'):
                if args.student_path == "./download_models/mobile-bert":
                    s_embed, t_embed = matching_alignement(s_hidden, t_hidden, 
                                                            matching_strategy=args.matching_strategy,
                                                            only_cls=False)
                    wrdist_loss = wrdist_function(t_embed, s_embed, inputs["attention_mask"], 
                                                args.lossfunc, 
                                                normalize=args.distnorm)
                    loss = loss + args.wrdist_w * wrdist_loss
                    loss_log["wrdist"] += wrdist_loss.item()
                else:
                    t_embed, s_embed = matching_alignement(t_hidden, s_hidden, 
                                                            matching_strategy=args.matching_strategy,
                                                            only_cls=False)
                    wrdist_loss = wrdist_function(t_embed, s_embed, inputs["attention_mask"], 
                                                args.lossfunc, 
                                                normalize=args.distnorm)
                    loss = loss + args.wrdist_w * wrdist_loss
                    loss_log["wrdist"] += wrdist_loss.item()

            if "wrdist_window" in args.distil_loss.split('+'):
                t_embed, s_embed = matching_alignement(t_hidden, s_hidden, 
                                                        matching_strategy=args.matching_strategy,
                                                        only_cls=False)
                wrdist_loss = wrdist_function(t_embed, s_embed, inputs["attention_mask"], 
                                            args.lossfunc, 
                                            normalize=args.distnorm,
                                            window=args.window_size)
                loss = loss + args.wrdist_w * wrdist_loss
                loss_log["wrdist_window"] += wrdist_loss.item()

            if "wrangle" in args.distil_loss.split('+'):
                t_embed, s_embed = matching_alignement(t_hidden, s_hidden, 
                                                        matching_strategy=args.matching_strategy,
                                                        only_cls=False)
                wrang_loss = wrang_function(t_embed, s_embed, 
                                        inputs["attention_mask"], 
                                        args.lossfunc)
                loss = loss + args.wrangle_w * wrang_loss
                loss_log["wrangle"] += wrang_loss.mean().item()

            if "wrangle_window" in args.distil_loss.split('+'):
                if args.student_path == "./download_models/mobile-bert":
                    s_embed, t_embed = matching_alignement(s_hidden, t_hidden, 
                                                            matching_strategy=args.matching_strategy,
                                                            only_cls=False)
                    wrang_loss = wrang_function(t_embed, s_embed, inputs["attention_mask"], 
                                            args.lossfunc,
                                            window=args.window_size)
                    loss = loss + args.wrangle_w * wrang_loss
                    loss_log["wrangle_window"] += wrang_loss.mean().item()
                else:
                    t_embed, s_embed = matching_alignement(t_hidden, s_hidden, 
                                                            matching_strategy=args.matching_strategy,
                                                            only_cls=False)
                    wrang_loss = wrang_function(t_embed, s_embed, inputs["attention_mask"], 
                                            args.lossfunc,
                                            window=args.window_size)
                    loss = loss + args.wrangle_w * wrang_loss
                    loss_log["wrangle_window"] += wrang_loss.mean().item()

            if "wtrdist" in args.distil_loss.split('+'):
                if args.student_path == "./download_models/mobile-bert":
                    s_embed, t_embed = matching_alignement(s_hidden, t_hidden, 
                                                            matching_strategy=args.matching_strategy,
                                                            only_cls=False)
                    wtrdist_loss = wtrdist_function(t_embed, s_embed, inputs["attention_mask"],
                                                args.lossfunc,
                                                normalize=args.distnorm)
                    loss = loss + args.wtrdist_w * wtrdist_loss
                    loss_log["wtrdist"] += wtrdist_loss.item()

                else:
                    t_embed, s_embed = matching_alignement(t_hidden, s_hidden, 
                                                            matching_strategy=args.matching_strategy,
                                                            only_cls=False)
                    wtrdist_loss = wtrdist_function(t_embed, s_embed, inputs["attention_mask"],
                                                args.lossfunc,
                                                normalize=args.distnorm)
                    loss = loss + args.wtrdist_w * wtrdist_loss
                    loss_log["wtrdist"] += wtrdist_loss.item()

            if "wtrangle" in args.distil_loss.split('+'):
                if args.student_path == "./download_models/mobile-bert":

                    s_embed, t_embed = matching_alignement(s_hidden, t_hidden, 
                                                            matching_strategy=args.matching_strategy,
                                                            only_cls=False)
                    wtrang_loss = wtrang_function(t_embed, s_embed, inputs["attention_mask"],
                                                args.lossfunc)
                    loss = loss + args.wtrangle_w * wtrang_loss
                    loss_log["wtrangle"] += wtrang_loss.mean().item()

                else:
                    t_embed, s_embed = matching_alignement(t_hidden, s_hidden, 
                                                            matching_strategy=args.matching_strategy,
                                                            only_cls=False)
                    wtrang_loss = wtrang_function(t_embed, s_embed, inputs["attention_mask"],
                                                args.lossfunc)            
                    loss = loss + args.wtrangle_w * wtrang_loss
                    loss_log["wtrangle"] += wtrang_loss.mean().item()

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward() ##4251

            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                len(epoch_iterator)<=args.gradient_accumulation_steps 
                and (step+1)==len(epoch_iterator)):
                
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                student.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logger.info("[TRAIN] Current [Epoch/Iter] : [{}/{}|{}] | LR - {}"
                                                .format(epoch, step, len(epoch_iterator),scheduler.get_lr()[0]))
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    train_text = "[TRAIN] Train Loss : "
                    for name, value in loss_log.items():
                        tb_writer.add_scalar("loss_{}".format(name), (value / args.logging_steps), global_step)
                        train_text += "{} : {:.4f} |".format(name.upper(), (value / args.logging_steps))
                        loss_log[name] = 0.0
                    logger.info(train_text)

                if args.local_rank == -1 and args.eval_steps > 0 and ((global_step % args.eval_steps == 0) or (global_step == t_total)):
                    results = evaluate(args, student, tokenizer, logger)
                    for key, value in results.items():
                        tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    sum_eval = results["exact"] + results["f1"]
                    if sum_eval > best_eval:
                        best_eval = sum_eval
                        best_f1 = results["f1"]
                        best_exact = results["exact"]
                        best_output_dir = os.path.join(args.checkpoint_dir, "best_checkpoint")
                        model_to_save = (student.module if hasattr(student, "module") else student)
                        model_to_save.save_pretrained(best_output_dir)
                        tokenizer.save_pretrained(best_output_dir)
                        torch.save(args, os.path.join(best_output_dir, "training_args.bin"))
                        torch.save(optimizer.state_dict(), os.path.join(best_output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(best_output_dir, "scheduler.pt"))
                        logger.info("Best Model Saving optimizer and scheduler states to %s", best_output_dir)                        

                    logger.info("[EVAL] Current [Epoch/Iter][{}/{}|{}] | Exact : {} | F1 - {}"
                                    .format(epoch, step, len(epoch_iterator), results["exact"], results["f1"]))
                    logger.info("[EVAL] Best Results : Exact - {} F1 - {}".format(best_exact, best_f1))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.checkpoint_dir, "checkpoint-{}".format(global_step))
                    model_to_save = (student.module if hasattr(student, "module") else student)
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

    if args.local_rank in [-1, 0]:
        tb_writer.close()
    return global_step, loss_log, best_exact, best_f1 

def evaluate(args, model, tokenizer, logger, prefix=""):
    dataset, examples, features = load_squad_dataset(args, tokenizer, logger, evaluate=True, output_examples=True)
    
    args.eval_batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    #logger.info("***** Running evaluation {} *****".format(prefix))
    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], 
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2]}

            example_indices = batch[3]
            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            output = [output[i].detach().cpu().tolist() for output in outputs]
            
            ##TODO: Extend to other netowrk (XLNET~)
            start_logits, end_logits = output
            result = SquadResult(unique_id, start_logits, end_logits)
            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    #logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))
    output_prediction_file = os.path.join(args.result_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.result_dir, "nbest_predictions_{}.json".format(prefix))

    if args.version_2:
        output_null_log_odds_file = os.path.join(args.result_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    ##TODO: Extend to other network
    predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            False,                          #verbose_logging
            args.version_2,
            args.null_score_diff_threshold,
            tokenizer,
        )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    return results
