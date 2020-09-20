import os
import random
import time

import fire
import numpy as np
import sentencepiece as spm
import torch
import tqdm
import wandb
from loguru import logger
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from representjs import RUN_DIR, CSNJS_DIR
from representjs.data.precomputed_dataset import PrecomputedDataset
from representjs.models.code_mlm import CodeMLM, CodeContrastiveMLM
from representjs.models.code_moco import CodeMoCo
from representjs.utils import accuracy, count_parameters, get_linear_schedule_with_warmup

DEFAULT_CSNJS_TRAIN_FILEPATH = str(CSNJS_DIR / "javascript_dedupe_definitions_nonoverlap_v2_train.jsonl.gz")
DEFAULT_SPM_UNIGRAM_FILEPATH = str(CSNJS_DIR / "csnjs_8k_9995p_unigram_url.model")


def training_step(model, batch, use_cuda=False):
    imgs, lengths, _ = batch
    if use_cuda:
        imgs = imgs.cuda(non_blocking=True)
    imgs_k, imgs_q = imgs[:, 0, :], imgs[:, 1, :]
    lengths_k, lengths_q = lengths[:, 0], lengths[:, 1]
    output, target = model(imgs_q, imgs_k, lengths_k, lengths_q)
    loss = F.cross_entropy(output, target)
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    logs = {
        "pretrain/loss": loss.item(),
        "pretrain/acc@1": acc1[0].item(),
        "pretrain/acc@5": acc5[0].item(),
        "pretrain/queue_ptr": model.module.queue_ptr.item(),
    }
    return {"loss": loss, "log": logs}


def mask_mlm(seq, pad_id, mask_id, vocab_start_range, vocab_end_range):
    # The training data generator chooses 15% of the token positions at random for prediction.
    # If the i-th token is chosen, we replace the i-th token with
    # (0) not masked
    # (1) the [MASK] token 80% of the time (0.12)
    # (2) a random token 10% of the time (0.015)
    # (3) the unchanged i-th token 10% of the time (0.015)
    #
    # https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/dataset/dataset.py#L63
    rand_replacements = torch.zeros_like(seq, dtype=torch.long).random_(vocab_start_range, vocab_end_range)

    masked_tokens = (torch.rand_like(seq, dtype=torch.float) < 0.15) & (seq != pad_id)
    mask_type_prob = torch.rand_like(seq, dtype=torch.float)
    mask_token_prob = (mask_type_prob < 0.8) & masked_tokens
    random_token_prob = (mask_type_prob < 0.9) & (mask_type_prob >= 0.8) & masked_tokens
    identity_token_prob = (mask_type_prob >= 0.9) & masked_tokens
    assert torch.sum(masked_tokens) == torch.sum(mask_token_prob | random_token_prob | identity_token_prob)

    targets = torch.zeros_like(seq).fill_(pad_id)
    targets[masked_tokens] = seq[masked_tokens]

    seq[mask_token_prob] = mask_id
    seq[random_token_prob] = rand_replacements[random_token_prob]
    return seq, targets


def training_step_mlm(sp, model, batch, mask_id: int, pad_id: int, vocab_start_idx: int, vocab_end_idx: int,
                      use_cuda=True):
    seq, lengths, _ = batch  # B x L
    if use_cuda:
        seq = seq.cuda()
    B, L = seq.shape
    seq_masked, targets = mask_mlm(seq, pad_id, mask_id, vocab_start_idx, vocab_end_idx)
    # logger.debug(f"Example transform:\t{sp.DecodeIds(seq_masked[0].cpu().numpy().tolist())}")
    output = model(seq_masked, lengths)  # B x L x Vocab
    assert targets.shape == (B, L), f"{targets.shape} versus {B}x{L}"
    assert output.shape == (B, L, output.shape[-1]), output.shape
    mask = targets.ne(0)

    logits = output[mask]
    tgts = targets[mask]
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        tgts.view(-1),
        # reduction='sum',
        ignore_index=pad_id,
    )
    loss = F.cross_entropy(output.flatten(end_dim=1), targets.flatten(), ignore_index=pad_id)
    # print('loss: ', loss)
    acc1, acc5 = accuracy(output[targets != pad_id], targets[targets != pad_id], topk=(1, 5))
    return {
        "loss": loss,
        "log": {"pretrain/loss": loss.item(), "pretrain/acc@1": acc1[0].item(), "pretrain/acc@5": acc5[0].item()},
    }


def training_step_hybrid(sp, model, batch, mask_id, pad_id, vocab_start_idx, vocab_end_idx, use_cuda):
    imgs, _lengths, _ = batch
    # TODO: implement LSTM for hybrid model and pass lengths to model call
    imgs_k, imgs_q = imgs[:, 0, :], imgs[:, 1, :]
    imgs_q, mlm_targets = mask_mlm(imgs_q, pad_id, mask_id, vocab_start_idx, vocab_end_idx)
    if use_cuda:
        imgs_k = imgs_k.cuda(non_blocking=True)
        imgs_q = imgs_q.cuda(non_blocking=True)
        mlm_targets = mlm_targets.cuda(non_blocking=True)
    predicted_masked_tokens, moco_logits, moco_targets = model(imgs_k, imgs_q)
    moco_loss = F.cross_entropy(moco_logits, moco_targets)
    moco_acc1, moco_acc5 = accuracy(moco_logits, moco_targets, topk=(1, 5))
    mlm_loss = F.cross_entropy(predicted_masked_tokens.flatten(end_dim=1), mlm_targets.flatten(), ignore_index=pad_id)
    mlm_acc1, mlm_acc5 = accuracy(predicted_masked_tokens[mlm_targets != pad_id], mlm_targets[mlm_targets != pad_id],
                                  topk=(1, 5))
    loss = 4 * moco_loss + mlm_loss
    logs = {
        "pretrain/moco/loss": moco_loss.item(),
        "pretrain/moco/acc@1": moco_acc1[0].item(),
        "pretrain/moco/acc@5": moco_acc5[0].item(),
        "pretrain/moco/queue_ptr": model.module.queue_ptr.item(),
        "pretrain/mlm/loss": mlm_loss.item(),
        "pretrain/mlm/acc@1": mlm_acc1[0].item(),
        "pretrain/mlm/acc@5": mlm_acc5[0].item(),
        "pretrain/hybrid_loss": loss,
    }
    return {"loss": loss, "log": logs}

def pretrain(
    run_name: str,
    # Data
    train_filepath: str = DEFAULT_CSNJS_TRAIN_FILEPATH,
    spm_filepath: str = DEFAULT_SPM_UNIGRAM_FILEPATH,
    num_workers=1,
    limit_dataset_size=-1,
    max_length=1024,
    max_sequence_length=1024,
    augment_window_crop_size=6,
    subword_regularization_alpha: float = 0,
    program_mode="contrastive",
    loss_mode="infonce",
    min_alternatives=1,
    # Model
    encoder_type: str = "transformer",
    lstm_project_mode: str = "hidden",
    n_encoder_layers: int = 6,
    d_model: int = 512,
    # Optimization
    num_epochs: int = 100,
    save_every: int = 1,
    batch_size: int = 256,
    lr: float = 8e-4,
    weight_decay: float = 0,
    adam_betas=(0.9, 0.98),
    warmup_steps: int = 5000,
    num_steps: int = 600000,
    # Computational
    use_cuda: bool = True,
    seed: int = 1,
):
    run_name = str(run_name)  # support numerical run ids
    slurm_job_id, slurm_job_hostname = (
        os.environ.get("SLURM_JOB_ID"),
        os.environ.get("SLURM_JOB_NODELIST"),
    )
    config = locals()
    logger.info("Training configuration: {}".format(config))
    logger.info(
        "CUDA_VISIBLE_DEVICES = '{}', CUDA_DEVICE_ORDER = '{}'".format(
            os.environ.get("CUDA_VISIBLE_DEVICES"), os.environ.get("CUDA_DEVICE_ORDER")
        )
    )
    print("torch.cuda.is_available(): ", torch.cuda.is_available())
    assert not use_cuda or torch.cuda.is_available(), "CUDA not available. Check env configuration, or pass --use_cuda False"
    assert loss_mode in ["infonce", "mlm", 'hybrid']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    run_dir = RUN_DIR / "{}_{}".format(run_name, int(time.time()))
    run_dir.mkdir(exist_ok=True, parents=True)
    logger.add(str((run_dir / "train.log").resolve()))
    logger.info(f"Saving logs, model checkpoints to {run_dir}")
    # wandb.init(
    #     name=run_name, config=config, job_type="training", project="moco-pretrain", entity="ml4code",
    # )

    sp = spm.SentencePieceProcessor()
    sp.Load(spm_filepath)
    pad_id = sp.PieceToId("[PAD]")
    mask_id = sp.PieceToId("[MASK]")

    # Create training dataset and dataloader
    # assert train_filepath.endswith(".pickle")
    assert train_filepath.endswith(".pickle") or train_filepath.endswith(".gz")

    def pad_collate(batch):
        B = len(batch)
        if config["program_mode"] == "contrastive":
            X1, X2 = zip(*batch)
            X = X1 + X2
        else:
            X = batch

        # Create tensor of sequence lengths, [B] or [2B]
        lengths = torch.tensor([len(x) for x in X], dtype=torch.long)

        # Create padded tensor for batch, [B, T] or [2B, T]
        X = pad_sequence(X, batch_first=True, padding_value=pad_id)

        if config["program_mode"] == "contrastive":
            # Reshape X to [B, 2, T]
            T = X.size(-1)
            X = torch.reshape(X, (2, B, -1))
            X = torch.transpose(X, 0, 1)
            assert X.shape == (B, 2, T)
            lengths = torch.reshape(lengths, (2, B)).transpose(0, 1)
            assert lengths.shape == (B, 2)
        return X, lengths, None

    train_dataset = PrecomputedDataset(
        config["train_filepath"],
        min_alternatives=config["min_alternatives"],
        program_mode=config["program_mode"],
        limit_size=config["limit_dataset_size"],
        sp=sp,
        subword_regularization_alpha=config["subword_regularization_alpha"],
        max_length=config["max_length"]
    )
    data = train_dataset.__getitem__(0)
    # print('data[0]: ', data)
    # exit()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate, num_workers=num_workers,
        drop_last=True,
    ) # shuffle=True

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=batch_size, shuffle=False
    # ) # shuffle=True
    # pbar = tqdm.tqdm(train_loader)
    # for batch in pbar:
    #     print('batch: ', batch)
    # Create model
    if config["loss_mode"] == "infonce":
        model = CodeMoCo(sp.GetPieceSize(), pad_id=pad_id, d_model=config["d_model"], encoder_config=dict(
            encoder_type=config["encoder_type"],
            lstm_project_mode=config["lstm_project_mode"],
            n_encoder_layers=config["n_encoder_layers"]
        ))
        logger.info(f"Created CodeMoCo model with {count_parameters(model)} params")
    elif config["loss_mode"] == "mlm":
        model = CodeMLM(sp.GetPieceSize(), pad_id=pad_id, encoder_type=config["encoder_type"],
                        n_encoder_layers=config["n_encoder_layers"])
        logger.info(f"Created CodeMLM model with {count_parameters(model)} params")
    elif config["loss_mode"] == "hybrid":
        model = CodeContrastiveMLM(sp.GetPieceSize(), pad_id=pad_id)
        logger.info(f"Created CodeContrastiveMLM model with {count_parameters(model)} params")
    else:
        raise ValueError(f"Bad loss mode {config['loss_mode']}")

    # model = nn.DataParallel(model)
    model = model.cuda() if use_cuda else model
    params = model.parameters()
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], betas=config["adam_betas"], eps=1e-6,
                                 weight_decay=config["weight_decay"])
    sched = get_linear_schedule_with_warmup(optimizer, config["warmup_steps"], config["num_steps"])

    global_step = 0
    min_eval_loss = float("inf")
    for epoch in tqdm.trange(1, num_epochs + 1, desc="training", unit="epoch", leave=False):
        logger.info(f"Starting epoch {epoch}\n")
        model.train()
        pbar = tqdm.tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            optimizer.zero_grad()
            if config["loss_mode"] == "infonce":
                train_metrics = training_step(model, batch, use_cuda=config["use_cuda"])
            elif config["loss_mode"] == "mlm":
                # replace tokens randomly with tokens from _ (8)
                train_metrics = training_step_mlm(
                    sp, model, batch, pad_id=pad_id, mask_id=mask_id, vocab_start_idx=8, vocab_end_idx=7999,
                    use_cuda=config["use_cuda"]
                )
            elif config["loss_mode"] == "hybrid":
                train_metrics = training_step_hybrid(
                    sp, model, batch, mask_id=mask_id, pad_id=pad_id, vocab_start_idx=0, vocab_end_idx=7999,
                    use_cuda=config["use_cuda"]
                )
            else:
                raise ValueError("Bad loss type")

            loss = train_metrics["loss"]
            loss.backward()
            optimizer.step()
            sched.step()

            # Log loss
            global_step += 1
            # wandb.log(dict(epoch=epoch, **train_metrics["log"]), step=global_step)
            pbar.set_description(f"epoch {epoch} loss {loss.item():.4f}")

            # Save checkpoint
            if save_every and global_step % save_every == 0:
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "config": config,
                }
                model_file = run_dir / f"ckpt_pretrain_ep{epoch:04d}_step{global_step:07d}.pth"
                logger.info(f"Saving checkpoint to {model_file}...")
                torch.save(checkpoint, str(model_file.resolve()))
                # wandb.save(model_file)
                logger.info("Done.")


if __name__ == "__main__":
    fire.Fire(pretrain)
