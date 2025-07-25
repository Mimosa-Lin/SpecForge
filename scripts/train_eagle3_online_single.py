import argparse
import hashlib
import os

import torch
import torch.distributed as dist
import wandb
from accelerate.utils import set_seed
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from specforge import (
    AutoDistributedTargetModel,
    AutoDraftModelConfig,
    AutoEagle3DraftModel,
    OnlineEagle3Model,
)
from specforge.data import (
    build_eagle3_dataset,
    generate_vocab_mapping_file,
    prepare_dp_dataloaders,
)
from specforge.distributed import destroy_distributed, get_dp_group, init_distributed
from specforge.lr_scheduler import CosineAnnealingWarmupLR
from specforge.utils import get_last_checkpoint, print_with_rank, rank_0_priority


def parse_args():
    parser = argparse.ArgumentParser(description="Train Eagle3 with online data")

    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--draft-model-config", type=str, required=True)
    parser.add_argument("--embedding-key", type=str, default="model.embed_tokens.weight")

    parser.add_argument("--train-data-path", type=str, required=True)
    parser.add_argument("--eval-data-path", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--warmup-ratio", type=float, default=0.02)

    parser.add_argument("--chat-template", type=str, default="llama3")
    parser.add_argument("--tp-size", type=int, default=1)

    parser.add_argument("--cache-key", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dist-timeout", type=int, default=20)
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-key", type=str, default=None)

    return parser.parse_args()


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1


def all_reduce(tensor):
    if is_distributed():
        dist.all_reduce(tensor)
    return tensor


def init_wandb(args):
    wandb.login(key=args.wandb_key)
    wandb.init(project=args.wandb_project, name=args.wandb_name)


def wandb_log_if_initialized(log_dict):
    if get_rank() == 0 and wandb.run is not None:
        wandb.log(log_dict)


def print_on_rank0(message):
    if get_rank() == 0:
        print(message)


def main():
    args = parse_args()
    set_seed(args.seed)

    multi_gpu = torch.cuda.device_count() > 1

    if multi_gpu:
        init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
        print_with_rank("Initialized distributed environment")
    else:
        print("Running in single-GPU mode")

    if args.wandb and get_rank() == 0:
        init_wandb(args)

    draft_model_last_checkpoint = None
    if args.resume and os.path.isdir(args.output_dir):
        print_on_rank0(args.output_dir)
        draft_model_last_checkpoint = get_last_checkpoint(args.output_dir)
        print_on_rank0(f"Last checkpoint detected: {draft_model_last_checkpoint}")

    if args.tp_size > 1:
        target_model = AutoDistributedTargetModel.from_pretrained(
            pretrained_model_name_or_path=args.target_model_path,
            torch_dtype=torch.bfloat16,
            device="cuda",
        ).eval()
    else:
        target_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.target_model_path,
            torch_dtype=torch.bfloat16,
        ).eval().cuda()

    print_with_rank("Initialized target model")

    draft_model_config = AutoDraftModelConfig.from_file(args.draft_model_config)
    if draft_model_last_checkpoint:
        draft_model = AutoEagle3DraftModel.from_pretrained(draft_model_last_checkpoint).cuda().to(torch.bfloat16)
    else:
        draft_model = AutoEagle3DraftModel.from_config(draft_model_config).cuda().to(torch.bfloat16)

    draft_model.load_embedding(args.target_model_path, embedding_key=args.embedding_key)
    draft_model.freeze_embedding()

    print_with_rank("Initialized draft model")

    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    cache_key = hashlib.md5(args.train_data_path.encode()).hexdigest()
    train_dataset = load_dataset("json", data_files=args.train_data_path)["train"]

    with rank_0_priority():
        train_eagle3_dataset = build_eagle3_dataset(
            dataset=train_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
            cache_key=cache_key,
        )
        vocab_mapping_path = generate_vocab_mapping_file(
            dataset=train_eagle3_dataset,
            target_vocab_size=draft_model_config.vocab_size,
            draft_vocab_size=draft_model_config.draft_vocab_size,
            cache_dir=os.path.join(args.cache_dir, "vocab_mapping"),
            cache_key=cache_key,
        )

    train_dataloader = prepare_dp_dataloaders(
        train_eagle3_dataset,
        args.batch_size,
        num_workers=4,
        shuffle=True,
        process_group=get_dp_group() if multi_gpu else None,
    )
    print_with_rank("Initialized train dataloader")

    draft_model.load_vocab_mapping(vocab_mapping_path)
    print_with_rank("Loaded vocab mapping")

    if args.eval_data_path is not None:
        eval_dataset = load_dataset("json", data_files=args.eval_data_path)["train"]
        eval_eagle3_dataset = build_eagle3_dataset(
            eval_dataset,
            tokenizer,
            args.chat_template,
            args.max_length,
        )
        eval_dataloader = prepare_dp_dataloaders(
            eval_eagle3_dataset,
            args.batch_size,
            num_workers=4,
            shuffle=False,
            process_group=get_dp_group() if multi_gpu else None,
        )
        print_with_rank("Initialized eval dataloader")

    eagle3_model = OnlineEagle3Model(
        target_model=target_model,
        draft_model=draft_model,
    )

    if multi_gpu:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

        eagle3_model = FSDP(
            eagle3_model,
            use_orig_params=True,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            ignored_modules=[target_model],
            process_group=get_dp_group(),
        )

    print_with_rank("Initialized Eagle3 model")

    optimizer = torch.optim.AdamW(eagle3_model.parameters(), lr=args.learning_rate)
    total_steps = args.num_epochs * len(train_dataloader)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = CosineAnnealingWarmupLR(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)

    print_with_rank("Initialized optimizer and scheduler")

    start_epoch = 0
    if draft_model_last_checkpoint is not None:
        print_on_rank0(f"Resuming from checkpoint: {draft_model_last_checkpoint}")
        state_path = os.path.join(draft_model_last_checkpoint, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
            start_epoch = state["epoch"] + 1

    if is_distributed():
        dist.barrier()

    print_on_rank0(f"Starting training from epoch {start_epoch}")
    for epoch in range(start_epoch, args.num_epochs):
        train_dataloader.sampler.set_epoch(epoch + 1) if hasattr(train_dataloader.sampler, "set_epoch") else None
        draft_model.train()
        epoch_acces = [[] for _ in range(eagle3_model.module.length if multi_gpu else eagle3_model.length)]
        epoch_plosses = [[] for _ in range(len(epoch_acces))]

        for data in tqdm(train_dataloader, desc=f"Training Epoch {epoch}"):
            optimizer.zero_grad()
            plosses, _, acces = eagle3_model(
                input_ids=data["input_ids"].cuda(),
                attention_mask=data["attention_mask"].cuda(),
                loss_mask=data["loss_mask"].cuda(),
            )

            ploss_weight = [0.8**i for i in range(len(plosses))]
            ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
            ploss.backward()
            optimizer.step()
            scheduler.step()

            logdict = {"train/lr": optimizer.param_groups[0]["lr"]}
            for i in range(len(plosses)):
                logdict[f"train/ploss_{i}"] = plosses[i].item()
            for i in range(len(acces)):
                logdict[f"train/acc_{i}"] = acces[i]
            wandb_log_if_initialized(logdict)

            for i in range(len(acces)):
                epoch_acces[i].append(acces[i])
                epoch_plosses[i].append(plosses[i].item())

        for i in range(len(epoch_acces)):
            acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
            acc_i = all_reduce(acc_i) / get_world_size()
            wandb_log_if_initialized({f"train/epochacc_{i}": acc_i.item()})
            print_on_rank0(f"Train Epoch [{epoch + 1}/{args.num_epochs}], pos {i}, Acc: {acc_i.item():.2f}")

        for i in range(len(epoch_plosses)):
            loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
            loss_i = all_reduce(loss_i) / get_world_size()
            wandb_log_if_initialized({f"train/epochploss_{i}": loss_i.item()})
            print_on_rank0(f"Train Epoch [{epoch + 1}/{args.num_epochs}], pos {i}, pLoss: {loss_i.item():.2f}")

        if epoch % args.save_interval == 0:
            epoch_output_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
            if get_rank() == 0:
                os.makedirs(epoch_output_dir, exist_ok=True)
            if is_distributed():
                dist.barrier()

            state_to_save = {
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }

            if multi_gpu:
                from torch.distributed.fsdp import StateDictType
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

                with FSDP.state_dict_type(eagle3_model, StateDictType.FULL_STATE_DICT):
                    model_state_dict = eagle3_model.state_dict()
            else:
                model_state_dict = eagle3_model.state_dict()

            draft_model_state_dict = {
                k.replace("draft_model.", ""): v
                for k, v in model_state_dict.items()
                if "draft_model." in k
            }

            if get_rank() == 0:
                torch.save(state_to_save, os.path.join(epoch_output_dir, "training_state.pt"))
                draft_model.save_pretrained(epoch_output_dir, state_dict=draft_model_state_dict)

    if is_distributed():
        destroy_distributed()


if __name__ == "__main__":
    main()
