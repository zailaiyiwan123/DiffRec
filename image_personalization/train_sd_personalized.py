import argparse

from .sd_lora_trainer import SDPersonalizationConfig, SDPersonalizationTrainer
from .dataloader import build_batches


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl_gz", type=str, required=True)
    p.add_argument("--model_root", type=str, default="small-stable-diffusion-v0")
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_target_modules", type=str, default="to_q,to_k,to_v")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--adaptive_weight", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = SDPersonalizationConfig(
        model_root=args.model_root,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        learning_rate=args.lr,
    )
    trainer = SDPersonalizationTrainer(cfg)

    for _ in range(args.epochs):
        for batch in build_batches(args.train_jsonl_gz, batch_size=args.batch_size, adaptive_weight=args.adaptive_weight):
            # 将 batch 的样本逐个训练（SD 文生图成本较高，小batch/逐条适合CPU/GPU调试）
            for sample in batch:
                trainer.training_step(sample)


if __name__ == "__main__":
    main()


