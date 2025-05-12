import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Train different prompt tuning methods.")
    parser.add_argument("--method", type=str, required=True,
                        choices=["coop", "cocoop", "maple", "promptsrc", "vanilla"],
                        help="Which training method to use.")
    
    args = parser.parse_args()

    if args.method == "coop":
        from trainers.coop_prompt_tuning_train import main as train_fn
    elif args.method == "cocoop":
        from cocoop_prompt_learned_fast import main as train_fn
    elif args.method == "maple":
        from trainers.maple_train import main as train_fn
    elif args.method == "promptsrc":
        from trainers.promptsrc_train import main as train_fn
    elif args.method == "vanilla":
        from trainers.prompt_tuning_train import main as train_fn
    else:
        print(f"Unknown method: {args.method}")
        sys.exit(1)

    train_fn()

if __name__ == "__main__":
    main()
