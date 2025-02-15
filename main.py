import argparse
import neptune

from trainer import Trainer
from utils import init_logger, load_tokenizer, set_seed, MODEL_CLASSES, MODEL_PATH_MAP
from preprocess import load_and_cache_examples


def set_argument():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="./data", type=str, help="Training dataset directory")
    parser.add_argument("--task", default="baemin", type=str, help="Task to handle")
    parser.add_argument("--mode", default="train", type=str, help="Execution type")
    parser.add_argument("--train_filepath", default="abuse_train.csv", type=str, help="File Path")  # Use `abuse_train.csv` which contains the `shop_no` with abuse patterns
    parser.add_argument("--dev_filepath", default="abuse_dev.csv", type=str, help="File Path")
    parser.add_argument("--test_filepath", default="test.csv", type=str, help="File Path")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to save, load model")
    
    parser.add_argument("--model_type", default="kobert", type=str, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default="monologg/kobert", type=str, help="Model Name")
    parser.add_argument('--seed', default=42, type=int, help="random seed for initialization")

    parser.add_argument("--shop_no_size", default=0, type=int, help="Number of unique shop_no")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")
    parser.add_argument("--max_seq_len", default=50, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=20.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=500, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--logger", action="store_true", help="Activate logger")
    args = parser.parse_args()
    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    return args


def main(args):
    init_logger()
    set_seed(args)

    if args.logger:
        neptune.init("wjdghks950/sandbox")
        neptune.create_experiment(name="Fake Detection Model for AI Championship")
        neptune.append_tag("BertForSequenceClassification", "finetuning", "fake detection")

    tokenizer = load_tokenizer(args)
    train_dataset, vocab_size = load_and_cache_examples(args, tokenizer, mode="train")
    dev_dataset, _ = load_and_cache_examples(args, tokenizer, mode="dev")
    trainer = Trainer(args, train_dataset, dev_dataset, vocab_size=vocab_size)

    if args.do_train:
        trainer.train()

    if args.logger:
        neptune.stop()


if __name__ == "__main__":
    args = set_argument()
    main(args)
