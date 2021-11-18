import argparse

lg = 'fr'
sample_num = 4
sn = '32'
def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lg",
                        default="fr",
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--test_lg",
                        default="zh",
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")


    parser.add_argument("--sn",
                        default='32',
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    parser.add_argument("--data_dir",
                        default="./data/condition_target_ner/raw_data/",
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_name_or_path",
                        default="xlm-roberta-base",
                        type=str,
                        help="the path of the model")
    parser.add_argument("--output_dir",
                        default='./output/',
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--test_entity_path",
                        type=str,
                        default="/home/zhq/Moco/trans/test/test-en-" + lg + "-" + sn + "-entity.txt",
                        help="train-entity")

    parser.add_argument("--train_entity_path",
                        type=str,
                        default="/home/zhq/Moco/trans/train/train-en-" + lg + "-" + sn + "-entity.txt",
                        help="train-entity")
    parser.add_argument("--dev_entity_path",
                        type=str,
                        default="/home/zhq/Moco/trans/dev/dev-en-" + lg + "-" + sn + "-entity.txt",
                        help="dev-entity")
    parser.add_argument("--src_context_path",
                        type=str,
                        default="/home/zhq/Moco/trans/en-" + lg + "-entity-sentences." + sn + ".tsv",
                        help="src-entity-context-pairs")
    parser.add_argument("--trg_context_path",
                        type=str,
                        default="/home/zhq/Moco/trans/" + lg + "-entity-sentences." + sn + ".tsv",
                        help="trg-entity-context-pairs")

    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--gpu_id",
                        default="0",
                        type=str,
                        help="gpu you want to use")
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval or not.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to run predict or not.")
    parser.add_argument("--eval_on",
                        default="dev",
                        help="Whether to run eval on the dev set or test set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--crf_learning_rate",
                        default=1e-4,
                        type=float,
                        help="The initial crf learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=100,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument("--weight_decay", default=1e-4, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    parser.add_argument("--distributed",
                        type=bool,
                        default=True,
                        help="whether distributed or not")
    parser.add_argument('--seed',
                        type=int,
                        default=40,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for eval.")


    parser.add_argument('--z',type=str,default='./result/' + str(sample_num) + lg+ '-'+sn + '/best.pt')
    parser.add_argument('--all_sentence_num',type=int,default=32)    
    parser.add_argument('--dev_all_sentence_num',type=int,default=32,help="only for predict")           
    parser.add_argument('--dev_sample_num',type=int,default=int(sn))
    parser.add_argument('--cut_type',type=str,default='eos-eos')
    parser.add_argument('--wo_span_eos',type=str,default='true')
    parser.add_argument('--is_type',type=int,default=-1,help='token_type')
    parser.add_argument('--train_sample_num',type=int,default=sample_num)
    parser.add_argument('--sentence_max_len',type=int,default=80)
    parser.add_argument('--quene_length',type=int,default=2048)
    parser.add_argument('--momentum',type=float,default=0.999)
    parser.add_argument('--T_para',type=float,default=0.04)
    parser.add_argument('--layer_id',type=int,default=12)
    parser.add_argument('--unsupervised',type=int,default=0)
    parser.add_argument('--simclr',type=int,default=1)
    parser.add_argument('--test_dev',type=int,default=0)
    parser.add_argument('--sample_num',type=int,default=4)
    parser.add_argument('--wolinear',type=int,default=0)
    parser.add_argument('--dev_only_q_encoder',type=int,default=0)
    parser.add_argument('--adapt_to_dataset',type=int,default=0)
    parser.add_argument('--output_loss_dir',type=str,default='./result/' + str(sample_num) + '-' + lg+ '-'+sn)
    parser.add_argument('--output_log_dir',type=str,default='result')

    args = parser.parse_args()
    return args

