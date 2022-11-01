import argparse

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lg", default="all", type=str,
                        help="The language where model trained.")
    parser.add_argument("--test_lg", default="zh", type=str,
                        help="The language where model tests")
    parser.add_argument("--model_name_or_path",default="xlm-roberta-base",type=str,
                        help="the path of the model")
    parser.add_argument("--test_phrase_path",type=str, default='',
                        help="test-phrase is changed in predict.py")
    parser.add_argument("--train_phrase_path",type=str,default='',
                        help="train-phrase is changed in predict.py")
    parser.add_argument("--dev_phrase_path", type=str,default='',
                        help="dev-phrase is changed in predict.py")
    parser.add_argument("--src_context_path",type=str,default='',
                        help="src-phrase-context-pairs is changed in predict.py")
    parser.add_argument("--trg_context_path",type=str,default='',
                        help="trg-phrase-context-pairs is changed in predict.py")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--pad_to_max_length", action="store_true",
                        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--distributed", type=bool, default=False,
                        help="whether distributed or not")
    parser.add_argument('--seed', type=int, default=40,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--eval_batch_size", default=16, type=int, help="Total batch size for eval.")

    
    parser.add_argument('--sentence_max_len', type=int, default=80)
    parser.add_argument('--queue_length', type=int, default=2048)
    parser.add_argument('--momentum', type=float, default=0.999)
    parser.add_argument('--T_para', type=float, default=0.04)
    parser.add_argument('--layer_id', type=int, default=12)
    parser.add_argument('--dev_only_q_encoder', type=int, default=0)
    # For test
    parser.add_argument('--dev_all_sentence_num', type=int, default=32, help="only for predict")
    parser.add_argument('--dev_sample_num', type=int, default=32)
    parser.add_argument('--train_sample_num', type=int, default=4)
    parser.add_argument('--dataset_path',type=str,default='./data/')
    parser.add_argument('--load_model_path', type=str, default='result')
    parser.add_argument('--unsupervised', type=int, default=0)
    parser.add_argument('--test_dev', type=int, default=0)
    args = parser.parse_args()
    return args
