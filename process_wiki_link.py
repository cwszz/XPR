import argparse
import csv
import re
from tqdm import tqdm

def parse_url(url_str):
    if not isinstance(url_str, str):
        raise ValueError
    if not url_str.startswith("<http://"): return None
    url_str = url_str[len("<http://"):]
    if not url_str.endswith(">"): return None
    url_str = url_str[:-1]

    _split = url_str.split("dbpedia.org/resource/")
    # print(_split)
    if len(_split) != 2:
        # print("[W] url_str %s is not valid" % url_str)
        return None

    lang, word = _split
    if lang == "":
        lang = "en"
    elif lang.endswith("."):
        lang = lang[:-1]
    else:
        return None
    word = word.replace("_", " ")
    return lang, word


def parse_line(line):
    # print(line)
    urls = line.strip().split()
    r0 = parse_url(urls[0])
    r1 = parse_url(urls[2])
    if r0 is None or r1 is None: return None
    lang0, word0 = r0
    lang1, word1 = r1
    return lang0, word0, lang1, word1


def main(args):
    # lines = []
    # trg_langs = [args.trg_lang1,args.trg_lang2,args.trg_lang3,args.trg_lang4,args.trg_lang5,args.trg_lang6]
    trg_langs = [args.trg_lang7]
    csv_writers = [csv.writer(open('en-{}.txt'.format(each_trg), 'w'), delimiter='\t') for each_trg in trg_langs]
    count = [0] * len(trg_langs)
    with open(args.input_file) as fp:
        for line in tqdm(fp):
            ret = parse_line(line)
            if ret is None: continue
            lang0, word0, lang1, word1 = ret
            if lang1 == args.src_lang and lang0 in trg_langs:
                lang0, lang1 = lang1, lang0
                word0, word1 = word1, word0
            if lang0 == args.src_lang and lang1 in trg_langs:
                if not re.search(r'\d',word0) and ' ' in word0 and word1 != word0:
                    this_index = trg_langs.index(lang1)
                    csv_writers[this_index].writerow([word0,word1])
                    count[this_index] += 1
    print(count)
    # for line in lines:
    # parse_line(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="./interlanguage_links_en.ttl")
    parser.add_argument("--src_lang", type=str, default="en")
    # parser.add_argument("--trg_lang1", type=str, default="de")
    # parser.add_argument("--trg_lang2", type=str, default="es")
    # parser.add_argument("--trg_lang3", type=str, default="ru")
    # parser.add_argument("--trg_lang4", type=str, default="ro")
    # parser.add_argument("--trg_lang5", type=str, default="zh")
    # parser.add_argument("--trg_lang6", type=str, default="ko")
    parser.add_argument("--trg_lang7", type=str, default="ar")
    # parser.add_argument("--trg_lang8", type=str, default="fr")

    args = parser.parse_args()
    main(args)