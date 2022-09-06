import argparse
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

parser = argparse.ArgumentParser()
# High Resources
parser.add_argument("--ende_num", type=int, default=int(1e6))
parser.add_argument("--enzh_num", type=int, default=int(1e6))
# Medium Resources
parser.add_argument("--roen_num", type=int, default=int(2e5))
parser.add_argument("--eten_num", type=int, default=int(2e5))
# Low Resources
parser.add_argument("--neen_num", type=int, default=int(5e4))
parser.add_argument("--sien_num", type=int, default=int(5e4))
# PATH
parser.add_argument("--train_path", default="./train/train_1M_200k_50k.pt")
parser.add_argument("--valid_path", default="./train/valid_1M_200k_50k.pt")
# other
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

lang_pairs = ["ende", "enzh", "roen", "eten", "neen", "sien"]

sample_num_dict = {
    "ende": args.ende_num,
    "enzh": args.enzh_num,
    "roen": args.roen_num,
    "eten": args.eten_num,
    "neen": args.neen_num,
    "sien": args.sien_num,
}

lang_num_dict = {"en": 0, "de": 1, "zh": 2, "ro": 3, "et": 4, "ne": 5, "si": 6}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model_name = "sentence-transformers/LaBSE"

random.seed(args.seed)


def sampling_train_data(lang_pairs, sample_num_dict):
    print("Sampling data...")

    for lang_pair in lang_pairs:
        src_lang = lang_pair[:2]
        trg_lang = lang_pair[2:]

        sample_num = sample_num_dict[lang_pair]
        print(f"{lang_pair}: {sample_num:<8d} [pairs]")

        with open(f"./train/train.{lang_pair}.{src_lang}.detok", "r") as f:
            src_sentences = f.read().rstrip()
            src_sentences = src_sentences.split("\n")

        with open(f"./train/train.{lang_pair}.{trg_lang}.detok", "r") as f:
            trg_sentences = f.read().rstrip()
            trg_sentences = trg_sentences.split("\n")

        train = []
        for s, t in zip(src_sentences, trg_sentences):
            if s != "" and t != "":
                train.append((s, t))

        sample_train = random.sample(train, sample_num)
        sample_src_sentences = [t[0] for t in sample_train]
        sample_trg_sentences = [t[1] for t in sample_train]

        with open(f"./train/train.{lang_pair}.{src_lang}.{sample_num}.detok", "w") as f:
            f.write("\n".join(sample_src_sentences))

        with open(f"./train/train.{lang_pair}.{trg_lang}.{sample_num}.detok", "w") as f:
            f.write("\n".join(sample_trg_sentences))


def embedding(tokenizer, base_model, sentences, batch_size, device):
    base_model.to(device)

    all_embeddings = []

    length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
    sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

    for i in tqdm(range(-(-len(sentences) // batch_size))):
        sentence_batch = sentences_sorted[i * batch_size : (i + 1) * batch_size]

        encoded = tokenizer(sentence_batch, return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            outputs = base_model(**encoded.to(device))

        # last hidden state
        embeddings = outputs[0][:, 0, :]

        all_embeddings.extend(embeddings.to("cpu"))

    all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
    all_embeddings = torch.stack(all_embeddings)

    return all_embeddings


def embedding_train_data(base_model_name, lang_pairs, sample_num_dict, batch_size, device):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModel.from_pretrained(base_model_name)

    for lang_pair in lang_pairs:
        print(f"{lang_pair}: Embedding...")

        src_lang = lang_pair[:2]
        trg_lang = lang_pair[2:]
        sample_num = sample_num_dict[lang_pair]

        with open(f"./train/train.{lang_pair}.{src_lang}.{sample_num}.detok", "r") as f:
            src_sentences = f.read().rstrip()
            src_sentences = src_sentences.split("\n")

        with open(f"./train/train.{lang_pair}.{trg_lang}.{sample_num}.detok", "r") as f:
            trg_sentences = f.read().rstrip()
            trg_sentences = trg_sentences.split("\n")

        src_embeddings = embedding(tokenizer, base_model, src_sentences, batch_size, device)
        trg_embeddings = embedding(tokenizer, base_model, trg_sentences, batch_size, device)

        torch.save(src_embeddings, f"./train/train.{lang_pair}.{src_lang}.{sample_num}.detok.emb.pt")
        torch.save(trg_embeddings, f"./train/train.{lang_pair}.{trg_lang}.{sample_num}.detok.emb.pt")


def train_valid_split(lang_pairs, sample_num_dict, train_path, valid_path, seed):
    src_emb_all = []
    trg_emb_all = []
    src_lang_all = []
    trg_lang_all = []
    for lang_pair in lang_pairs:
        print(lang_pair)

        src_lang = lang_pair[:2]
        trg_lang = lang_pair[2:]
        sample_num = sample_num_dict[lang_pair]

        src_emb = torch.load(f"./train/train.{lang_pair}.{src_lang}.{sample_num}.detok.emb.pt")
        trg_emb = torch.load(f"./train/train.{lang_pair}.{trg_lang}.{sample_num}.detok.emb.pt")
        src_emb_all.extend(src_emb)
        trg_emb_all.extend(trg_emb)

        src_lang_num = lang_num_dict[src_lang]
        trg_lang_num = lang_num_dict[trg_lang]
        tmp = torch.tensor([[src_lang_num] for _ in range(len(src_emb))])
        src_lang_all.extend(tmp)
        tmp = torch.tensor([[trg_lang_num] for _ in range(len(trg_emb))])
        trg_lang_all.extend(tmp)

    src_emb_all = torch.stack(src_emb_all)
    trg_emb_all = torch.stack(trg_emb_all)
    src_lang_all = torch.stack(src_lang_all)
    trg_lang_all = torch.stack(trg_lang_all)

    train_src_emb, valid_src_emb = train_test_split(src_emb_all, test_size=0.1, random_state=seed)
    train_trg_emb, valid_trg_emb = train_test_split(trg_emb_all, test_size=0.1, random_state=seed)
    train_src_lang, valid_src_lang = train_test_split(src_lang_all, test_size=0.1, random_state=seed)
    train_trg_lang, valid_trg_lang = train_test_split(trg_lang_all, test_size=0.1, random_state=seed)

    train = {
        "src_emb": train_src_emb,
        "trg_emb": train_trg_emb,
        "src_lang": train_src_lang,
        "trg_lang": train_trg_lang,
    }
    torch.save(train, train_path)

    valid = {
        "src_emb": valid_src_emb,
        "trg_emb": valid_trg_emb,
        "src_lang": valid_src_lang,
        "trg_lang": valid_trg_lang,
    }
    torch.save(valid, valid_path)


def main():
    sampling_train_data(lang_pairs, sample_num_dict)
    embedding_train_data(base_model_name, lang_pairs, sample_num_dict, args.batch_size, device)
    train_valid_split(lang_pairs, sample_num_dict, args.train_path, args.valid_path, args.seed)


if __name__ == "__main__":
    main()
