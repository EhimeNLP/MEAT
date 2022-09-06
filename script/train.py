import argparse
import time

import torch
import torch.nn as nn

from model import MLP, Discriminator

parser = argparse.ArgumentParser()
# PATH
parser.add_argument("--save_model_path", default="../output/1M_200k_50k/best_val.pt")
parser.add_argument("--train_path", default="../data/train/train_1M_200k_50k.pt")
parser.add_argument("--valid_path", default="../data/train/valid_1M_200k_50k.pt")
# train parameter
parser.add_argument("--lr", default=1e-5)
parser.add_argument("--lang_num", type=int, default=7)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--seed", type=int, default=100)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, src_emb, trg_emb, src_lang, trg_lang):
        self.src_emb = src_emb
        self.trg_emb = trg_emb
        self.src_lang = src_lang
        self.trg_lang = trg_lang

    def __len__(self):
        return len(self.src_emb)

    def __getitem__(self, idx):
        return {
            "src_emb": self.src_emb[idx],
            "trg_emb": self.trg_emb[idx],
            "src_lang": self.src_lang[idx],
            "trg_lang": self.trg_lang[idx],
        }


def calculate_loss(me_src, le_src, li_me_src, me_trg, le_trg, li_me_trg, src_emb, trg_emb, lang_num):
    cos_fn = nn.CosineEmbeddingLoss()
    cross_fn = nn.CrossEntropyLoss()

    y = torch.ones(me_src.size(0), device=device)

    loss_recon = cos_fn(me_src + le_src, src_emb, y) + cos_fn(me_trg + le_trg, trg_emb, y)

    loss_cross_recon = cos_fn(me_src + le_trg, trg_emb, y) + cos_fn(me_trg + le_src, src_emb, y)

    loss_lang_emb = cos_fn(le_src, le_trg, -y)

    y = torch.full(li_me_src.size(), fill_value=1 / lang_num, device=device)
    loss_adv = cross_fn(li_me_src, y) + cross_fn(li_me_trg, y)

    loss = loss_recon + loss_cross_recon + loss_lang_emb + loss_adv

    return loss


def calculate_discriminator_loss(li_me_src, li_me_trg, src_lang, trg_lang):
    cross_fn = nn.CrossEntropyLoss()

    src_lang = torch.squeeze(src_lang).long()
    trg_lang = torch.squeeze(trg_lang).long()

    loss_adv = cross_fn(li_me_src, src_lang) + cross_fn(li_me_trg, trg_lang)

    return loss_adv


def train_model(
    model,
    discriminator,
    dataset_train,
    dataset_valid,
    optimizer,
    discriminator_optimizer,
    lang_num,
    batch_size,
    save_model_path,
):
    model.to(device)
    discriminator.to(device)

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
    dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)

    min_valid_loss = float("inf")
    for epoch in range(10000):
        # train
        s_time = time.time()
        train_loss = 0
        for data in dataloader_train:
            src_emb = data["src_emb"].to(device)
            trg_emb = data["trg_emb"].to(device)
            src_lang = data["src_lang"].to(device)
            trg_lang = data["trg_lang"].to(device)

            me_src, le_src = model(src_emb)
            me_trg, le_trg = model(trg_emb)
            li_me_src = discriminator(me_src)
            li_me_trg = discriminator(me_trg)

            discriminator_optimizer.zero_grad()
            loss = calculate_discriminator_loss(li_me_src, li_me_trg, src_lang, trg_lang)
            loss.backward()
            discriminator_optimizer.step()

            me_src, le_src = model(src_emb)
            me_trg, le_trg = model(trg_emb)
            li_me_src = discriminator(me_src)
            li_me_trg = discriminator(me_trg)

            optimizer.zero_grad()
            loss = calculate_loss(me_src, le_src, li_me_src, me_trg, le_trg, li_me_trg, src_emb, trg_emb, lang_num)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        # eval
        with torch.no_grad():
            valid_loss = 0
            for data in dataloader_valid:
                src_emb = data["src_emb"].to(device)
                trg_emb = data["trg_emb"].to(device)

                me_src, le_src = model(src_emb)
                me_trg, le_trg = model(trg_emb)
                li_me_src = discriminator(me_src)
                li_me_trg = discriminator(me_trg)
                loss = calculate_loss(
                    me_src, le_src, li_me_src, me_trg, le_trg, li_me_trg, src_emb, trg_emb, lang_num
                )
                valid_loss += loss.item()

            print(
                f"epoch:{epoch + 1: <2}, "
                f"train_loss: {train_loss / len(dataloader_train):.5f}, "
                f"valid_loss: {valid_loss / len(dataloader_valid):.5f}, "
                f"{(time.time() - s_time) / 60:.1f}[min]"
            )

            if valid_loss < min_valid_loss:
                epochs_no_improve = 0
                min_valid_loss = valid_loss
                torch.save(model.to("cpu").state_dict(), save_model_path)
                model.to(device)
            else:
                epochs_no_improve += 1

        if epochs_no_improve >= 10:
            break


def main():
    data_train = torch.load(args.train_path)
    dataset_train = TextDataset(
        data_train["src_emb"], data_train["trg_emb"], data_train["src_lang"], data_train["trg_lang"]
    )
    data_valid = torch.load(args.valid_path)
    dataset_valid = TextDataset(
        data_valid["src_emb"], data_valid["trg_emb"], data_valid["src_lang"], data_valid["trg_lang"]
    )

    model = MLP()
    discriminator = Discriminator(lang_num=args.lang_num)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    train_model(
        model,
        discriminator,
        dataset_train,
        dataset_valid,
        optimizer,
        discriminator_optimizer,
        args.lang_num,
        args.batch_size,
        args.save_model_path,
    )


if __name__ == "__main__":
    main()
