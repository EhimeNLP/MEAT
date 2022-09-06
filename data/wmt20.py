import os

import pandas as pd


def main():
    lang_pairs = ["ende", "enzh", "roen", "eten", "neen", "sien"]
    for lang_pair in lang_pairs:
        df = pd.read_table(f"test/test20/test20.{lang_pair}.df.short.tsv", quoting=3)

        dir_path = f"test/test20/{lang_pair[:2] + '-' + lang_pair[2:]}-test20"
        os.makedirs(dir_path, exist_ok=True)

        with open(f"{dir_path}/test20.src", "w") as f:
            src_text = df["original"].values.tolist()
            f.write("\n".join(src_text))

        with open(f"{dir_path}/test20.mt", "w") as f:
            trg_text = df["translation"].values.tolist()
            f.write("\n".join(trg_text))

        with open(f"{dir_path}/test20.da", "w") as f:
            score = df["z_mean"].values.tolist()
            score = list(map(str, score))
            f.write("\n".join(score))

        # os.remove(f"test/test20/test20.{lang_pair}.df.short.tsv")


if __name__ == "__main__":
    main()
