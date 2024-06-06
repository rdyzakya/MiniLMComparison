import pandas as pd
import re
import os
import random

pattern = r"[a-z\s]"

df = pd.read_csv("./raw/podcastdata_dataset.csv")

# lengths = {
#     "short" : 128,
#     "medium" : 256,
#     "long" : 512
# }

# os.makedirs("./short-medium-long", exist_ok=True)

# 2000 text only
N_DATA = 2000
LENGTH = 1024

# for k, v in lengths.items():
os.makedirs("./long-text", exist_ok=True)
res = []

df = df.sample(frac=1.0, random_state=42)
for i, row in df.iterrows():
    text = ""
    for c in row["text"]:
        text += c
        if len(re.findall(pattern, text.lower())) >= LENGTH:
            res.append(text)
            text = ""
    if len(res) >= N_DATA:
        break

random.seed(42)
random.shuffle(res)

res = res[:N_DATA]

train = res[:int(N_DATA * 0.8)]
test = res[int(N_DATA * 0.8):]

train = '\n'.join(train)
test = '\n'.join(test)

with open(os.path.join("./long-text", "train.txt"), 'w', encoding="utf-8") as fp:
    fp.write(train)

with open(os.path.join("./long-text", "test.txt"), 'w', encoding="utf-8") as fp:
    fp.write(test)