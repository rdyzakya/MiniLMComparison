import os
import shutil
import numpy as np

proc = "./proc"

listdir = os.listdir(proc)

# DATA LENGTH

train_length = []

for artist in listdir:
    train_path = os.path.join(proc, artist, "train.txt")
    with open(train_path, 'r', encoding="utf-8") as fp:
        data = fp.read().splitlines()
    train_length.append(len(data))

max_len = np.max(train_length)
min_len = np.min(train_length)
median_len = int(np.median(train_length))

max_idx = np.argmax(train_length)
min_idx = np.argmin(train_length)
median_idx = np.argsort(train_length)[len(train_length)//2]

print("MAX - MEDIAN - MIN")
print(listdir[max_idx], ':', max_len)
print(listdir[median_idx], ':', median_len)
print(listdir[min_idx], ':', min_len)

os.makedirs("./small-medium-large", exist_ok=True)

shutil.copytree(os.path.join(proc, listdir[max_idx]), "./small-medium-large/large")
shutil.copytree(os.path.join(proc, listdir[median_idx]), "./small-medium-large/medium")
shutil.copytree(os.path.join(proc, listdir[min_idx]), "./small-medium-large/small")

with open("./small-medium-large/artist.txt", 'w') as fp:
    fp.write(' '.join(["large :", listdir[max_idx], '-', str(max_len)]) + '\n')
    fp.write(' '.join(["medium :", listdir[median_idx], '-', str(median_len)]) + '\n')
    fp.write(' '.join(["small :", listdir[min_idx], '-', str(min_len)]) + '\n')