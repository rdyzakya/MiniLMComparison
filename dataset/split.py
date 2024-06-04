import os
from sklearn.model_selection import train_test_split

listdir = os.listdir("./raw")

for fname in listdir:
    path = os.path.join("./raw", fname)
    with open(path, 'r', encoding="utf-8") as fp:
        data = fp.read()

    data = data.strip().splitlines()
    unique_data = []
    for line in data:
        if line not in unique_data:
            unique_data.append(line)
    
    train, test = train_test_split(unique_data, test_size=0.2, random_state=42, shuffle=True)

    train = '\n'.join(train)
    test = '\n'.join(test)

    foldername = os.path.join("./proc", fname.replace(".txt", ''))
    os.makedirs(foldername, exist_ok=True)

    with open(os.path.join(foldername, "train.txt"), 'w', encoding="utf-8") as fp:
        fp.write(train)

    with open(os.path.join(foldername, "test.txt"), 'w', encoding="utf-8") as fp:
        fp.write(test)