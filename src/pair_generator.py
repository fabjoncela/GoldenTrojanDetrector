import numpy as np




def make_pairs(normal, trojan, num_pairs=10000):
    pairs_1, pairs_2, labels = [], [], []


    for _ in range(num_pairs // 2):
        i, j = np.random.choice(len(normal), 2, replace=False)
        pairs_1.append(normal[i])
        pairs_2.append(normal[j])
        labels.append(1)


    for _ in range(num_pairs // 2):
        i = np.random.randint(len(normal))
        j = np.random.randint(len(trojan))
        pairs_1.append(normal[i])
        pairs_2.append(trojan[j])
        labels.append(0)


    return np.array(pairs_1), np.array(pairs_2), np.array(labels)