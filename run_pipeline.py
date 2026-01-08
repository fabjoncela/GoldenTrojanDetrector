import numpy as np
from src.preprocessing import preprocess
from src.pair_generator import make_pairs
from src.train import train


preprocess(
"data/raw/normal",
"data/raw/trojan/triggered",
"data/processed/data.npz"
)


data = np.load("data/processed/data.npz")
p1, p2, y = make_pairs(data["normal"], data["trojan"])
train(p1, p2, y)