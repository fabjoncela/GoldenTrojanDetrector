import torch
from typing import Optional
from torch.utils.data import DataLoader, TensorDataset
from .model import SiameseNet
from .loss import contrastive_loss
from .config import *




def train(p1, p2, labels, progress_callback=None, epochs: Optional[int] = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"


    model = SiameseNet(FEATURES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


    dataset = TensorDataset(
        torch.tensor(p1, dtype=torch.float32),
        torch.tensor(p2, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32)
    )


    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


    history = []

    num_epochs = epochs if epochs is not None else EPOCHS

    for epoch in range(num_epochs):
        total_loss = 0
        for x1, x2, y in loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            z1, z2 = model(x1, x2)
            loss = contrastive_loss(z1, z2, y, MARGIN)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        history.append(avg_loss)
        if progress_callback is not None:
            progress_callback(epoch + 1, avg_loss)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    torch.save(model.state_dict(), "siamese_model.pt")
    return history