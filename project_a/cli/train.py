import torch
from torch import nn
from torch.nn import functional as F

from refiner.refiner import Refiner


def seg_loss(logits, g, ignore_index=None):
    g = g.float()
    valid = torch.ones_like(g) if ignore_index is None else (g != ignore_index).float()
    g = g * valid
    bce = F.binary_cross_entropy_with_logits(logits, g, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp(min=1)
    p = torch.sigmoid(logits) * valid
    dice = 1 - (2 * (p * g).sum() + 1) / (p.sum() + g.sum() + 1)
    return bce + dice


def train(refiner, baseline, loader, K=4, epochs=10, lr=1e-4, device="cpu", ignore_index=None, save_path="refiner.pt"):
    refiner.to(device)
    baseline.to(device).eval()
    for p in baseline.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(refiner.parameters(), lr=lr, weight_decay=0.01)
    for epoch in range(epochs):
        refiner.train()
        total = 0.0
        for image, g in loader:
            image, g = image.to(device), g.to(device)
            with torch.no_grad():
                f, y0 = baseline(image)
            ys = refiner(f, y0, K, return_all=True)
            loss = sum(seg_loss(y, g, ignore_index) for y in ys) / len(ys)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
            opt.step()
            total += loss.item()
        print(f"epoch {epoch + 1}: loss {total / max(len(loader), 1):.4f}")
    torch.save(refiner.state_dict(), save_path)
    return refiner


class DemoBaseline(nn.Module):

    def __init__(self, c=64, patch=8) -> None:
        super().__init__()
        self.enc = nn.Conv2d(3, c, kernel_size=patch, stride=patch)
        self.h0 = nn.Conv2d(c, 1, kernel_size=1)
        self.patch = patch

    def forward(self, x):
        f = self.enc(x)
        y0 = F.interpolate(self.h0(f), scale_factor=self.patch, mode="bilinear", align_corners=False)
        return f, y0


if __name__ == "__main__":
    torch.manual_seed(0)
    images = torch.randn(8, 3, 64, 80)
    masks = (torch.rand(8, 1, 64, 80) > 0.8).float()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images, masks), batch_size=4)
    refiner = Refiner(c_in=64, d=128, patch=8, nz=16, nhead=4, num_layers=2)
    train(refiner, DemoBaseline(c=64, patch=8), loader, K=2, epochs=2, save_path="refiner_demo.pt")
    print("DEMO ONLY: random data and a random baseline")
