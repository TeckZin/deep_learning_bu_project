import torch
from torch import nn

from tokenizer.Tx import Tx, pos2d
from tokenizer.Ty import Ty
from transformer.core import Core
from refiner.memory import Initializer
from refiner.head import Head
from refiner.refiner import Refiner
from train import DemoBaseline, seg_loss

B, C, H, W, PATCH, D, NZ = 2, 64, 64, 80, 8, 128, 16
h, w = H // PATCH, W // PATCH
N = h * w


def make_model():
    torch.manual_seed(0)
    return Refiner(c_in=C, d=D, patch=PATCH, nz=NZ, nhead=4, num_layers=2)


def make_inputs():
    torch.manual_seed(1)
    return torch.randn(B, C, h, w), torch.randn(B, 1, H, W)


def wake_head(model):
    torch.manual_seed(2)
    nn.init.normal_(model.head.proj.weight, std=0.02)
    nn.init.normal_(model.head.proj.bias, std=0.02)


def count_core_calls(model):
    calls = [0]
    model.core.register_forward_hook(lambda m, i, o: calls.__setitem__(0, calls[0] + 1))
    return calls


def test_pos2d():
    pe = pos2d(h, w, D)
    assert pe.shape == (N, D)
    assert torch.unique(pe, dim=0).shape[0] == N


def test_tx():
    f, _ = make_inputs()
    assert Tx(C, D)(f).shape == (B, N, D)


def test_ty():
    _, y0 = make_inputs()
    assert Ty(D, PATCH)(y0).shape == (B, N, D)


def test_core():
    x = torch.randn(B, 2 * N + NZ, D)
    assert Core(D, 4, 2)(x).shape == x.shape


def test_initializer():
    img, ans = torch.randn(B, N, D), torch.randn(B, N, D)
    assert Initializer(D, NZ, 4)(img, ans).shape == (B, NZ, D)


def test_head():
    head = Head(D, PATCH)
    out = head(torch.randn(B, N, D), h, w, out_size=(H, W))
    assert out.shape == (B, 1, H, W)
    assert torch.count_nonzero(out) == 0


def test_output_shape():
    model = make_model().eval()
    f, y0 = make_inputs()
    with torch.no_grad():
        assert model(f, y0, 3).shape == (B, 1, H, W)


def test_k0_identity():
    model = make_model().eval()
    calls = count_core_calls(model)
    f, y0 = make_inputs()
    with torch.no_grad():
        out = model(f, y0, 0)
    assert torch.equal(out, y0)
    assert calls[0] == 0


def test_untrained_returns_baseline():
    model = make_model().eval()
    f, y0 = make_inputs()
    with torch.no_grad():
        assert torch.allclose(model(f, y0, 4), y0)


def test_core_call_count():
    model = make_model().eval()
    calls = count_core_calls(model)
    f, y0 = make_inputs()
    with torch.no_grad():
        for K in (1, 2, 4, 6):
            calls[0] = 0
            model(f, y0, K)
            assert calls[0] == K, f"K={K}: {calls[0]} core calls"


def test_prefix_consistency():
    model = make_model().eval()
    wake_head(model)
    f, y0 = make_inputs()
    with torch.no_grad():
        y2 = model(f, y0, 2)
        y4 = model(f, y0, 4, return_all=True)
    assert len(y4) == 4
    assert torch.allclose(y2, y4[1], atol=1e-5)


def test_feedback():
    model = make_model().eval()
    wake_head(model)
    f, y0 = make_inputs()
    with torch.no_grad():
        ys = model(f, y0, 3, return_all=True)
        img = model.tx(f)
        zt = model.init(img, model.ty(y0))
        yt = y0
        for t in range(3):
            out = model.core(torch.cat([img, model.ty(yt), zt], dim=1))
            _, ans_out, zt = out.split([N, N, NZ], dim=1)
            yt = yt + model.head(ans_out, h, w, out_size=(H, W))
            assert torch.allclose(yt, ys[t], atol=1e-5), f"step {t + 1} does not feed back"
    assert not torch.allclose(ys[0], ys[1])


def test_gradients_reach_refiner_only():
    model = make_model().train()
    wake_head(model)
    baseline = DemoBaseline(c=C, patch=PATCH)
    for p in baseline.parameters():
        p.requires_grad = False
    image = torch.randn(B, 3, H, W)
    g = (torch.rand(B, 1, H, W) > 0.8).float()
    with torch.no_grad():
        f, y0 = baseline(image)
    loss = seg_loss(model(f, y0, 2), g)
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient for: {missing}"
    assert all(p.grad is None for p in baseline.parameters())


def test_ignore_index():
    logits = torch.zeros(1, 1, 4, 4)
    g = torch.full((1, 1, 4, 4), 255.0)
    g[..., 0, 0] = 1.0
    assert torch.isfinite(seg_loss(logits, g, ignore_index=255))


def test_training_lowers_loss():
    torch.manual_seed(3)
    model = Refiner(c_in=C, d=64, patch=PATCH, nz=8, nhead=4, num_layers=1, dropout=0.0)
    baseline = DemoBaseline(c=C, patch=PATCH).eval()
    image = torch.randn(B, 3, H, W)
    g = (torch.rand(B, 1, H, W) > 0.8).float()
    with torch.no_grad():
        f, y0 = baseline(image)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(40):
        loss = seg_loss(model(f, y0, 2), g)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"loss went {losses[0]:.4f} -> {losses[-1]:.4f}"


TESTS = [
    test_pos2d, test_tx, test_ty, test_core, test_initializer, test_head,
    test_output_shape, test_k0_identity, test_untrained_returns_baseline,
    test_core_call_count, test_prefix_consistency, test_feedback,
    test_gradients_reach_refiner_only, test_ignore_index, test_training_lowers_loss,
]


if __name__ == "__main__":
    failed = 0
    for test in TESTS:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {test.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    raise SystemExit(1 if failed else 0)
