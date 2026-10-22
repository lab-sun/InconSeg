import torch
import torch.nn.functional as F

from model.InconSeg import InconSeg
from util.loss import fusion_loss

N_CLASS = 3


def _model():
    torch.manual_seed(0)
    model = InconSeg(N_CLASS, pretrained=False)
    model.eval()
    return model


def test_forward_output_count_and_shapes():
    model = _model()
    x = torch.randn(1, 4, 64, 96)
    with torch.no_grad():
        outputs = model(x)
    assert len(outputs) == 8
    expected = [
        (1, N_CLASS, 64, 96),
        (1, N_CLASS, 64, 96),
        (1, N_CLASS, 64, 96),
        (1, N_CLASS, 64, 96),
        (1, N_CLASS, 32, 48),
        (1, N_CLASS, 32, 48),
        (1, N_CLASS, 16, 24),
        (1, N_CLASS, 16, 24),
    ]
    for output, shape in zip(outputs, expected):
        assert tuple(output.shape) == shape
        assert torch.isfinite(output).all()


def test_state_dict_roundtrip(tmp_path):
    model = _model()
    x = torch.randn(1, 4, 64, 96)
    with torch.no_grad():
        reference = model(x)
    path = tmp_path / 'inconseg.pth'
    torch.save(model.state_dict(), path)
    loaded = torch.load(path, map_location='cpu', weights_only=True)
    restored = InconSeg(N_CLASS, pretrained=False)
    restored.eval()
    restored.load_state_dict(loaded)
    with torch.no_grad():
        outputs = restored(x)
    assert len(outputs) == len(reference)
    for got, expected in zip(outputs, reference):
        assert torch.equal(got, expected)


def test_one_sgd_step_on_cpu():
    torch.manual_seed(0)
    model = InconSeg(N_CLASS, pretrained=False)
    model.train()
    images = torch.randn(2, 4, 64, 96)
    labels = torch.randint(0, N_CLASS, (2, 64, 96))
    before = [param.detach().clone() for param in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    _depth_result, rgb_result, rgb_seg_f1, depth_add_f1, *_rest = model(images)
    loss_r = F.cross_entropy(rgb_result, labels)
    loss_pr_seg_f1, loss_add_f1, _add_map = fusion_loss(rgb_seg_f1, depth_add_f1, label=labels)
    loss = loss_r + loss_pr_seg_f1 + loss_add_f1
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    changed = any(
        not torch.equal(param, prev)
        for param, prev in zip(model.parameters(), before)
    )
    assert changed
