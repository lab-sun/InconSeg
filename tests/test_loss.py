import torch

from util.loss import fusion_loss


def test_fusion_loss_finite_and_grads():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 16, 20, requires_grad=True)
    depth_add = torch.randn(2, 3, 16, 20, requires_grad=True)
    label = torch.randint(0, 3, (2, 16, 20))
    loss_pr, loss_add, add_map = fusion_loss(logits, depth_add, label)
    assert loss_pr.ndim == 0 and torch.isfinite(loss_pr)
    assert loss_add.ndim == 0 and torch.isfinite(loss_add)
    assert add_map.shape == label.shape
    assert add_map.dtype == torch.long
    (loss_pr + loss_add).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert depth_add.grad is not None and torch.isfinite(depth_add.grad).all()
    assert logits.grad.abs().sum() > 0
    assert depth_add.grad.abs().sum() > 0


def test_fusion_loss_label_size_differs():
    torch.manual_seed(1)
    logits = torch.randn(1, 3, 8, 12, requires_grad=True)
    depth_add = torch.randn(1, 3, 8, 12, requires_grad=True)
    label = torch.randint(0, 3, (1, 20, 30))
    loss_pr, loss_add, add_map = fusion_loss(logits, depth_add, label)
    assert torch.isfinite(loss_pr) and torch.isfinite(loss_add)
    assert add_map.shape == (1, 8, 12)
    assert loss_pr.device == label.device
    assert add_map.device == label.device
    (loss_pr + loss_add).backward()
    assert logits.grad is not None and logits.grad.shape == logits.shape
    assert depth_add.grad is not None and torch.isfinite(depth_add.grad).all()
    assert logits.grad.abs().sum() > 0
    assert depth_add.grad.abs().sum() > 0
