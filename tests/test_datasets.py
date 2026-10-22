import numpy as np
import PIL.Image
import pytest
import torch

from util.RGB_Depth_dataset import RGB_Depth_dataset
from util.RGB_Disparity_dataset import RGB_Disparity_dataset


def _write_split(root, folder, head):
    for sub in ('left', 'labels', folder):
        (root / sub).mkdir(parents=True)
    names = ['0001', '0002']
    (root / 'train.txt').write_text('\n'.join(names) + '\n')
    for name, zero in (('0001', True), ('0002', False)):
        rgb = np.zeros((30, 40, 3), dtype=np.uint8)
        rgb[:] = (12, 34, 56)
        PIL.Image.fromarray(rgb).save(root / 'left' / f'left{name}.png')
        label = np.zeros((30, 40), dtype=np.uint8)
        label[5:15, 5:15] = 1
        label[15:25, 15:25] = 2
        PIL.Image.fromarray(label).save(root / 'labels' / f'label{name}.png')
        depth = np.zeros((30, 40), dtype=np.uint8)
        if not zero:
            depth[:, :] = 40
            depth[8:20, 10:28] = 180
        PIL.Image.fromarray(depth).save(root / folder / f'{head}{name}.png')
    return names


@pytest.mark.parametrize('dataset_cls,folder,head', [
    (RGB_Depth_dataset, 'depth', 'depth'),
    (RGB_Disparity_dataset, 'disparity', 'disparity'),
])
def test_dataset_reads_pngs_and_zero_map(tmp_path, dataset_cls, folder, head):
    root = tmp_path / 'data'
    _write_split(root, folder, head)
    dataset = dataset_cls(str(root), 'train', input_h=32, input_w=48)
    assert len(dataset) == 2

    image0, label0, name0 = dataset[0]
    assert name0 == '0001'
    assert tuple(image0.shape) == (4, 32, 48)
    assert tuple(label0.shape) == (32, 48)
    assert label0.dtype == torch.int64
    assert torch.isfinite(image0).all()
    assert not torch.isnan(image0).any()
    assert torch.equal(image0[3], torch.zeros_like(image0[3]))
    assert set(torch.unique(label0).tolist()).issubset({0, 1, 2})

    image1, label1, name1 = dataset[1]
    assert name1 == '0002'
    assert torch.isfinite(image1).all()
    assert not torch.isnan(image1).any()
    assert image1[3].max().item() == pytest.approx(1.0)
    assert image1[:3].min() >= 0 and image1[:3].max() <= 1
    assert label1.dtype == torch.int64


def _write_flat_rgb(root, folder, head, pixels, height, width):
    for sub in ('left', 'labels', folder):
        (root / sub).mkdir(parents=True)
    names = [f'{pixel:04d}' for pixel in pixels]
    (root / 'train.txt').write_text('\n'.join(names) + '\n')
    label = np.zeros((height, width), dtype=np.uint8)
    depth = np.zeros((height, width), dtype=np.uint8)
    for name, pixel in zip(names, pixels):
        rgb = np.full((height, width, 3), pixel, dtype=np.uint8)
        PIL.Image.fromarray(rgb).save(root / 'left' / f'left{name}.png')
        PIL.Image.fromarray(label).save(root / 'labels' / f'label{name}.png')
        PIL.Image.fromarray(depth).save(root / folder / f'{head}{name}.png')
    return names


@pytest.mark.parametrize('dataset_cls,folder,head', [
    (RGB_Depth_dataset, 'depth', 'depth'),
    (RGB_Disparity_dataset, 'disparity', 'disparity'),
])
def test_rgb_pixels_scale_by_255(tmp_path, dataset_cls, folder, head):
    root = tmp_path / 'data'
    height, width = 4, 6
    names = _write_flat_rgb(root, folder, head, (255, 0, 51), height, width)
    dataset = dataset_cls(str(root), 'train', input_h=height, input_w=width)

    scaled = {}
    for index, name in enumerate(names):
        image, _, got_name = dataset[index]
        assert got_name == name
        scaled[name] = image[:3]

    assert torch.equal(scaled['0255'], torch.ones_like(scaled['0255']))
    assert torch.equal(scaled['0000'], torch.zeros_like(scaled['0000']))
    assert scaled['0051'].numpy() == pytest.approx(0.2, abs=1e-6)


@pytest.mark.parametrize('dataset_cls,folder,head', [
    (RGB_Depth_dataset, 'depth', 'depth'),
    (RGB_Disparity_dataset, 'disparity', 'disparity'),
])
def test_default_input_size(tmp_path, dataset_cls, folder, head):
    root = tmp_path / 'data'
    _write_flat_rgb(root, folder, head, (0,), 2, 2)
    dataset = dataset_cls(str(root), 'train')
    image, label, _ = dataset[0]
    assert tuple(image.shape) == (4, 288, 512)
    assert tuple(label.shape) == (288, 512)
