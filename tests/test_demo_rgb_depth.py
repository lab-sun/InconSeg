import gc
import os
import runpy
import stat
import sys

import numpy as np
import PIL.Image
import torch
from torchvision import models

from model.InconSeg import InconSeg

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WEIGHT_NAME = 'InconSeg_RGB_depth'
NAMES = ['000001', '000002', '000003']


def _write_synthetic_dataset(data_dir):
    for sub in ('left', 'labels', 'depth'):
        (data_dir / sub).mkdir(parents=True)
    (data_dir / 'test.txt').write_text('\n'.join(NAMES) + '\n')
    rng = np.random.default_rng(0)
    for index, name in enumerate(NAMES):
        rgb = rng.integers(0, 256, size=(288, 512, 3), dtype=np.uint8)
        PIL.Image.fromarray(rgb, mode='RGB').save(data_dir / 'left' / f'left{name}.png')
        label = np.zeros((288, 512), dtype=np.uint8)
        label[:, 170:340] = 1
        label[:, 340:] = 2
        label[index * 30:(index + 1) * 30, :] = index % 3
        PIL.Image.fromarray(label, mode='L').save(data_dir / 'labels' / f'label{name}.png')
        depth = rng.integers(0, 256, size=(288, 512), dtype=np.uint8)
        PIL.Image.fromarray(depth, mode='L').save(data_dir / 'depth' / f'depth{name}.png')


def test_run_demo_rgb_depth_cpu(tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / 'dataset'
    _write_synthetic_dataset(data_dir)

    weights_root = tmp_path / 'weights'
    weight_dir = weights_root / WEIGHT_NAME
    weight_dir.mkdir(parents=True)
    torch.manual_seed(0)
    model = InconSeg(3, pretrained=False)
    torch.save(model.state_dict(), weight_dir / 'final.pth')
    del model
    gc.collect()

    torch_home = tmp_path / 'torch_home'
    torch_home.mkdir()
    monkeypatch.setenv('TORCH_HOME', str(torch_home))

    work = tmp_path / 'work'
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.syspath_prepend(ROOT)

    real_resnet152 = models.resnet152

    def resnet152_ignore_weights(*args, **kwargs):
        kwargs['weights'] = None
        return real_resnet152(*args, **kwargs)

    monkeypatch.setattr(models, 'resnet152', resnet152_ignore_weights)
    monkeypatch.setattr(sys, 'argv', [
        os.path.join(ROOT, 'run_demo_RGB_Depth.py'),
        '-g', '-1',
        '-j', '0',
        '-dr', str(data_dir),
        '-wd', str(weights_root),
        '-w', WEIGHT_NAME,
        '-f', 'final.pth',
        '-d', 'test',
        '-nc', '3',
        '-ih', '288',
        '-iw', '512',
    ])

    runpy.run_path(os.path.join(ROOT, 'run_demo_RGB_Depth.py'), run_name='__main__')

    captured = capsys.readouterr()
    assert f'testing InconSeg: {WEIGHT_NAME} on cpu with pytorch' in captured.out

    runs = work / 'runs'
    runs_mode = os.stat(runs).st_mode
    assert runs_mode & stat.S_IRWXU
    assert runs_mode & stat.S_IRWXO == stat.S_IRWXO
    for name in NAMES:
        prediction = runs / f'Pred_{WEIGHT_NAME}_{name}.png'
        assert prediction.is_file()
        assert prediction.stat().st_size > 0
    conf = runs / f'conf_{WEIGHT_NAME}.mat'
    assert conf.is_file()
    assert conf.stat().st_size > 0
