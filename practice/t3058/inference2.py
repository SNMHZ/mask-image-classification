import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset
import albumentations as A
import numpy as np
from torch import nn


def load_model(saved_model, num_classes, device,i):
    model_cls = None
    model_cls = getattr(import_module("model"), f"{args.model}{i}")
    model = model_cls(
        num_classes=num_classes
    )
    model = nn.DataParallel(model)


    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    # model_path = os.path.join(saved_model, 'best.pth')
    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args,i):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = 18  # 18
    model = load_model(model_dir, num_classes, device,i)
    model.to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    all_predictions = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model(images) / 3
            pred += model(torch.flip(images, dims=(-1,))) / 3
            pred += model(torch.flip(images, dims= (-2,))) / 3
            all_predictions.extend(pred.cpu().numpy())

        fold_pred = np.array(all_predictions)
    return fold_pred / args.n_splits


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=20, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(512, 384), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='MyModel', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))
    parser.add_argument('--n_splits', default=3, help='Num_K for K-Fold')

    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    oof_pred = None

    # model_dir = f"{args.model_dir}2"
    # oof_pred = inference(data_dir, model_dir, output_dir, args,2)
    for i in range(args.n_splits):
        model_dir = f"{args.model_dir}{i}"
        if oof_pred is None:
            oof_pred = inference(data_dir, model_dir, output_dir, args,i)
        else:
            oof_pred += inference(data_dir, model_dir, output_dir, args,i)

    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)
    info['ans'] = np.argmax(oof_pred, axis = 1)
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
    print(f'Inference Done!')
