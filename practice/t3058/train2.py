import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset, MaskSplitByProfileDataset
from loss import create_criterion
from PIL import Image
import albumentations as A

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = np.ceil(n ** 0.5)
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

def getDataloader(dataset, use_cuda, train_set,val_set):

    # -- augmentation
    # dataset.set_resize(args.resize)
    # transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation

    # -- data_loader


    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )
    return train_loader, val_loader

    
def f1score(y_pred,y_true):
    import torch.nn.functional as F
    epsilon = 1e-7
    y_true = F.one_hot(y_true, 18).to(torch.float32)
    y_pred = F.softmax(y_pred, dim=1)

    tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
    # tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1 = f1.clamp(min=epsilon, max=1 - epsilon)
    return f1

def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    # save_dir = model_dir
    
    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: TrainDataset
    dataset = dataset_module(data_dir,)
    # num_classes = dataset.num_classes  # 18

    best_val_acc = [0,0,0]
    best_val_loss = np.inf
    
    for i in range(args.n_splits):
        save_dir = increment_path(f"{model_dir}/{args.name}{i}")
    
        # -- model
        model = [3,2,3]
        model_module = getattr(import_module("model"), args.model)  # default: MyModel
        optimizer = [0,0,0]
        criterion = [0,0,0]
        for j in range(3):
            model[j] = model_module(
                num_classes=model[j]
            ).to(device)
            model[j] = torch.nn.DataParallel(model[j])

        # -- loss & metric
            criterion[j] = create_criterion(args.criterion)  # default: cross_entropy
            opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: Adam
            optimizer[j] = opt_module(
                filter(lambda p: p.requires_grad, model[j].parameters()),
                lr=args.lr,
                weight_decay=1e-3
            )
            scheduler = StepLR(optimizer[j], args.lr_decay_step, gamma=0.5)

        # -- logging
        logger = SummaryWriter(log_dir=f"{save_dir}")
        with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)



        for epoch in range(args.epochs):
            train_set, val_set = dataset.split_dataset(epoch)
            train_loader, val_loader = getDataloader(dataset, use_cuda, train_set,val_set)
            # train loop
            model[0].train()
            model[1].train()
            model[2].train()
            loss_value0 = 0
            loss_value1 = 0
            loss_value2 = 0
            matches = 0
            for idx, train_batch in enumerate(train_loader):    #get_item 을 배치사이즈만큼 부름
                inputs, labels = train_batch
                inputs = inputs.to(device)
                mask_labels, gender_labels, age_labels = labels
                mask_labels = mask_labels.to(device)
                gender_labels = gender_labels.to(device)
                age_labels = age_labels.to(device)

                # outs = model(inputs)
                mask_out = model[0](inputs)
                gender_out = model[1](inputs)
                age_out = model[2](inputs)
                # outs = 6*mask_out + 3*gender_out + age_out
                # preds = torch.argmax(outs, dim=-1)
                preds = 6*torch.argmax(mask_out, dim=-1) + 3*torch.argmax(gender_out, dim=-1) + torch.argmax(age_out, dim=-1)
                loss0 = criterion[0](mask_out, mask_labels)
                loss1 = criterion[1](gender_out, gender_labels)
                loss2 = criterion[2](age_out, age_labels)

                loss0.backward()
                loss1.backward()
                loss2.backward()
                
                
                if idx%2==0:
                    for j in range(3):
                        optimizer[j].step()
                        optimizer[j].zero_grad()


                loss_value0 += loss0.item()
                loss_value1 += loss1.item()
                loss_value2 += loss2.item()
                matches += (preds == 6*mask_labels+3*gender_labels+age_labels).sum().item()
                if (idx + 1) % args.log_interval == 0:
                    train_loss0 = loss_value0 / args.log_interval
                    train_loss1 = loss_value1 / args.log_interval
                    train_loss2 = loss_value2 / args.log_interval
                    train_acc = matches / args.batch_size / args.log_interval
                    current_lr = get_lr(optimizer[0])
                    print(
                        f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss0 {train_loss0:4.4} || training loss1 {train_loss1:4.4} || training loss2 {train_loss2:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr} "
                    )
                    logger.add_scalar("Train/loss0", train_loss0, epoch * len(train_loader) + idx)
                    logger.add_scalar("Train/loss1", train_loss1, epoch * len(train_loader) + idx)
                    logger.add_scalar("Train/loss2", train_loss2, epoch * len(train_loader) + idx)
                    logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)
                    logger.add_scalar("Train/lr", current_lr, epoch * len(train_loader) + idx)

                    loss_value0 = 0
                    loss_value1 = 0
                    loss_value2 = 0
                    matches = 0

            scheduler.step()

            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model[0].eval()
                model[1].eval()
                model[2].eval()
                val_loss_items = [[],[],[]]
                val_acc_items = []
                figure = None
                for val_batch in val_loader:
                    inputs, labels = val_batch
                    inputs = inputs.to(device)
                    mask_labels, gender_labels, age_labels = labels
                    mask_labels = mask_labels.to(device)
                    gender_labels = gender_labels.to(device)
                    age_labels = age_labels.to(device)

                    # outs = model(inputs)
                    mask_out = model[0](inputs)
                    gender_out = model[1](inputs)
                    age_out = model[2](inputs)
                    lbl = 6*mask_labels+3*gender_labels+age_labels
                    preds = 6*torch.argmax(mask_out, dim=-1) + 3*torch.argmax(gender_out, dim=-1) + torch.argmax(age_out, dim=-1)   

                    
                    outt =[mask_out,gender_out,age_out]
                    labell = [mask_labels,gender_labels,age_labels]
                    acc_item = (preds == lbl).sum().item()
                    for k in range(3):
                        val_loss_items[k].append(criterion[k](outt[k], labell[k]).item())
                    val_acc_items.append(acc_item)

                    if figure is None:
                        inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                        inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                        figure = grid_image(
                            inputs_np, lbl, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                        )
                for k in range(3):
                    val_loss = np.sum(val_loss_items[k]) / len(val_loader)
                    val_acc = np.sum(val_acc_items) / len(val_set)
                    best_val_loss = min(best_val_loss, val_loss)
                    if val_acc > best_val_acc[k]:
                        print(f"New best model for val accuracy{k} : {val_acc:4.2%}! saving the best model..")
                        torch.save(model[k].state_dict(), f"{save_dir}/best{k}.pth")
                        best_val_acc[k] = val_acc
                    torch.save(model[k].state_dict(), f"{save_dir}/last{k}.pth")
                    print(

                        f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                        f"best acc : {best_val_acc[k]:4.2%}, best loss: {best_val_loss:4.2} "
                        # f"f1 {f1score(outs,labels)}"
                    )
                    
                    logger.add_scalar(f"Val/loss{k}", val_loss, epoch)
                    logger.add_scalar(f"Val/accuracy{k}", val_acc, epoch)
                logger.add_figure("results", figure, epoch)
                print()

        logger.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    import os

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=4, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskSplitByProfileDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[228, 228 ], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=20, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=20, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='exp12_resnet101Adamax', help='model type (default: MyModel)')
    parser.add_argument('--optimizer', type=str, default='Adamax', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='label_smoothing', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=1, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--n_splits', default=1, help='Num_K for K-Fold')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)



    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)