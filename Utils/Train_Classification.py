import copy
import os
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import torch
import time
import torchmetrics.functional as metrics
from torch.utils.tensorboard import SummaryWriter

from Utils.Utils import adjust_learning_rate


def Train_Classification(model, train_loader, valid_loader, epochs, num_classes, runs='./EXP/', EXP='MY_EXP', criterion=None, optimizer=None,
                         lr_schedule=None, wd_schedule=None, class_distribute=None, device='cuda', if_ema=False, mixup_fn=None, recoverFromNEpoch=0):
    performance = torch.zeros(epochs, 3)
    model = model.to(device)
    if criterion is None:
        criterion = nn.CrossEntropyLoss(weight=class_distribute)
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters())

    logger = SummaryWriter(runs)
    scaler = GradScaler()
    for epoch in range(recoverFromNEpoch, epochs):
        model.train()
        preds, labels, start, loss = [], [], time.time(), 0
        pbar = tqdm(train_loader)
        for it, (batch, label) in enumerate(pbar):
            with autocast():
                if lr_schedule is not None:
                    it_global = len(train_loader) * epoch + it  # global training iteration
                    for i, param_group in enumerate(optimizer.param_groups):
                        param_group["lr"] = lr_schedule[it_global]
                        if wd_schedule is not None:
                            if i == 0:  # only the first group is regularized
                                param_group["weight_decay"] = wd_schedule[it_global]
                else:
                    # not being able to adjust WD
                    adjust_learning_rate(optimizer, it / len(train_loader) + epoch, args)
                batch, label = batch.to(device, non_blocking=True), label.to(device, non_blocking=True)
                if mixup_fn is not None:
                    label_ori = copy.deepcopy(label)
                    labels.append(label_ori)
                    batch, label = mixup_fn(batch, label)
                if if_ema:
                    with torch.no_grad():
                        model.update_ema()
                out = model(batch)
                loss_batch = criterion(out, label)
                preds.append(torch.argmax(out, dim=1))
                if mixup_fn is None:
                    labels.append(label)
                loss += loss_batch.item()

                optimizer.zero_grad()
                scaler.scale(loss_batch).backward()
                scaler.step(optimizer)
                scaler.update()

                pbar.set_postfix(Cross_Entropy=loss_batch.item())
        preds = torch.Tensor.cpu(torch.cat(preds))
        labels = torch.Tensor.cpu(torch.cat(labels))

        OA = metrics.accuracy(preds=preds, target=labels, num_classes=num_classes)
        AA = metrics.recall(preds=preds, target=labels, average='macro', num_classes=num_classes)
        kappa = metrics.cohen_kappa(preds=preds, target=labels, num_classes=num_classes)
        print('epoch:   ', epoch)
        print('train loss:{:.4f}    OA:{:.4f}      AA:{:.4f}    kappa:{:.4f}'.format(loss, OA, AA, kappa))
        logger.add_scalar("Train_Loss", loss, global_step=epoch)
        logger.add_scalar("Train_Acc", OA, global_step=epoch)
        performance = Evaluate(model=model, valid_loader=valid_loader, num_classes=num_classes, logger=logger, epoch=epoch, performance=performance, if_ema=if_ema, EXP=EXP)
    return torch.max(performance[:, 0])


def Evaluate(model, valid_loader, num_classes, logger, epoch, performance, EXP, if_ema=False, device='cuda'):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        preds, preds_ema, labels, labels_ema, start, loss, loss_ema = [], [], [], [], time.time(), 0, 0

        pbar = tqdm(valid_loader)
        for batch, label in pbar:
            batch, label = batch.to(device=device, non_blocking=True), label.to(device=device, non_blocking=True)
            out = model(batch)

            loss_batch = criterion(out, label)
            loss += loss_batch.item()
            preds.append(torch.argmax(out, dim=1))
            labels.append(label)

            if if_ema:
                out_ema = model.forward_ema(batch)
                loss_batch_ema = criterion(out_ema, label)
                loss_ema += loss_batch_ema.item()
                preds_ema.append(torch.argmax(out_ema, dim=1))
                labels_ema.append(label)

            pbar.set_postfix(Cross_Entropy=loss_batch.item())
        preds = torch.Tensor.cpu(torch.cat(preds))
        labels = torch.Tensor.cpu(torch.cat(labels))
        OA = metrics.accuracy(preds=preds, target=labels, num_classes=num_classes)
        AA = metrics.recall(preds=preds, target=labels, average='macro', num_classes=num_classes)
        kappa = metrics.cohen_kappa(preds=preds, target=labels, num_classes=num_classes, task="multiclass")
        performance[epoch, 0], performance[epoch, 1], performance[epoch, 2] = OA, AA, kappa
        best = (OA + AA + kappa)/3
        if if_ema:
            preds_ema = torch.Tensor.cpu(torch.cat(preds_ema))
            labels_ema = torch.Tensor.cpu(torch.cat(labels_ema))
            OA_ema = metrics.accuracy(preds=preds_ema, target=labels_ema, num_classes=num_classes)
            AA_ema = metrics.recall(preds=preds_ema, target=labels_ema, average='macro', num_classes=num_classes)
            kappa_ema = metrics.cohen_kappa(preds=preds_ema, target=labels_ema, num_classes=num_classes)
            print('valid loss ema:{:.4f}    OA ema:{:.4f}      AA ema:{:.4f}    kappa ema:{:.4f}'.format(loss_ema, OA_ema, AA_ema,
                                                                                                         kappa_ema))
            best = max((OA + AA + kappa)/3, (OA_ema + AA_ema + kappa_ema)/3)
            if (OA_ema + AA_ema + kappa_ema)/3 > (OA + AA + kappa)/3:
                performance[epoch, 0], performance[epoch, 1], performance[epoch, 2] = OA_ema, AA_ema, kappa_ema
            logger.add_scalar("Valid_Loss_ema", loss_ema, global_step=epoch)
            logger.add_scalar("Valid_Acc_ema", OA_ema, global_step=epoch)

        if best >= performance.mean(dim=1).max():
            os.makedirs('CheckPoints', exist_ok=True)
            torch.save(model, 'CheckPoints/' + EXP + '.pth')
            print('*' * 5, '   saving!')
        print('valid loss:{:.4f}    OA:{:.4f}      AA:{:.4f}    kappa:{:.4f}'.format(loss, OA, AA, kappa) + '\n')
        logger.add_scalar("Valid_Loss", loss, global_step=epoch)
        logger.add_scalar("Valid_Acc", OA, global_step=epoch)
    return performance
