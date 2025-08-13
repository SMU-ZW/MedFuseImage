# trainers/cxr_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from .trainer import Trainer  # 复用你原来的基类
from models.cxr_models import CXRModels
from tqdm import tqdm

class CXRTrainer(Trainer):
    def __init__(self, train_dl, val_dl, args, test_dl=None):
        super().__init__(args)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(torch.cuda.is_available())  # True 表示有 GPU
        # print(torch.cuda.device_count())  # 显示 GPU 数量
        # print(torch.cuda.current_device())  # 当前使用的 GPU ID
        # print(torch.cuda.get_device_name(0))  # 第0块GPU的名字
        self.args = args
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.epoch = 0

        # 只保留 CXR 模型
        self.model = CXRModels(self.args, self.device).to(self.device)
        # self.model.cxr_model = self.model  # 兼容可能存在的旧代码调用

        self.loss = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), args.lr, betas=(0.9, self.args.beta_1))
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=10, mode='min')

        self.best_auroc = 0.0
        self.best_stats = None
        self.epochs_stats = {'loss train': [], 'loss val': [], 'auroc val': []}

        if getattr(self.args, "pretrained", False) and getattr(self.args, "load_state", None):
            self.load_cxr_state(self.args.load_state)

    def _forward_loss(self, img, y):
        # CXRModels 会返回 (preds, bce_loss, feats)
        preds, loss_bce, _ = self.model(img, y)
        return preds, loss_bce

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)

        for i, (img, y) in enumerate(tqdm(self.train_dl, desc="Training", leave=True)):
            img = img.to(self.device)
            y = y.to(self.device).float()

            self.optimizer.zero_grad()
            preds, loss = self._forward_loss(img, y)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            outPRED = torch.cat((outPRED, preds.detach()), dim=0)
            outGT   = torch.cat((outGT, y.detach()), dim=0)

        self.epochs_stats['loss train'].append(epoch_loss / max(1, i+1))
        return self.computeAUROC(outGT.cpu().numpy(), outPRED.cpu().numpy(), 'train')

    def validate(self, dl):
        self.model.eval()
        epoch_loss = 0.0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)

        with torch.no_grad():
            # for i, (img, y) in enumerate(dl):
            for i, (img, y) in enumerate(tqdm(dl, desc="Validating", leave=True)):
                img = img.to(self.device)
                y = y.to(self.device).float()
                preds, loss = self._forward_loss(img, y)

                epoch_loss += loss.item()
                outPRED = torch.cat((outPRED, preds), dim=0)
                outGT   = torch.cat((outGT, y), dim=0)

        avg_loss = epoch_loss / max(1, i+1)
        self.scheduler.step(avg_loss)
        self.epochs_stats['loss val'].append(avg_loss)

        ret = self.computeAUROC(outGT.cpu().numpy(), outPRED.cpu().numpy(), 'validation')
        self.epochs_stats['auroc val'].append(ret['auroc_mean'])
        return ret
      
      
#     def validate(self, dl):
#         self.model.eval()
#         epoch_loss = 0.0
#         outGT = torch.FloatTensor().to(self.device)
#         outPRED = torch.FloatTensor().to(self.device)

#         batch_count = 0
#         with torch.no_grad():
#             for img, y in dl:
#                 batch_count += 1
#                 img = img.to(self.device)
#                 y = y.to(self.device).float()
#                 preds, loss = self._forward_loss(img, y)

#                 epoch_loss += loss.item()
#                 outPRED = torch.cat((outPRED, preds), dim=0)
#                 outGT   = torch.cat((outGT, y),    dim=0)

#         if batch_count == 0:
#             print("[validate] WARNING: validation loader is empty.")
#             # 防止调度器报错：给个大的loss，让LR别乱动，或干脆不step
#             # self.scheduler.step(float("inf"))
#             ret = {"auroc_mean": float("nan"), "auprc_mean": float("nan")}
#             self.epochs_stats['loss val'].append(float("nan"))
#             self.epochs_stats['auroc val'].append(float("nan"))
#             return ret

#         avg_loss = epoch_loss / batch_count
#         self.scheduler.step(avg_loss)
#         self.epochs_stats['loss val'].append(avg_loss)

#         ret = self.computeAUROC(outGT.cpu().numpy(), outPRED.cpu().numpy(), 'validation')
#         self.epochs_stats['auroc val'].append(ret.get('auroc_mean', float("nan")))
#         return ret


    def train(self):
        for self.epoch in range(self.start_epoch, self.args.epochs):
            # 先训练再验证
            _ = self.train_epoch()
            ret = self.validate(self.val_dl)
            self.save_checkpoint(prefix='last')

            if ret['auroc_mean'] > self.best_auroc:
                self.best_auroc = ret['auroc_mean']
                self.best_stats = ret
                self.save_checkpoint()  # best
                self.print_and_write(ret, isbest=True)
                self.patience = 0
            else:
                self.print_and_write(ret, isbest=False)
                self.patience += 1
                if self.patience >= self.args.patience:
                    break

        if self.best_stats is not None:
            self.print_and_write(self.best_stats, isbest=True)

    def eval(self):
        self.model.eval()
        ret = self.validate(self.test_dl)
        self.print_and_write(ret, isbest=True, prefix='cxr test', filename='results_test.txt')
        return
