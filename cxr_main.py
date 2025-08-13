from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import argparse
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from datasets.cxr_dataset import get_cxr_datasets
from trainers.cxr_trainer import CXRTrainer  
from arguments import args_parser

parser = args_parser()
args = parser.parse_args()
print(args)

if args.missing_token is not None:
    from trainers.fusion_tokens_trainer import FusionTokensTrainer as FusionTrainer
    
path = Path(args.save_dir)
path.mkdir(parents=True, exist_ok=True)

seed = 1002
torch.manual_seed(seed)
np.random.seed(seed)

print("==> Loading CXR data")
cxr_train_ds, cxr_val_ds, cxr_test_ds = get_cxr_datasets(args)

train_dl = DataLoader(cxr_train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_dl   = DataLoader(cxr_val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
test_dl  = DataLoader(cxr_test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

with open(f"{args.save_dir}/args.txt", 'w') as results_file:
    for arg in vars(args): 
        print(f"  {arg:<40}: {getattr(args, arg)}")
        results_file.write(f"  {arg:<40}: {getattr(args, arg)}\n")

print("==> Start creating model")
trainer = CXRTrainer(train_dl, val_dl, args, test_dl=test_dl)

if args.mode == 'train':
    print("==> training CXR only")
    trainer.train()
elif args.mode == 'eval':
    print("==> evaluating CXR only")
    trainer.eval()
else:
    raise ValueError("Not implemented for args.mode")
