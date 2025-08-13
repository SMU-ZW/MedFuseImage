import os
import numpy as np
from PIL import Image
import pandas as pd 

import torch
from torch.utils.data import Dataset
# import 
import glob
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

# datasets/cxr_dataset.py

class MIMICCXR(Dataset):
    def __init__(self, paths, args, transform=None, split='train'):
        self.data_dir = args.cxr_data_dir
        self.args = args
        self.transform = transform

        self.filenames_to_path = {p.split('/')[-1].split('.')[0]: p for p in paths}

        labels_set = getattr(args, "labels_set", "radiology").lower()

        if labels_set in ("radiology"):
            metadata = pd.read_csv(f"{self.data_dir}/mimic-cxr-2.0.0-metadata.csv.gz")
            splits   = pd.read_csv(f"{self.data_dir}/mimic-cxr-2.0.0-split.csv.gz")
            labels   = pd.read_csv(f"{self.data_dir}/mimic-cxr-2.0.0-chexpert.csv.gz")

            self.CLASSES = [
                'Atelectasis','Cardiomegaly','Consolidation','Edema',
                'Enlarged Cardiomediastinum','Fracture','Lung Lesion',
                'Lung Opacity','No Finding','Pleural Effusion','Pleural Other',
                'Pneumonia','Pneumothorax','Support Devices'
            ]
            labels[self.CLASSES] = labels[self.CLASSES].fillna(0).replace(-1.0, 0.0)

            split_ids = splits.loc[splits['split'] == split, 'dicom_id'].astype(str).tolist()

            meta_lab = metadata.merge(labels[self.CLASSES + ['study_id']], on='study_id', how='inner')
            meta_lab['dicom_id'] = meta_lab['dicom_id'].astype(str)
            labels_map = dict(zip(meta_lab['dicom_id'].values, meta_lab[self.CLASSES].values))
            self.num_classes = 14

        elif labels_set in ("mortality"):
            metadata = pd.read_csv(f"../split_cxr_mortality_medFuse.csv")
            splits   = pd.read_csv(f"../split_cxr_mortality_medFuse.csv")

            mortality_csv_path = "../split_cxr_mortality_medFuse.csv"
            mort = pd.read_csv(mortality_csv_path)
            if "subject_id" not in mort.columns or "Mortality" not in mort.columns:
                raise ValueError("Mortality CSV must contain: subject_id, Mortality")

            mort["subject_id"] = mort["subject_id"].astype(str)
            mort["Mortality"]  = mort["Mortality"].astype(float)
            mort = mort.groupby("subject_id", as_index=False)["Mortality"].max()

            split_ids = splits.loc[splits['split'] == split, 'dicom_id'].astype(str).tolist()

            meta_sid = metadata[["dicom_id", "subject_id"]].copy()
            meta_sid["dicom_id"]   = meta_sid["dicom_id"].astype(str)
            meta_sid["subject_id"] = meta_sid["subject_id"].astype(str)

            meta_mort = meta_sid.merge(mort[["subject_id", "Mortality"]], on="subject_id", how="left")
            meta_mort["Mortality"] = meta_mort["Mortality"].fillna(0.0)

            labels_map = dict(zip(meta_mort["dicom_id"].values, meta_mort["Mortality"].values))
            self.CLASSES = ["mortality"]
            self.num_classes = 1
            
        elif labels_set in ("readmission"):
            metadata = pd.read_csv(f"../split_cxr_readmission_medFuse.csv")
            splits   = pd.read_csv(f"../split_cxr_readmission_medFuse.csv")

            readmission_csv_path = "../split_cxr_readmission_medFuse.csv"
            read = pd.read_csv(readmission_csv_path)
            if "subject_id" not in read.columns or "Readmission" not in read.columns:
                raise ValueError("Readmission CSV must contain: subject_id, Readmission")

            read["subject_id"] = read["subject_id"].astype(str)
            read["Readmission"]  = read["Readmission"].astype(float)
            read = read.groupby("subject_id", as_index=False)["Readmission"].max()

            split_ids = splits.loc[splits['split'] == split, 'dicom_id'].astype(str).tolist()

            meta_sid = metadata[["dicom_id", "subject_id"]].copy()
            meta_sid["dicom_id"]   = meta_sid["dicom_id"].astype(str)
            meta_sid["subject_id"] = meta_sid["subject_id"].astype(str)

            meta_read = meta_sid.merge(read[["subject_id", "Readmission"]], on="subject_id", how="left")
            meta_read["Readmission"] = meta_read["Readmission"].fillna(0.0)

            labels_map = dict(zip(meta_read["dicom_id"].values, meta_read["Readmission"].values))
            self.CLASSES = ["readmission"]
            self.num_classes = 1


        else:
            raise ValueError(f"Unknown labels_set: {labels_set}")

        valid_ids = []
        for did in tqdm(split_ids, desc=f"Filtering {split}"):
            if did in labels_map and did in self.filenames_to_path:
                valid_ids.append(did)

        self._id_list = sorted(valid_ids)
        self.filesnames_to_labels = labels_map

    def __len__(self):
        return len(self._id_list)

    def __getitem__(self, index):
        dicom_id = index if isinstance(index, str) else self._id_list[index]

        img_path = self.filenames_to_path[dicom_id]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        y = self.filesnames_to_labels[dicom_id]
        if isinstance(y, (list, tuple, np.ndarray)):
            y = torch.tensor(y, dtype=torch.float32)
        else:
            y = torch.tensor([float(y)], dtype=torch.float32)

        return img, y      


def get_transforms(args):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transforms = []
    train_transforms.append(transforms.Resize(256))
    train_transforms.append(transforms.RandomHorizontalFlip())
    train_transforms.append(transforms.RandomAffine(degrees=45, scale=(.85, 1.15), shear=0, translate=(0.15, 0.15)))
    train_transforms.append(transforms.CenterCrop(224))
    train_transforms.append(transforms.ToTensor())
    train_transforms.append(normalize)      


    test_transforms = []
    test_transforms.append(transforms.Resize(args.resize))


    test_transforms.append(transforms.CenterCrop(args.crop))

    test_transforms.append(transforms.ToTensor())
    test_transforms.append(normalize)


    return train_transforms, test_transforms

import itertools

def get_cxr_datasets(args):
    train_transforms, test_transforms = get_transforms(args)
    data_dir = args.cxr_data_dir
    
    paths = list(tqdm(
        glob.iglob(f'{data_dir}/resized/**/*.jpg', recursive=True) ,
        desc="Finding CXR images"
    ))
    dataset_train = MIMICCXR(paths, args, split='train', transform=transforms.Compose(train_transforms))
    dataset_validate = MIMICCXR(paths, args, split='validate', transform=transforms.Compose(test_transforms),)
    dataset_test = MIMICCXR(paths, args, split='test', transform=transforms.Compose(test_transforms),)
    
    return dataset_train, dataset_validate, dataset_test
