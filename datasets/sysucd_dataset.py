import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from .transforms import get_transforms


class SYSUCDDataset(Dataset):
    """
    Dataset loader for SYSU-CD (Change Detection due to human activities).
    
    Supports:
    1. Subfolder splits:
       dataset_root/
         train/ (or val/ or test/)
           time1/ (or A/ or T1/)
           time2/ (or B/ or T2/)
           label/ (or mask/)
           
    2. Split list files:
       dataset_root/
         time1/, time2/, label/
         list/train.txt (or train.txt in root)
         list/val.txt
         list/test.txt
         
    3. Automatic fallback split if only flat folders exist.
    """
    def __init__(self, root_dir, split='train', img_size=(256, 256), transform=None):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.transform = transform if transform is not None else get_transforms(mode=split, img_size=img_size)

        self.samples = self._find_samples()
        if len(self.samples) == 0:
            raise RuntimeError(f"No image pairs found for split '{split}' in root directory: {root_dir}")

    def _find_dir_with_aliases(self, base_path, candidate_names):
        """Finds existing directory among potential alias names."""
        for name in candidate_names:
            p = os.path.join(base_path, name)
            if os.path.isdir(p):
                return p
        return None

    def _find_samples(self):
        samples = []
        
        # Strategy A: Check subfolder structure: root_dir / split / (time1, time2, label)
        split_dir = os.path.join(self.root_dir, self.split)
        if os.path.isdir(split_dir):
            t1_dir = self._find_dir_with_aliases(split_dir, ['time1', 'time_1', 'T1', 't1', 'A', 'im1', 'image1'])
            t2_dir = self._find_dir_with_aliases(split_dir, ['time2', 'time_2', 'T2', 't2', 'B', 'im2', 'image2'])
            lbl_dir = self._find_dir_with_aliases(split_dir, ['label', 'labels', 'mask', 'masks', 'label1', 'GT', 'gt'])

            if t1_dir and t2_dir and lbl_dir:
                exts = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp')
                t1_files = []
                for ext in exts:
                    t1_files.extend(glob.glob(os.path.join(t1_dir, ext)))
                
                for f1 in sorted(t1_files):
                    basename = os.path.basename(f1)
                    f2 = os.path.join(t2_dir, basename)
                    flbl = os.path.join(lbl_dir, basename)

                    # Also try matching if extensions differ (e.g. .png vs .jpg)
                    if not os.path.exists(f2):
                        name_no_ext = os.path.splitext(basename)[0]
                        matches_f2 = glob.glob(os.path.join(t2_dir, f"{name_no_ext}.*"))
                        if matches_f2:
                            f2 = matches_f2[0]

                    if not os.path.exists(flbl):
                        name_no_ext = os.path.splitext(basename)[0]
                        matches_flbl = glob.glob(os.path.join(lbl_dir, f"{name_no_ext}.*"))
                        if matches_flbl:
                            flbl = matches_flbl[0]

                    if os.path.exists(f2) and os.path.exists(flbl):
                        samples.append({
                            't1': f1,
                            't2': f2,
                            'mask': flbl,
                            'name': os.path.splitext(basename)[0]
                        })
                return samples

        # Strategy B: Check flat folders + list txt file
        t1_dir = self._find_dir_with_aliases(self.root_dir, ['time1', 'time_1', 'T1', 't1', 'A', 'im1', 'image1'])
        t2_dir = self._find_dir_with_aliases(self.root_dir, ['time2', 'time_2', 'T2', 't2', 'B', 'im2', 'image2'])
        lbl_dir = self._find_dir_with_aliases(self.root_dir, ['label', 'labels', 'mask', 'masks', 'label1', 'GT', 'gt'])

        if t1_dir and t2_dir and lbl_dir:
            # Check for split list files
            list_candidates = [
                os.path.join(self.root_dir, 'list', f"{self.split}.txt"),
                os.path.join(self.root_dir, 'list_dir', f"{self.split}.txt"),
                os.path.join(self.root_dir, f"{self.split}.txt"),
                os.path.join(self.root_dir, f"{self.split}_list.txt"),
            ]
            list_file = None
            for lc in list_candidates:
                if os.path.isfile(lc):
                    list_file = lc
                    break

            if list_file:
                with open(list_file, 'r') as f:
                    lines = [line.strip() for line in f if line.strip()]

                for line in lines:
                    # Handle possible space-separated entries: t1_path t2_path mask_path or single filename
                    parts = line.split()
                    if len(parts) >= 3:
                        f1, f2, flbl = parts[0], parts[1], parts[2]
                        f1 = f1 if os.path.isabs(f1) else os.path.join(self.root_dir, f1)
                        f2 = f2 if os.path.isabs(f2) else os.path.join(self.root_dir, f2)
                        flbl = flbl if os.path.isabs(flbl) else os.path.join(self.root_dir, flbl)
                        bname = os.path.splitext(os.path.basename(f1))[0]
                    else:
                        bname = os.path.splitext(parts[0])[0]
                        f1 = os.path.join(t1_dir, parts[0]) if not os.path.exists(parts[0]) else parts[0]
                        f2 = os.path.join(t2_dir, parts[0]) if not os.path.exists(parts[0]) else parts[0]
                        flbl = os.path.join(lbl_dir, parts[0]) if not os.path.exists(parts[0]) else parts[0]

                        # Suffix fallback if direct match fails
                        if not os.path.exists(f1):
                            m = glob.glob(os.path.join(t1_dir, f"{bname}.*"))
                            if m: f1 = m[0]
                        if not os.path.exists(f2):
                            m = glob.glob(os.path.join(t2_dir, f"{bname}.*"))
                            if m: f2 = m[0]
                        if not os.path.exists(flbl):
                            m = glob.glob(os.path.join(lbl_dir, f"{bname}.*"))
                            if m: flbl = m[0]

                    if os.path.exists(f1) and os.path.exists(f2) and os.path.exists(flbl):
                        samples.append({'t1': f1, 't2': f2, 'mask': flbl, 'name': bname})
                return samples

            # Strategy C: Flat directory automatic 80/10/10 split
            exts = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp')
            t1_files = []
            for ext in exts:
                t1_files.extend(glob.glob(os.path.join(t1_dir, ext)))
            t1_files = sorted(t1_files)

            all_valid = []
            for f1 in t1_files:
                basename = os.path.basename(f1)
                bname = os.path.splitext(basename)[0]
                m2 = glob.glob(os.path.join(t2_dir, f"{bname}.*"))
                mlbl = glob.glob(os.path.join(lbl_dir, f"{bname}.*"))
                if m2 and mlbl:
                    all_valid.append({'t1': f1, 't2': m2[0], 'mask': mlbl[0], 'name': bname})

            # Deterministic split (80% train, 10% val, 10% test)
            n = len(all_valid)
            n_train = int(0.8 * n)
            n_val = int(0.1 * n)

            if self.split == 'train':
                return all_valid[:n_train]
            elif self.split in ('val', 'valid', 'validation'):
                return all_valid[n_train:n_train + n_val]
            elif self.split in ('test', 'eval'):
                return all_valid[n_train + n_val:]

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        t1_img = Image.open(item['t1']).convert('RGB')
        t2_img = Image.open(item['t2']).convert('RGB')
        mask_img = Image.open(item['mask']).convert('L')

        if self.transform is not None:
            t1_tensor, t2_tensor, mask_tensor = self.transform(t1_img, t2_img, mask_img)
        else:
            t1_tensor, t2_tensor, mask_tensor = t1_img, t2_img, mask_img

        return {
            't1': t1_tensor,
            't2': t2_tensor,
            'mask': mask_tensor,
            'name': item['name']
        }


def build_dataloader(root_dir, split='train', batch_size=16, num_workers=4, shuffle=True, img_size=(256, 256)):
    """Convenience helper to construct PyTorch DataLoader."""
    dataset = SYSUCDDataset(root_dir=root_dir, split=split, img_size=img_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == 'train')
    )
    return loader, dataset
