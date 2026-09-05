import os
import cv2
import torch
import random

from torch.utils.data import Dataset, DataLoader


# ============================================================
# PATHS
# ============================================================

TIME1_PATH = r"C:\Users\ps502\Downloads\1\1\train\time1"
TIME2_PATH = r"C:\Users\ps502\Downloads\1\1\train\time2"
LABEL_PATH = r"C:\Users\ps502\Downloads\1\1\train\label"


# ============================================================
# DATASET
# ============================================================

class ChangeDetectionDataset(Dataset):

    def __init__(
        self,
        files,
        time1_path,
        time2_path,
        label_path,
        image_size=256,
        augment=False
    ):

        self.files = files
        self.time1_path = time1_path
        self.time2_path = time2_path
        self.label_path = label_path

        self.image_size = image_size
        self.augment = augment


    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):

        filename = self.files[index]

        # ----------------------------------------------------
        # Read images
        # ----------------------------------------------------

        img1 = cv2.imread(
            os.path.join(self.time1_path, filename)
        )

        img2 = cv2.imread(
            os.path.join(self.time2_path, filename)
        )

        label = cv2.imread(
            os.path.join(self.label_path, filename),
            cv2.IMREAD_GRAYSCALE
        )


        # ----------------------------------------------------
        # Convert BGR → RGB
        # ----------------------------------------------------

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)


        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        img1 = cv2.resize(
            img1,
            (self.image_size, self.image_size)
        )

        img2 = cv2.resize(
            img2,
            (self.image_size, self.image_size)
        )

        label = cv2.resize(
            label,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST
        )


        # ====================================================
        # DATA AUGMENTATION
        # ====================================================

        if self.augment:

            # Horizontal flip
            if random.random() < 0.5:

                img1 = cv2.flip(img1, 1)
                img2 = cv2.flip(img2, 1)
                label = cv2.flip(label, 1)


            # Vertical flip
            if random.random() < 0.5:

                img1 = cv2.flip(img1, 0)
                img2 = cv2.flip(img2, 0)
                label = cv2.flip(label, 0)


            # Rotate 90 degrees
            if random.random() < 0.5:

                img1 = cv2.rotate(
                    img1,
                    cv2.ROTATE_90_CLOCKWISE
                )

                img2 = cv2.rotate(
                    img2,
                    cv2.ROTATE_90_CLOCKWISE
                )

                label = cv2.rotate(
                    label,
                    cv2.ROTATE_90_CLOCKWISE
                )


        # ====================================================
        # NORMALIZATION
        # ====================================================

        img1 = img1.astype("float32") / 255.0
        img2 = img2.astype("float32") / 255.0


        # ====================================================
        # CONVERT TO TENSOR
        # ====================================================

        img1 = torch.tensor(
            img1,
            dtype=torch.float32
        )

        img2 = torch.tensor(
            img2,
            dtype=torch.float32
        )

        label = torch.tensor(
            label,
            dtype=torch.float32
        )


        # HWC → CHW
        img1 = img1.permute(2, 0, 1)
        img2 = img2.permute(2, 0, 1)


        # Binary mask
        label = (label > 0).float()


        return img1, img2, label


# ============================================================
# GET MATCHED FILES
# ============================================================

time1_files = set(os.listdir(TIME1_PATH))
time2_files = set(os.listdir(TIME2_PATH))
label_files = set(os.listdir(LABEL_PATH))

all_files = sorted(
    list(time1_files & time2_files & label_files)
)


print("Total matched images:", len(all_files))


# ============================================================
# SHUFFLE
# ============================================================

random.seed(42)

random.shuffle(all_files)


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

total = len(all_files)

train_end = int(0.70 * total)
valid_end = int(0.85 * total)

train_files = all_files[:train_end]

valid_files = all_files[
    train_end:valid_end
]

test_files = all_files[
    valid_end:
]


print("\nDataset split:")
print("Train:", len(train_files))
print("Validation:", len(valid_files))
print("Test:", len(test_files))


# ============================================================
# CREATE DATASETS
# ============================================================

train_dataset = ChangeDetectionDataset(
    train_files,
    TIME1_PATH,
    TIME2_PATH,
    LABEL_PATH,
    augment=True
)


valid_dataset = ChangeDetectionDataset(
    valid_files,
    TIME1_PATH,
    TIME2_PATH,
    LABEL_PATH,
    augment=False
)


test_dataset = ChangeDetectionDataset(
    test_files,
    TIME1_PATH,
    TIME2_PATH,
    LABEL_PATH,
    augment=False
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)


valid_loader = DataLoader(
    valid_dataset,
    batch_size=8,
    shuffle=False
)


test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":

    img1, img2, label = train_dataset[0]

    print("\nSample shapes:")
    print("Time1:", img1.shape)
    print("Time2:", img2.shape)
    print("Label:", label.shape)

    img1, img2, label = next(
        iter(train_loader)
    )

    print("\nTrain batch:")
    print("Time1:", img1.shape)
    print("Time2:", img2.shape)
    print("Label:", label.shape)