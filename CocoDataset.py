from pycocotools.coco import COCO
from PIL import Image
from torch.utils.data import Dataset
import os

class CocoDataset(Dataset):
    def __init__(self, root_dir, ann_file, transform=None, oversample='N'):
        self.root_dir = root_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.oversample = oversample
        self.data = []  # To store processed (img_path, label) pairs

        # Preprocess and load data to handle oversampling
        for img_id in self.ids:
            img_info = self.coco.imgs[img_id]
            img_path = os.path.join(self.root_dir, img_info['file_name'])
            if not img_path.endswith(('.jpg', '.png', '.jpeg')):
                continue  # Skip non-image files

            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            # Determine label
            label = 0
            for ann in anns:
                category_id = ann['category_id']
                if category_id != 0:  # Non-zero category ID indicates defect
                    label = 1
                    break

            self.data.append((img_path, label))

        # Perform oversampling if required
        if self.oversample == 'Y':
            self._perform_oversampling()

    def _perform_oversampling(self):
        # Calculate class distribution
        num_defects = sum(1 for _, label in self.data if label == 1)
        num_non_defects = len(self.data) - num_defects

        # Determine the minority class
        minority_class = 1 if num_defects < num_non_defects else 0

        # Collect minority class samples
        minority_data = [item for item in self.data if item[1] == minority_class]
        oversample_factor = max(1, len(self.data) // len(minority_data))

        # Add oversampled data
        oversampled_data = minority_data * (oversample_factor - 1)
        self.data.extend(oversampled_data)

    def __getitem__(self, index):
        img_path, label = self.data[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data)