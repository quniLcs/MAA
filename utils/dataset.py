import json
import os
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# from utils.lazy_list import LazyList
# from utils.log import log, logger


class ImageText(Dataset):
    def __init__(
            self,
            dataset_name = 'context',
            dataset_split = 0,
            mode: str = 'train',
            transform: Optional[Callable] = None,
            max_text_len = 100,
        ):

        super(ImageText, self).__init__()

        self.dataset_name = dataset_name
        self.dataset_split = dataset_split

        assert mode in ('train', 'test')
        self.mode = mode

        annotation_path = os.path.join('datasets', self.dataset_name, 'split_%s.json' % dataset_split)
        with open(annotation_path, 'r') as file:
            annotation = json.load(file)
        self.annotation = annotation[mode]
        self.image_list = list(self.annotation.keys())

        text_path = os.path.join('datasets', self.dataset_name, 'text.json')
        with open(text_path, 'r') as file:
            self.text = json.load(file)

        self.transform = transform
        self.max_text_len = max_text_len

    def __getitem__(self, index: int):
        image_name = self.image_list[index]
        image_path = os.path.join('datasets', self.dataset_name, image_name)

        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        text = self.text[image_name]
        text_len = len(text)

        if not text:
            text = '[EMPTY]'
        elif text_len <= self.max_text_len:
            text = ' '.join(text)
        else:
            text_indexes = np.random.choice(text_len, self.max_text_len, replace = False)
            text_indexes.sort()
            text_chosen = [text[text_index] for text_index in text_indexes]
            text = ' '.join(text_chosen)

        target = self.annotation[image_name]
        target = int(target) - 1

        return image, text, target, image_name

    def __len__(self) -> int:
        return len(self.image_list)

    def extra_repr(self) -> str:
        return 'mode: %s' % self.mode
