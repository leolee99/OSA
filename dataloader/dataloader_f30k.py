import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class F30K_Dataset(Dataset):
    def __init__(self, args, image_root, annFile_root, preprocess, tokenizer, ids=None, subset='train', logger=None):
        logger.info("========== Initial the %s set ==========", subset)
        self.args = args
        self.image_root = image_root
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.subset = subset
        self.num_anns = 1
        self.images_id, self.captions = [], []
        self.noise_ratio = args.noise_ratio
        self.clean_ids = np.load(os.path.join(annFile_root, '{}_clean_index.npy'.format(self.noise_ratio)))

        if subset == 'train':
            self.num_anns = 5

        if subset == 'train':
            # get noisy annotation captions
            for line in open(os.path.join(annFile_root, '{}_noise_train_caps.txt'.format(self.noise_ratio)), 'r', encoding='utf-8'):
                self.captions.append(line.strip())

            if args.train_clean:
                self.captions = [self.captions[int(w)] for w in self.clean_ids]

        else:
            self.noise_ratio = 0.0
            for line in open(os.path.join(annFile_root, '%s_caps.txt' % subset), 'r', encoding='utf-8'):
                self.captions.append(line.strip())

        for line in open(os.path.join(annFile_root, '%s_ids.txt' % subset), 'r', encoding='utf-8'):
            for i in range(self.num_anns):
                self.images_id.append(int(line))

        if subset == 'train':
            if args.train_clean:
                self.images_id = [self.images_id[int(w)] for w in self.clean_ids]

        # get images' name list
        image_name = []
        for line in open(os.path.join(annFile_root, 'image_name.txt'), 'r', encoding='utf-8'):
            image_name.append(line.strip())
        self.image_name = [image_name[int(id)] for id in self.images_id]

        self.texts  = self.tokenizer(self.captions, truncate=True)
        self.img_length = len(set(self.images_id))
        self.txt_length = len(self.captions)
        logger.info('%d images have been loaded.', self.img_length)
        logger.info('%d captions have been loaded.', self.txt_length)
        logger.info("%s set initialization completed!", subset)

    def __len__(self):
        return self.txt_length

    def __getitem__(self, idx):
        image = self.preprocess(Image.open(os.path.join(self.image_root, self.image_name[idx]))) # Image from PIL module
        text = self.texts[idx]
        img_id = self.images_id[idx]


        return image, text, img_id
