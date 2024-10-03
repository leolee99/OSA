import os
from torch.utils.data import Dataset
from PIL import Image


class CC_Dataset(Dataset):
    def __init__(self, args, image_root, annFile_root, preprocess, tokenizer, ids=None, subset='train', logger=None):
        logger.info("========== Initial the %s set ==========", subset)
        self.args = args
        self.image_root = image_root
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.subset = subset
        self.num_anns = 1
        self.images_id, self.captions = [], []

        for line in open(os.path.join(annFile_root, '%s_caps.txt' % subset), 'r', encoding='utf-8'):
            self.captions.append(line.strip())

        for line in open(os.path.join(annFile_root, '%s_ids.txt' % subset), 'r', encoding='utf-8'):
            for i in range(self.num_anns):
                self.images_id.append(str(line.strip()))

        # get images' name list
        image_name = []
        for id in self.images_id:
            image_name.append(str(id)+'.jpg')

        self.image_name = image_name

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
