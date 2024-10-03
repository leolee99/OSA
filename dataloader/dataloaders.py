import os
from torch.utils.data import DataLoader
from dataloader.dataloader_coco import MSCOCO_Dataset
from dataloader.dataloader_f30k import F30K_Dataset
from dataloader.dataloader_cc import CC_Dataset


def prepare_coco_dataloaders(args,
                             dataset_root,
                             preprocess,
                             tokenizer,
                             logger):
    """Prepare MS-COCO Caption train / val / test dataloaders
    Args:
        dataset_root (str): root of your MS-COCO dataset (see README.md for detailed dataset hierarchy)
        preprocess: preprocess function for images
        tokenizer: the tokenizer used to encode captions
        logger: logger
    Returns:
        dataloaders (dict): keys = ["train", "val", "test"], values are the corresponding dataloaders.
    """

    image_root = os.path.join(dataset_root, 'images/')
    ann_root = os.path.join(dataset_root, 'annotations/')

    dataloaders = {}

    if args.eval:
        dataloaders['train'] = None, None
    else:
        dataloaders['train'] = dataloader_mscoco_train(
            args, image_root, ann_root, preprocess, tokenizer,
            None, 'train', logger,
        )

    dataloaders['val'] = dataloader_mscoco_test(
        args, image_root, ann_root, preprocess, tokenizer,
        None, 'dev', logger,
    )

    dataloaders['test'] = dataloader_mscoco_test(
        args, image_root, ann_root, preprocess, tokenizer,
        None, 'test', logger,
    )

    return dataloaders

def dataloader_mscoco_train(args, image_root, annFile, preprocess, tokenizer, ids, subset, logger):
    msrvtt_dataset = MSCOCO_Dataset(
                                    args,
                                    image_root,
                                    annFile,
                                    preprocess,
                                    tokenizer,
                                    ids=ids,
                                    subset=subset,
                                    logger=logger,
    )

    #train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size,
        shuffle=(subset == 'train'),
        num_workers=args.num_workers,
        pin_memory=True,
        #shuffle=(train_sampler is None),
        #sampler=train_sampler,
        drop_last=False,
    )

    return dataloader, len(msrvtt_dataset)

def dataloader_mscoco_test(args, image_root, annFile, preprocess, tokenizer, ids, subset, logger):
    msrvtt_dataset = MSCOCO_Dataset(
                                    args,
                                    image_root,
                                    annFile,
                                    preprocess,
                                    tokenizer,
                                    ids=ids,
                                    subset=subset,
                                    logger=logger,
    )

    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.eval_batch_size,
        shuffle=(subset == 'train'),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return dataloader, len(msrvtt_dataset)#, train_sampler



def prepare_f30k_dataloaders(args,
                             dataset_root,
                             preprocess,
                             tokenizer,
                             logger):
    """Prepare Flickr30K train / val / test dataloaders
    Args:
        dataset_root (str): root of your MS-COCO dataset (see README.md for detailed dataset hierarchy)
        preprocess: preprocess function for images
        tokenizer: the tokenizer used to encode captions
        logger: logger
    Returns:
        dataloaders (dict): keys = ["train", "val", "test"], values are the corresponding dataloaders.
    """

    image_root = os.path.join(dataset_root, 'images/')
    ann_root = os.path.join(dataset_root, 'annotations/')

    dataloaders = {}

    if args.eval:
        dataloaders['train'] = None, None
    else:
        dataloaders['train'] = dataloader_f30k_train( 
            args, image_root, ann_root, preprocess, tokenizer,
            None, 'train', logger, 
        ) 

    dataloaders['val'] = dataloader_f30k_test(
        args, image_root, ann_root, preprocess, tokenizer,
        None, 'dev', logger,
    )

    dataloaders['test'] = dataloader_f30k_test(
        args, image_root, ann_root, preprocess, tokenizer,
        None, 'test', logger,
    )

    return dataloaders


def dataloader_f30k_train(args, image_root, annFile, preprocess, tokenizer, ids, subset, logger):
    msrvtt_dataset = F30K_Dataset(
                                    args,
                                    image_root,
                                    annFile,
                                    preprocess,
                                    tokenizer,
                                    ids=ids,
                                    subset=subset,
                                    logger=logger,
    )

    #train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size,
        shuffle=(subset == 'train'),
        num_workers=args.num_workers,
        pin_memory=True,
        #shuffle=(train_sampler is None),
        #sampler=train_sampler,
        drop_last=False,
    )

    return dataloader, len(msrvtt_dataset)

def dataloader_f30k_test(args, image_root, annFile, preprocess, tokenizer, ids, subset, logger):
    msrvtt_dataset = F30K_Dataset(
                                    args,
                                    image_root,
                                    annFile,
                                    preprocess,
                                    tokenizer,
                                    ids=ids,
                                    subset=subset,
                                    logger=logger,
    )

    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.eval_batch_size,
        shuffle=(subset == 'train'),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return dataloader, len(msrvtt_dataset)#, train_sampler


def prepare_cc_dataloaders(args,
                             dataset_root,
                             preprocess,
                             tokenizer,
                             logger):
    """Prepare CC120K Caption train / val / test dataloaders
    Args:
        dataset_root (str): root of your MS-COCO dataset (see README.md for detailed dataset hierarchy)
        preprocess: preprocess function for images
        tokenizer: the tokenizer used to encode captions
        logger: logger
    Returns:
        dataloaders (dict): keys = ["train", "val", "test"], values are the corresponding dataloaders.
    """

    image_root = os.path.join(dataset_root, 'images/')
    ann_root = os.path.join(dataset_root, 'annotations/')

    dataloaders = {}

    if args.eval:
        dataloaders['train'] = None, None
    else:
        dataloaders['train'] = dataloader_cc_train(
            args, image_root, ann_root, preprocess, tokenizer,
            None, 'train', logger, 
        ) 

    dataloaders['val'] = dataloader_cc_test(
        args, image_root, ann_root, preprocess, tokenizer,
        None, 'dev', logger,
    )

    dataloaders['test'] = dataloader_cc_test(
        args, image_root, ann_root, preprocess, tokenizer,
        None, 'test', logger,
    )

    return dataloaders

def dataloader_cc_train(args, image_root, annFile, preprocess, tokenizer, ids, subset, logger):
    msrvtt_dataset = CC_Dataset(
                                    args,
                                    image_root,
                                    annFile,
                                    preprocess,
                                    tokenizer,
                                    ids=ids,
                                    subset=subset,
                                    logger=logger,
    )

    #train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size,
        shuffle=(subset == 'train'),
        num_workers=args.num_workers,
        pin_memory=True,
        #shuffle=(train_sampler is None),
        #sampler=train_sampler,
        drop_last=False,
    )

    return dataloader, len(msrvtt_dataset)

def dataloader_cc_test(args, image_root, annFile, preprocess, tokenizer, ids, subset, logger):
    msrvtt_dataset = CC_Dataset(
                                    args,
                                    image_root,
                                    annFile,
                                    preprocess,
                                    tokenizer,
                                    ids=ids,
                                    subset=subset,
                                    logger=logger,
    )

    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.eval_batch_size,
        shuffle=(subset == 'train'),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return dataloader, len(msrvtt_dataset)#, train_sampler


def prepare_dataloaders(args,
                        dataset_root,
                        preprocess,
                        tokenizer=None,
                        logger=None,):
    if args.dataset == 'coco':
        return prepare_coco_dataloaders(args, dataset_root, preprocess, tokenizer, logger)
    if args.dataset == 'f30k':
        return prepare_f30k_dataloaders(args, dataset_root, preprocess, tokenizer, logger)
    if args.dataset == 'cc':
        return prepare_cc_dataloaders(args, dataset_root, preprocess, tokenizer, logger)