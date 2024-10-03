"""
# The pytorch implementation of the paper "One-step Noisy Label Mitigation".
# 
# https://github.com/leolee99/OSA
#
# Writen by Hao Li, 2024
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from torch import optim
from util import set_seed_logger, get_logger
from params import parse_args
from scheduler import cosine_lr
from eval import evaluate
from dataloader.dataloaders import prepare_dataloaders

global logger

#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

def set_requires_grad(net: nn.Module, mode=True):
    for p in net.parameters():
        p.requires_grad_(mode)


def weight_function(x, thr=15):
    x = x.diag() - thr
    x[x < 0] = 0
    x = x / x.max()

    return -torch.pow(x, 2) * (x - 1) * (x + 1)

class CrossEn(nn.Module):
    def __init__(self,):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix, weight):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt * weight
        sim_loss = nce_loss.mean()
        return sim_loss


def main():
    global logger
    args = parse_args()

    seed = set_seed_logger(args)
    dir_path = os.path.join(args.checkpoint_path, args.experiments)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    logger = get_logger(os.path.join(dir_path, "log.txt"))

    logger.info("Effective parameters:")
    for key in sorted(args.__dict__):
        logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.

    model_clip, preprocess = clip.clip.load(args.vision_model, device=device, jit=False) # initial target model
    estimator_model, shadow_preprocess = clip.clip.load(args.vision_model, device=device, jit=False) # initial estimator model
    tokenizer = clip.clip.tokenize # initial tokenizer

    # resume model
    if args.resume:
        checkpoint = torch.load(args.resume)
        model = model_clip
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("Loaded model from {}".format(args.resume))

    else:
        model = model_clip
        logger.info("Model Initialized!")

    model = model.cuda()

    dataloader = prepare_dataloaders(args, args.dataset_root, preprocess, tokenizer, logger=logger)

    if args.eval:
        train_dataloader = None
        train_length = 0
        args.epochs = 0
        test_dataloader, test_length = dataloader['test']
        Mn_R1 = evaluate(args, model, test_dataloader, logger)

    else:
        train_dataloader, train_length = dataloader['train']
        val_dataloader, val_length = dataloader['val']
        test_dataloader, test_length = dataloader['test']


    loss_img = CrossEn()
    loss_txt = CrossEn()

    loss_img = loss_img.cuda()
    loss_txt = loss_txt.cuda()

    total_steps = train_length * args.epochs

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    for name, param in model.state_dict().items():
        estimator_model.state_dict()[name].copy_(param)
    set_requires_grad(estimator_model, False)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", train_length)
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Num steps = %d", total_steps)

    with torch.no_grad():
        estimator_model.eval()
        # shift threshold generation
        rand_img = torch.rand(args.batch_size, 3, 224, 224).cuda()
        rand_txt = (torch.rand(args.batch_size, 77) * 100).long().cuda()
        logits_rand_i, logits_rand_t = estimator_model(rand_img, rand_txt)
        thr = ((logits_rand_i.mean() + logits_rand_t.mean())/2).item()

    # print(thr)

    best_score = 0
    for epoch in range(args.epochs):
        model.train()
        sloss = 0
        for idx, batch in enumerate(train_dataloader):
            step = train_length * epoch + idx
            scheduler(step)

            optimizer.zero_grad()

            images, texts, *_ = batch
            images = images.cuda()
            texts = texts.cuda()

            logits_per_image, logits_per_text = model(images, texts)

            with torch.no_grad():
                estimator_model.eval()
                logits_per_image_est, logits_per_text_est = estimator_model(images, texts)


            img_diag = weight_function(logits_per_image_est, thr)
            txt_diag = weight_function(logits_per_text_est, thr)


            total_loss = (loss_img(logits_per_image, img_diag) + loss_txt(logits_per_text, txt_diag)) / 2
            total_loss.backward()

            sloss += float(total_loss)

            if device == "cpu":
                optimizer.step()
            else :
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

            if (idx % args.display == 0) and (idx != 0):
                logger.info("Epoch: %d/%d, step:%d/%d, lr: %.8f, loss: %f", epoch + 1, args.epochs, idx, len(train_dataloader), optimizer.param_groups[0]['lr'], sloss / args.display)
                sloss = 0

        save_path = os.path.join(dir_path, f"epoch{epoch + 1}.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                "name": args.name,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            save_path,
        )
        logger.info("Saved checkpoint {} (epoch {})".format(save_path, epoch + 1))

        ## Run on val dataset for selecting best model.
        logger.info("Eval on val dataset")
        Mn_R1 = evaluate(args, model, val_dataloader, logger)

        if best_score <= Mn_R1:
            best_score = Mn_R1
            best_output_model_file = save_path
        logger.info("The best model is: {}, the R1 is: {:.4f}".format(best_output_model_file, best_score))

    ## Run on test dataset for best model.
    if not args.eval:
        checkpoint = torch.load(best_output_model_file)
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("Loaded model from {}".format(best_output_model_file))
        logger.info("Eval on test dataset")
        Mn_R1 = evaluate(args, model, test_dataloader, logger)
    
if __name__ == '__main__':
    main()

