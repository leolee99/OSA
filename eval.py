import torch
from tqdm import tqdm

def evaluate(args, model, dataloader, logger, split=1):
    model.eval()
    with torch.no_grad():
        image_features = []
        text_features = []
        num_anns = dataloader.dataset.num_anns
        num_ids = len(dataloader.dataset)
        num_imgs = dataloader.dataset.img_length
        for idx, batch in enumerate(dataloader):
            images, texts, *_ = batch
            images = images.cuda()
            texts = texts.cuda()
            batch_image_features = model.encode_image(images)
            batch_text_features = model.encode_text(texts)



            batch_image_features = batch_image_features / batch_image_features.norm(dim=1, keepdim=True)
            batch_text_features = batch_text_features / batch_text_features.norm(dim=1, keepdim=True)

            image_features.append(batch_image_features)
            text_features.append(batch_text_features)

            if idx % args.display == 0:
                logger.info("step:%d/%d", idx, len(dataloader))

        images_ids = torch.arange(0, num_ids, num_anns).cuda()
        image_features = torch.cat(image_features, dim=0)[images_ids]
        text_features = torch.cat(text_features, dim=0)

        caption_num = 5

        if args.dataset == 'cc':
            caption_num = 1

        image_features = image_features.view(-1, caption_num, image_features.shape[-1]).mean(1).squeeze()
        sim_matrix = []
        
        for idx, image_feat in tqdm(enumerate(image_features)):
            logit_scale = 100
            sim_line = logit_scale * image_feat @ text_features.t()

            sim_matrix.append(sim_line.unsqueeze(0).cpu())
        
        sim_matrix = torch.cat(sim_matrix, dim=0)
        label = torch.eye(num_imgs).unsqueeze(-1).repeat(1,1,caption_num).view(-1, num_ids)

        if args.dataset == 'coco':
            # evaluate on COCO Test set
            if sim_matrix.shape[0] > 1000:
                results = {'i2t_R@1':0, 'i2t_R@5':0, 'i2t_R@10':0, 't2i_R@1':0, 't2i_R@5':0, 't2i_R@10':0, 'mean_R1':0}
                for i in range(5):
                    divided_sim = sim_matrix[1000 * i: 1000 * (i + 1) - 1, 5000 * i: 5000 * (i + 1) - 1]
                    divided_label = label[1000 * i: 1000 * (i + 1) - 1, 5000 * i: 5000 * (i + 1) - 1]
                    result = metric_compute(divided_sim, divided_label, logger)
                    for key, values in result.items():
                        results[key] += values
                for key, values in result.items():
                    results[key] /= 5

                logger.info("1K Image-to-Text:")
                logger.info('\t>>>  R@1: {:.2f} - R@5: {:.2f} - R@10: {:.2f}'.
                    format(results['i2t_R@1'], results['i2t_R@5'], results['i2t_R@10']))
                
                logger.info("1K Text-to-Image:")
                logger.info('\t>>>  R@1: {:.2f} - R@5: {:.2f} - R@10: {:.2f}'.
                    format(results['t2i_R@1'], results['t2i_R@5'], results['t2i_R@10']))
                
                logger.info("1K Mean R1: {:.2f}".format(results['mean_R1']))

                # test on COCO 5K
                results = metric_compute(sim_matrix, label, logger)
                
                logger.info("5K Image-to-Text:")
                logger.info('\t>>>  R@1: {:.2f} - R@5: {:.2f} - R@10: {:.2f}'.
                    format(results['i2t_R@1'], results['i2t_R@5'], results['i2t_R@10']))
                
                logger.info("5K Text-to-Image:")
                logger.info('\t>>>  R@1: {:.2f} - R@5: {:.2f} - R@10: {:.2f}'.
                    format(results['t2i_R@1'], results['t2i_R@5'], results['t2i_R@10']))
                
                logger.info("5K Mean R1: {:.2f}".format(results['mean_R1']))

            # evaluate on COCO valid set
            else:
                results = metric_compute(sim_matrix, label, logger)
                
                logger.info("1K Image-to-Text:")
                logger.info('\t>>>  R@1: {:.2f} - R@5: {:.2f} - R@10: {:.2f}'.
                    format(results['i2t_R@1'], results['i2t_R@5'], results['i2t_R@10']))
                
                logger.info("1K Text-to-Image:")
                logger.info('\t>>>  R@1: {:.2f} - R@5: {:.2f} - R@10: {:.2f}'.
                    format(results['t2i_R@1'], results['t2i_R@5'], results['t2i_R@10']))
                
                logger.info("1K Mean R1: {:.2f}".format(results['mean_R1']))

        elif args.dataset == 'f30k'or args.dataset == 'cc':
            results = metric_compute(sim_matrix, label, logger)
            
            logger.info("Image-to-Text:")
            logger.info('\t>>>  R@1: {:.2f} - R@5: {:.2f} - R@10: {:.2f}'.
                format(results['i2t_R@1'], results['i2t_R@5'], results['i2t_R@10']))
            
            logger.info("Text-to-Image:")
            logger.info('\t>>>  R@1: {:.2f} - R@5: {:.2f} - R@10: {:.2f}'.
                format(results['t2i_R@1'], results['t2i_R@5'], results['t2i_R@10']))
            
            logger.info("Mean R1: {:.2f}".format(results['mean_R1']))
            
    # ground_truth = torch.arange(len(images), dtype=torch.long).cuda()
    return results['mean_R1']


def metric_compute(sim_matrix, label, logger):
    results = {}
    # Image-to-Text
    i2t_rank_matrix = (-sim_matrix).argsort().argsort() + 1
    i2t_gt_rk_matrix = label * i2t_rank_matrix
    i2t_gt_rk_matrix[i2t_gt_rk_matrix==0] = 1e9
    i2t_min_rank = i2t_gt_rk_matrix.min(1).values

    results['i2t_R@1'] = 100 * torch.where(i2t_min_rank <= 1, 1, 0).type(torch.float32).mean()
    results['i2t_R@5'] = 100 * torch.where(i2t_min_rank <= 5, 1, 0).type(torch.float32).mean()
    results['i2t_R@10'] = 100 * torch.where(i2t_min_rank <= 10, 1, 0).type(torch.float32).mean()

    # Text-to-Image
    t2i_rank_matrix = (-sim_matrix.T).argsort().argsort() + 1
    t2i_gt_rk_matrix = label.T * t2i_rank_matrix
    t2i_gt_rk_matrix[t2i_gt_rk_matrix==0] = 1e9
    t2i_min_rank = t2i_gt_rk_matrix.min(1).values

    results['t2i_R@1'] = 100 * torch.where(t2i_min_rank <= 1, 1, 0).type(torch.float32).mean()
    results['t2i_R@5'] = 100 * torch.where(t2i_min_rank <= 5, 1, 0).type(torch.float32).mean()
    results['t2i_R@10'] = 100 * torch.where(t2i_min_rank <= 10, 1, 0).type(torch.float32).mean()
    
    results['mean_R1'] = (results['i2t_R@1'] + results['t2i_R@1']) / 2

    
    return results



    