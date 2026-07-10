import logging
import time
import torch
import torch.distributed as dist  # for mR early stopping (distributed broadcast)
from tqdm import tqdm
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
import numpy as np
import gc

def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("IRRA.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "cmpm_loss": AverageMeter(),
        "mlm_loss":AverageMeter(),
        "xpool_loss":AverageMeter(),
        "itc_loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "id_loss": AverageMeter(),

        "LLM_loss": AverageMeter(),
        "cfine_loss": AverageMeter(),
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_mR = 0.0
    evals_without_improvement = 0
    early_stop_patience = args.early_stop_patience
    stop_training = False
    arguments["epoch"] = start_epoch
    
    all = []
    # train
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()
        # store1_tem = []################
        # store2_tem = []################
        # idstore_tem = []###############
        # store1_tem.append(ifeats)################
        # store2_tem.append(tfeats)################
        # idstore_tem.append(batch['pids'])###############
        # similarity time 
        # start = time.time()
        # batch_sample = next(iter(train_loader))
        # print(batch_sample)

        # for n_iter, batch in tqdm(enumerate(train_loader)):
        for n_iter, batch in tqdm(enumerate(train_loader)):
            # print("Available keys in batch:", batch.keys())  # dict_keys(['caption_ids', 'image_ids', 'mlm_labels', 'mlm_ids', 'images', 'pids'])打印 batch 中的所 batch 中 labels 的一个样例
            # break 
        #     batch = {
        #         'images': images.to(device),
        # 'captions': captions.to(device),  # 确保 captions 是需要发送到模型的
        # 'labels': labels.to(device)
        #     }
            # images = batch['images'].to(device)  # 将图像数据传送到 GPU
            # caption_ids = batch['caption_ids'].to(device)  # 替换 'captions' 为实际的键 'caption_ids'
            # labels = batch['mlm_labels'].to(device)
            # tokens = batch['tokens'].to(device)

            batch = {k: (v.to(device) if k != 'caption' else v) for k, v in batch.items()}
            # imgids = batch.get('image_id', None)  # 获取图像ID
            # captions = batch.get('captions', None)  # 获取对应的文本描述

            # if imgids is not None and captions is not None:
            #     # 打印每个图像的 ID 和对应的文本
            #     for i, imgid in enumerate(imgids):
            #         print(f"Batch {n_iter} - Image ID: {imgid}")
            #         for caption in captions[i]:
            #             print(f"  Caption: {caption}")
            #     print()

            # img = img.to(device)

            # images = batch['images'].to(device)
            # tokens = batch['tokens'].to(device)
            # segments = batch['segments'].to(device)
            # input_masks = batch['attn_mask'].to(device)
            # images, captions, labels = batch[:3]
            # print("tokens data type:", tokens)
            # print("Batch content:", batch)

            t1 = time.time()
            ret = model(batch)
            t2 = time.time()
            all.append(t2-t1)

            # ret, ifeats,tfeats = model(batch,epoch,store1,store2,idstore)#########################################
            # store1_tem.append(ifeats)################
            # store2_tem.append(tfeats)################
            # idstore_tem.append(batch['pids'])###############
            # model = model.to(device).half()

            # ret = model(images, tokens, segments, input_masks)
            # ret = model(batch,images=images, tokens=tokens, segments=segments, input_masks=input_masks)
            # tokens = tokens.cuda()
            # segments = segments.cuda()
            # input_masks = input_masks.cuda()
            # images = images.cuda()
            # labels = labels.cuda()
           
            total_loss = sum([v for k, v in ret.items() if "loss" in k])

            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            meters['cmpm_loss'].update(ret.get('cmpm_loss', 0).item() if isinstance(ret.get('cmpm_loss', 0), torch.Tensor) else ret.get('cmpm_loss', 0), batch_size)
            meters['itc_loss'].update(ret.get('itc_loss', 0).item() if isinstance(ret.get('itc_loss', 0), torch.Tensor) else ret.get('itc_loss', 0), batch_size)
            meters['sdm_loss'].update(ret.get('sdm_loss', 0).item() if isinstance(ret.get('sdm_loss', 0), torch.Tensor) else ret.get('sdm_loss', 0), batch_size)
            # meters['id_loss'].update(ret.get('id_loss', 0).item() if isinstance(ret.get('id_loss', 0), torch.Tensor) else ret.get('id_loss', 0), batch_size)
            # meters['LLM_loss'].update(ret.get('LLM_loss', 0).item() if isinstance(ret.get('LLM_loss', 0), torch.Tensor) else ret.get('LLM_loss', 0), batch_size)
            # meters['mlmpre_loss'].update(ret.get('mlmpre_loss', 0), batch_size)
            # meters['mlmerror_loss'].update(ret.get('mlmerror_loss', 0), batch_size)
            # meters['mlm_loss'].update(ret.get('mlm_loss', 0).item() if isinstance(ret.get('mlm_loss', 0), torch.Tensor) else ret.get('mlm_loss', 0), batch_size)
            # meters['xpool_loss'].update(ret.get('xpool_loss', 0).item() if isinstance(ret.get('xpool_loss', 0), torch.Tensor) else ret.get('xpool_loss', 0), batch_size)
            meters['cfine_loss'].update(ret.get('cfine_loss', 0).item() if isinstance(ret.get('cfine_loss', 0), torch.Tensor) else ret.get('cfine_loss', 0), batch_size)
            # # meters['xpool_loss2'].update(ret.get('xpool_loss2', 0), batch_size)

            # meters['img_acc'].update(ret.get('img_acc', 0), batch_size)
            # meters['txt_acc'].update(ret.get('txt_acc', 0), batch_size)
            # meters['mlm_acc'].update(ret.get('mlm_acc', 0), 1)

            # img_f.extend(ret.get('image_feats').cpu().data.numpy())
            # img_ff = torch.tensor(img_f)
            # # print(img_ff.shape)
            # text_f.extend(ret.get('text_feats').cpu().data.numpy())
            # text_ff = torch.tensor(text_f)
            # # print(text_ff.shape)
            # label.extend(ret.get('labels').cpu().data.numpy())
            # pid.extend(ret.get('pids').cpu().data.numpy())
            optimizer.zero_grad()

            # print("Data type of total_loss before backward:", total_loss.dtype)
            # total_loss = total_loss.to(torch.float16)
            # if total_loss.dtype != torch.float16:
            #     total_loss = total_loss.half()

            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
            
        # img_f = torch.tensor(img_f)
        # pid = torch.tensor(pid)
        # text_f = torch.tensor(text_f)
        # label = torch.tensor(label)
        # print(img_f.shape,pid.shape,text_f.shape)
        # torch.save(img_f,'/home/hpc/LAB-data/disk-4T/syc_data/irra-v1/irra/feats/image.pt')
        # torch.save(text_f,'/home/hpc/LAB-data/disk-4T/syc_data/irra-v1/irra/feats/text.pt')
        # torch.save(label,'/home/hpc/LAB-data/disk-4T/syc_data/irra-v1/irra/feats/label.pt')
        # torch.save(pid,'/home/hpc/LAB-data/disk-4T/syc_data/irra-v1/irra/feats/pid.pt')
        # store1 = store1_tem[:]################
        # store2 = store2_tem[:]################
        # idstore = idstore_tem[:]###############
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)
        
        print("infer time:",np.average(all))

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    # top1 = evaluator.eval(model.module.eval())
                    mR = evaluator.eval(model.module.eval())
                else:
                    # top1 = evaluator.eval(model.eval())
                    mR = evaluator.eval(model.eval())

                torch.cuda.empty_cache()
                # if best_top1 < top1:
                #     best_top1 = top1
                if mR > best_mR:
                    best_mR = mR
                    evals_without_improvement = 0
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
                    logger.info(f"New best mR: {best_mR:.3f} at epoch {epoch}")
                else:
                    evals_without_improvement += 1
                    logger.info(
                        f"mR did not improve. Patience: {evals_without_improvement}/{early_stop_patience}"
                    )
                    if early_stop_patience > 0 and evals_without_improvement >= early_stop_patience:
                        logger.info(
                            f"Early stopping triggered. Best mR: {best_mR:.3f} at epoch {arguments['epoch']}"
                        )
                        stop_training = True

            if args.distributed:
                stop_flag = torch.tensor(
                    [1 if stop_training else 0], device="cuda", dtype=torch.int
                )
                dist.broadcast(stop_flag, src=0)
                stop_training = stop_flag.item() == 1

            if stop_training:
                break

    if get_rank() == 0:
        # logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")
        # # logger.info(f"best mR: {best_mR:.3f} at epoch {arguments['epoch']}")
        logger.info(f"best mR: {best_mR:.3f} at epoch {arguments['epoch']}")
    # similarity time 
    # end = time.time()
    # print("calculate similarity time:", end - start)
    
    gc.collect()                # 清理 Python 层内存垃圾
    torch.cuda.empty_cache()   # 清理显存缓存


def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())
