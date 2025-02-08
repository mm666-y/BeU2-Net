import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix

import utils
from utils import save_imgs,f_score
from torch.nn import functional as F
from utils import *


def train_one_epoch(train_loader,
                    model,
                    criterion_seg, 
                    criterion_edge,
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        images, targets, edges = data
        images, targets, edges = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float(), edges.cuda(non_blocking=True).float()

        out = model(images)
        out_seg = out[:7]
        out_edge = out[7]
        
        loss_seg = [criterion_seg(torch.sigmoid(out_seg[i]), targets) for i in range(len(out_seg))] # sum up the loss of 6 side maps and 1 fused map
        # loss_seg = [F.binary_cross_entropy_with_logits(out_seg[i], targets) for i in range(len(out_seg))]
        loss_seg = sum(loss_seg)
        
        edge0 = torch.unsqueeze(out_edge[:,0,:,:], dim =1)
        edge1 = torch.unsqueeze(out_edge[:,1,:,:], dim =1)
        edge2 = torch.unsqueeze(out_edge[:,2,:,:], dim =1)
        edge3 = torch.unsqueeze(out_edge[:,3,:,:], dim =1)
        edge4 = torch.unsqueeze(out_edge[:,4,:,:], dim =1)
        
        loss_edge0 = criterion_edge(edge0,torch.unsqueeze(edges[:,0,:,:], dim =1))
        loss_edge1 = criterion_edge(edge1,torch.unsqueeze(edges[:,1,:,:], dim =1))
        loss_edge2 = criterion_edge(edge2,torch.unsqueeze(edges[:,2,:,:], dim =1))
        loss_edge3 = criterion_edge(edge3,torch.unsqueeze(edges[:,3,:,:], dim =1))
        loss_edge4 = criterion_edge(edge4,torch.unsqueeze(edges[:,4,:,:], dim =1))
        loss_edge = loss_edge0 + loss_edge1 + loss_edge2 + loss_edge3 + loss_edge4
        
        
        loss = loss_seg + loss_edge
        
        # # 加上信息熵作为正则化项(试了，不行)
        # pred = out[0] #模型输出
        # pred = F.softmax(pred,dim = 1)
        # entropy =torch.sum(-torch.sum(pred*torch.log(pred),dim = 1)) 
        # loss = loss + entropy
        # ######################
       
        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step() 
    return step


def val_one_epoch(test_loader,
                    model,
                    criterion_seg,
                    criterion_edge,
                    epoch, 
                    logger,
                    config):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk,edges,fname = data
            img, msk,edges = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float(), edges.cuda(non_blocking=True).float()

            out = model(img)
            out_seg = out[:7]
            out_edge = out[7]

            loss_seg = [criterion_seg(torch.sigmoid(out_seg[i]), msk) for i in range(len(out_seg))]  # sum up the loss of 6 side maps and 1 fused map
            # loss_seg = [F.binary_cross_entropy_with_logits(out_seg[i], msk) for i in range(len(out_seg))]
            loss_seg = sum(loss_seg)
            edge0 = torch.unsqueeze(out_edge[:, 0, :, :], dim=1)
            edge1 = torch.unsqueeze(out_edge[:, 1, :, :], dim=1)
            edge2 = torch.unsqueeze(out_edge[:, 2, :, :], dim=1)
            edge3 = torch.unsqueeze(out_edge[:, 3, :, :], dim=1)
            edge4 = torch.unsqueeze(out_edge[:, 4, :, :], dim=1)

            loss_edge0 = criterion_edge(edge0, torch.unsqueeze(edges[:, 0, :, :], dim=1))
            loss_edge1 = criterion_edge(edge1, torch.unsqueeze(edges[:, 1, :, :], dim=1))
            loss_edge2 = criterion_edge(edge2, torch.unsqueeze(edges[:, 2, :, :], dim=1))
            loss_edge3 = criterion_edge(edge3, torch.unsqueeze(edges[:, 3, :, :], dim=1))
            loss_edge4 = criterion_edge(edge4, torch.unsqueeze(edges[:, 4, :, :], dim=1))
            loss_edge = loss_edge0 + loss_edge1 + loss_edge2 + loss_edge3 + loss_edge4

            loss = loss_seg + loss_edge

            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is list:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)
    
    return np.mean(loss_list)


def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    # preds = []
    # gts = []
    loss_list = []
    f1_score = []
    mIoU = 0
    PA = 0
    iu_class0 = 0
    iu_class1 = 0
    iu_class2 = 0
    iu_class3 = 0
    iu_class4 = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk,edges,fnames = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            # 在cuda上计算的部分
            out = model(img)
            _f_score = f_score(out[0],msk)
            f1_score.append(_f_score)
            loss = criterion(out[0], msk)
            loss_list.append(loss.item())
            # 在cpu上计算的部分
            msk = msk.squeeze(1).cpu().detach().numpy()
            pred = out[0]
            # pred = torch.softmax(pred)
            # pred = pred.argmax(1)#1,5,256,256 jia softmax tensor
            # gts.append(msk)
            pred = pred.squeeze(1).cpu().detach().numpy() #1,5,256,256 nd.array
            # preds.append(out)

            if i % config.save_interval == 0:
                save_imgs(img, msk, pred, fnames, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

            assert msk.shape == pred.shape
            msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
            msk = msk.argmax(axis=0)
            pred = np.where(np.squeeze(pred, axis=0) > config.threshold, 1, 0)
            pred = pred.argmax(axis=0)

            hist, labeled,correct = utils.hist_info(pred,msk,config.num_classes)
            iu, mean_IU, mean_IU_no_back, mean_pixel_acc = utils.compurte_score(hist, correct, labeled)
            mIoU = mean_IU + mIoU
            PA = PA + mean_pixel_acc
            iu_class0 = iu_class0 + iu[0]
            iu_class1 = iu_class1 + iu[1]
            iu_class2 = iu_class2 + iu[2]
            iu_class3 = iu_class3 + iu[3]
            iu_class4 = iu_class4 + iu[4]


        testdata_size = len(f1_score)
        f1_or_dsc = sum(f1_score)/testdata_size
        iou_class0 = iu_class0 / testdata_size
        iou_class1 = iu_class1 / testdata_size
        iou_class2 = iu_class2 / testdata_size
        iou_class3 = iu_class3 / testdata_size
        iou_class4 = iu_class4 / testdata_size
        mIoU = mIoU/testdata_size
        PA = PA/testdata_size

        # preds = np.array(preds).reshape(-1) #68*(1,5,256,256) -> 256*256*5*68=22282240
        # gts = np.array(gts).reshape(-1) #68*(1,5,256,256) ->
        #
        # y_pre = np.where(preds>=config.threshold, 1, 0) #256*256*5*68=22282240
        # y_true = np.where(gts>=0.5, 1, 0) #256*256*5*68=22282240
        #
        # confusion = confusion_matrix(y_true, y_pre)
        # TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]

        # accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        # sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        # specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        # # f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        # miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},mIoU: {mIoU}, f1_or_dsc: {f1_or_dsc}, IoU of background: {iou_class0},\
                IoU of LV: {iou_class1}, IoU of RV: {iou_class2},IoU of RA: {iou_class3},IoU of LA: {iou_class4}, PA:{PA}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)