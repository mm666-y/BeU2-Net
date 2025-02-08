import torch
from torch.utils.data import DataLoader
# import timm
from datasets.dataset import A4C_train_datasets, A4C_val_datasets,A4C_test_datasets
from tensorboardX import SummaryWriter
from models.u2net_var1 import u2net_lite

from engine import *
import os
import sys

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")



def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)





    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()





    print('#----------Preparing dataset----------#')
    train_dataset = A4C_train_datasets(config.data_path, config)
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers,
                                drop_last=True)
    val_dataset = A4C_val_datasets(config.data_path, config)
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)
    test_dataset = A4C_test_datasets(config.data_path, config, train=False,test=True)
    test_loader = DataLoader(test_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)




    print('#----------Prepareing Model----------#')
    # model_cfg = config.model_config
    # if config.network == 'vmunet':
    #     model = VMUNet(
    #         num_classes=model_cfg['num_classes'],
    #         input_channels=model_cfg['input_channels'],
    #         depths=model_cfg['depths'],
    #         depths_decoder=model_cfg['depths_decoder'],
    #         drop_path_rate=model_cfg['drop_path_rate'],
    #         load_ckpt_path=model_cfg['load_ckpt_path'],
    #     )
    #     model.load_from()
        
    # else: raise Exception('network in not right!')
    model = u2net_lite(out_ch=config.num_classes)
    model = model.cuda()

    # cal_params_flops(model, 256, logger)





    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion_seg = config.criterion_seg
    criterion_edge = config.criterion_edge
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)





    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1





    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'],strict = False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'],strict = False)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'],strict = False)
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)




    step = 0
    print('#----------Training----------#')
    total = sum([param.nelement() for param in model.parameters()])
    print('Number of parameter:%.2fM' % (total / 1e6) )
    print('Folds of the training:',config.train_list, 'length of the training dataset:',len(train_dataset))
    print('Folds of the validation:', config.val_list, 'length of the validation dataset:', len(val_dataset))
    print('Folds of the testing:', config.test_list, 'length of the testing dataset:', len(test_dataset))
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        step = train_one_epoch(
            train_loader,
            model,
            criterion_seg,
            criterion_edge,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer
        )

        loss = val_one_epoch(
                val_loader,
                model,
                criterion_seg,
                criterion_edge,
                epoch,
                logger,
                config
            )

        if loss < min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

        # torch.save(
        #     {
        #         'epoch': epoch,
        #         'min_loss': min_loss,

        #         'min_epoch': min_epoch,

        #         'loss': loss,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'scheduler_state_dict': scheduler.state_dict(),
        #     }, os.path.join(checkpoint_dir, 'latest.pth')) 
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'latest.pth')) 

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        loss = test_one_epoch(
                test_loader,
                model,
                criterion_seg,
                logger,
                config,
            )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )      


if __name__ == '__main__':
    config = setting_config
    main(config)