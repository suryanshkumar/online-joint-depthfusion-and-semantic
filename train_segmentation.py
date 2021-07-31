import torch
import torch.nn as nn
import argparse
import datetime
import os
import numpy as np

from tqdm import tqdm

from utils import setup
from utils import loading
from utils.metrics import runningScore

from modules.adapnet import AdapNet


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--comment', type=str, default='')

    args = parser.parse_args()

    return vars(args)


def prepare_input_data(batch, config, device):
    inputs = {}
    inputs['image'] = batch['image'] / 255.0
    inputs['image'] = inputs['image'].to(device).float() # (batch size, channels, height, width)

    if config.DATA.input != 'image':
        inputs[config.DATA.input] = batch[config.DATA.input]
        inputs[config.DATA.input] = inputs[config.DATA.input].unsqueeze(1).to(device).float()

    target = batch[config.DATA.target] # (batch size, height, width)
    target = target.to(device).long()

    return inputs, target


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
        #torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")


def train(args, config):

    # set seed for reproducibility
    if config.SETTINGS.seed:
        np.random.seed(config.SETTINGS.seed)
        torch.manual_seed(config.SETTINGS.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0" if config.SETTINGS.gpu else "cpu")

    config.TIMESTAMP = datetime.datetime.now().strftime('%y%m%d-%H%M%S')

    workspace = setup.get_workspace(config)
    config.SETTINGS.device = device

    # get train dataset
    train_data_config = setup.get_data_config(config, mode='train')
    train_dataset = setup.get_data(config.DATA.dataset, train_data_config)
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                batch_size=config.TRAINING.train_batch_size, 
                                                shuffle=config.TRAINING.train_shuffle, 
                                                pin_memory=True,
                                                num_workers=config.SETTINGS.num_workers)

    # get val dataset
    val_data_config = setup.get_data_config(config, mode='val')
    val_dataset = setup.get_data(config.DATA.dataset, val_data_config)
    val_loader = torch.utils.data.DataLoader(val_dataset, 
                                             batch_size=config.TRAINING.val_batch_size, 
                                             shuffle=config.TRAINING.val_shuffle, 
                                             pin_memory=True,
                                             num_workers=config.SETTINGS.num_workers)

    # define metrics
    ignore_index = 0 if config.DATA.dataset == 'ScanNet' else -100
    running_metrics_val = runningScore(config.SEMANTIC_MODEL.n_classes, ignore_index=ignore_index)

    # define model
    if config.SEMANTIC_MODEL.name == 'adapnet':
        model = AdapNet(config.SEMANTIC_MODEL)
        
        if config.SEMANTIC_MODEL.stage == 1:
            # encoder initialized with resnet50
            model.decoder.apply(weights_init) 
            model.eASPP.apply(weights_init)

        elif config.SEMANTIC_MODEL.stage == 2:
            model.apply(weights_init)

            weights = torch.load(config.SEMANTIC_MODEL.pretrained_rgb, map_location=device)["model_state"]
            # when using >1 gpus the model path begins with 'module.', thus remove it
            weights = loading.remove_parent(weights, 'module')
            # load rgb encoder weights
            encoder_weights = loading.select_child(weights, 'encoder_mod1')
            model.encoder_mod1.load_state_dict(encoder_weights)
            eASPP_weights = loading.select_child(weights, 'eASPP')
            model.eASPP_mod1.load_state_dict(eASPP_weights)

            weights = torch.load(config.SEMANTIC_MODEL.pretrained_tof, map_location=device)["model_state"]
            # when using >1 gpus the model path begins with 'module.', thus remove it
            weights = loading.remove_parent(weights, 'module')
            # load depth encoder weights
            encoder_weights = loading.select_child(weights, 'encoder_mod1')
            model.encoder_mod2.load_state_dict(encoder_weights)
            eASPP_weights = loading.select_child(weights, 'eASPP')
            model.eASPP_mod2.load_state_dict(eASPP_weights)

            model.no_resn50_dropout()
    else:
        print("Wrong model defined")

    load = False
    if config.SEMANTIC_MODEL.pretrained:
        if os.path.isfile(config.SEMANTIC_MODEL.pretrained):
            print("Using pre-trained semantic model {}".format(config.SEMANTIC_MODEL.pretrained))
            model.apply(weights_init)
            weights = torch.load(config.SEMANTIC_MODEL.pretrained, map_location=device)["model_state"]
            model.load_state_dict(weights)
            load = True
        else:
            print("No model found at '{}'".format(config.SEMANTIC_MODEL.pretrained))

    total_params = sum(p.numel() for p in model.parameters())
    print('Parameters:', total_params)    

    # define optimizer
    optimizer_cls = setup.get_optimizer(config.TRAINING.optimizer)
    optimizer_params = {k: v for k, v in config.TRAINING.optimizer.items() if k != "name"}
    optimizer = optimizer_cls(model.parameters(), **optimizer_params)

    # define scheduler
    scheduler = setup.get_scheduler(optimizer, config.TRAINING.scheduler)

    # define loss function
    criterion = setup.get_loss_function(config.TRAINING.loss, device)

    # load training checkpoint
    start_epoch = 0
    if config.TRAINING.resume:
        if os.path.isfile(config.TRAINING.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(config.TRAINING.resume))
            checkpoint = torch.load(config.TRAINING.resume, map_location=device)
            weights = loading.remove_parent(weights, 'module')
            model.load_state_dict(weights)
            model.to(device)
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_epoch = checkpoint["epoch"]
        else:
            print("No checkpoint found at '{}'".format(config.TRAINING.resume))

    n_train_batches = int(len(train_dataset) / config.TRAINING.train_batch_size)
    n_val_batches = int(len(val_dataset) / config.TRAINING.val_batch_size)

    # sample validation visualization frames
    val_vis_ids = np.random.choice(np.arange(0, n_val_batches), 10, replace=False)

    best_iou = -100.0

    # if available, divide batch size on multiple GPUs
    if torch.cuda.device_count() > 1 and config.SETTINGS.multigpu:
        print("Let's use {} GPUs".format(torch.cuda.device_count()))
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model = model.to(device)

    for epoch in range(start_epoch, config.TRAINING.n_epochs):
        print('epoch: ', epoch)

        val_loss_t = 0.
        train_loss_t = 0.

        # TRAINING
        model.train()
        for i, batch in enumerate(tqdm(train_loader, total=n_train_batches, mininterval=30)):

            optimizer.zero_grad()
            
            inputs, target = prepare_input_data(batch, config, device)

            if inputs[config.DATA.input].shape[1] == 1:
                inputs[config.DATA.input] = inputs[config.DATA.input].repeat(1, 3, 1, 1)

            if config.SEMANTIC_MODEL.stage == 1:
                output = model.forward(inputs[config.DATA.input]) # either 'image' or 'tof_depth'
            else:
                if config.TRAINING.optimization.random_mask:
                    if np.random.random_sample() <= config.TRAINING.optimization.mask_prob:
                        #workspace.log('Epoch {} Masking RGB channel'.format(epoch), mode='train')
                        inputs['image'] = torch.zeros_like(inputs['image'])
                    elif np.random.random_sample() <= config.TRAINING.optimization.mask_prob:
                        #workspace.log('Epoch {} Masking ToF channel'.format(epoch), mode='train')
                        inputs[config.DATA.input] = torch.zeros_like(inputs[config.DATA.input])
                output = model.forward(inputs['image'], inputs[config.DATA.input])

            # compute training loss
            loss = criterion.forward(output[0], target) + 0.6 * criterion.forward(output[1], target) + 0.5 * criterion.forward(output[2], target)

            # compute gradient and do SGD step
            loss.backward()
            optimizer.step()

            # record loss for analysis
            train_loss_t += loss.item()

        # compute lr
        scheduler.step()
        lr = scheduler.get_lr()
        workspace.log('Epoch {} Learning Rate {}'.format(epoch, lr[0]), mode='train')

        # log training metrics
        train_loss_t /= n_train_batches

        workspace.log('Epoch {} Training Loss {}'.format(epoch, train_loss_t), mode='train')
        workspace.writer.add_scalar('Train/loss_t', train_loss_t, global_step=epoch)

        torch.cuda.empty_cache()

        # VALIDATION
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, total=n_val_batches, mininterval=30)):

                inputs, target = prepare_input_data(batch, config, device)

                if inputs[config.DATA.input].shape[1] == 1:
                    inputs[config.DATA.input] = inputs[config.DATA.input].repeat(1, 3, 1, 1)

                if config.SEMANTIC_MODEL.stage == 1:
                    output = model.forward(inputs[config.DATA.input])
                else:
                    output = model.forward(inputs['image'], inputs[config.DATA.input]) # res, aux1 (before skip1), aux2 (before skip2)

                # compute validation loss
                loss = criterion.forward(output[0], target) + 0.6 * criterion.forward(output[1], target) + 0.5 * criterion.forward(output[2], target)

                val_loss_t += loss.item()

                # compute validation metrics
                res = torch.softmax(output[0], dim=1)
                gt = target.data.detach().cpu().numpy()
                pred = res.data.max(1)[1].detach().cpu().numpy()
                running_metrics_val.update(gt, pred)

                # visualize frames
                if i in val_vis_ids:
                    # get semantic images (c=1, but tensorboard works with c=3)
                    frame_est = torch.max(output[0], 1)[1]
                    frame_est = frame_est[0, :, :].repeat(3, 1, 1).cpu().detach().numpy()
                    frame_gt = target[0, :, :].repeat(3, 1, 1).cpu().detach().numpy()
                    frame_l1 = np.abs(frame_est - frame_gt)

                    # write to logger
                    workspace.writer.add_image('Val/est_{}'.format(i), frame_est, global_step=epoch)
                    workspace.writer.add_image('Val/gt_{}'.format(i), frame_gt, global_step=epoch)
                    workspace.writer.add_image('Val/l1_{}'.format(i), frame_l1, global_step=epoch)

        # log validation metrics
        val_loss_t /= n_val_batches

        workspace.log('Epoch {} Validation Loss {}'.format(epoch, val_loss_t), mode='val')
        workspace.writer.add_scalar('Val/loss_t', val_loss_t, global_step=epoch)

        score, class_iou = running_metrics_val.get_scores()

        for k, v in score.items():
            workspace.log("{:12}:\t{}".format(k, v), mode='val')            
            workspace.writer.add_scalar("val_metrics/{}".format(k), v, global_step=epoch)
        for k, v in class_iou.items():
            name = val_dataset.names_map[int(k)]
            workspace.log("{:2}: {:16}:\t{}".format(k, name, v), mode='val')
            workspace.writer.add_scalar("val_metrics/cls_{}".format(k), v, global_step=epoch)
        workspace.log("\n\n", mode='val')

        running_metrics_val.reset()

        # store best model
        if score["Mean IoU"] >= best_iou:
            best_iou = score["Mean IoU"]
            state = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "best_iou": best_iou,
            }
            workspace.log('Found new best model with score {} at epoch {}'.format(best_iou, epoch), mode='val')
            workspace.save_model_state(state, is_best=True)
        
        # store last model
        state = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }
        workspace.save_model_state(state)

        torch.cuda.empty_cache()


if __name__ == '__main__':

    # get arguments
    args = arg_parser()
    print(args['comment'])

    # get configs
    config = loading.load_config_from_yaml(args['config'])

    # train
    train(args, config)

