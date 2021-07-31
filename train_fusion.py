import torch
import argparse
import datetime
import numpy as np
import random
import os
import sys

from tqdm import tqdm

from utils import setup
from utils import loading
from utils import transform

from modules.pipeline import Pipeline

torch.autograd.set_detect_anomaly(True) 


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--comment', type=str, default='')

    args = parser.parse_args()
    return vars(args)


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)


# @profile
def train_fusion(args, config):

    # set seed for reproducibility
    if config.SETTINGS.seed:
        random.seed(config.SETTINGS.seed)
        np.random.seed(config.SETTINGS.seed)
        torch.manual_seed(config.SETTINGS.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    config.TIMESTAMP = datetime.datetime.now().strftime('%y%m%d-%H%M%S')

    # get workspace
    workspace = setup.get_workspace(config)

    if config.SETTINGS.gpu:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

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

    # get database
    train_database = setup.get_database(train_dataset, train_data_config)
    val_database = setup.get_database(val_dataset, val_data_config)

    # setup pipeline
    pipeline = Pipeline(config)
    pipeline.apply(weights_init)

    if config.FUSION_MODEL.pretrained:
        checkpoint = torch.load(config.FUSION_MODEL.pretrained, map_location=device)
        pipeline._fusion_network.load_state_dict(checkpoint["model_state"])

    total_params = sum(p.numel() for p in pipeline._fusion_network.parameters())
    print('Fusion Parameters:', total_params)  

    if config.DATA.semantics and config.DATA.semantic_strategy == 'predict':
        checkpoint = torch.load(config.TESTING.semantic_2d_model_path, map_location=device)
        weights = loading.remove_parent(checkpoint['model_state'], 'module')
        pipeline._semantic_2d_network.load_state_dict(weights)
        pipeline._semantic_2d_network.eval()

    pipeline = pipeline.to(device)

    # define optimizer and scheduler
    optimizer_cls = setup.get_optimizer(config.TRAINING.optimizer)
    optimizer_params = {k: v for k, v in config.TRAINING.optimizer.items() if k != 'name'}
    optimizer = optimizer_cls(pipeline._fusion_network.parameters(), **optimizer_params)

    scheduler = setup.get_scheduler(optimizer, config.TRAINING.scheduler)

    # define loss function
    criterion = setup.get_loss_function(config.TRAINING.loss, device)

    # load training checkpoint
    start_epoch = 0
    if config.TRAINING.resume:
        if os.path.isfile(config.TRAINING.resume):
            print('Loading model and optimizer from checkpoint {}'.format(config.TRAINING.resume))
            checkpoint = torch.load(config.TRAINING.resume, map_location=device)
            pipeline.load_state_dict(checkpoint['model_state'])
            pipeline.to(device)
            if config.DATA.semantics and config.DATA.semantic_strategy == 'predict':
                pipeline._semantic_2d_network.eval()
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            scheduler.load_state_dict(checkpoint['scheduler_state'])
            start_epoch = checkpoint['epoch']
        else:
            print('No checkpoint found at {}'.format(config.TRAINING.resume))

    # define some parameters
    n_batches = float(len(train_dataset) / config.TRAINING.train_batch_size)

    # evaluation metrics
    best_iou = 0.
    best_acc = 0.

    pipeline = pipeline.to(device)

    for epoch in range(start_epoch, config.TRAINING.n_epochs):

        workspace.log('Training epoch {}/{}'.format(epoch, config.TRAINING.n_epochs), mode='train')

        # resetting databases before each epoch starts
        train_database.reset()
        val_database.reset()

        train_loss = 0

        pipeline._fusion_network.train()

        for i, batch in tqdm(enumerate(train_loader), total=len(train_dataset), mininterval=30):

            if not torch.all(torch.isfinite(batch['extrinsics'])):
                continue

            # put all data on GPU
            batch = transform.to_device(batch, device)

            # reset the database for every new trajectory
            if batch['frame_id'][0].split('/')[-1] == '0' and config.DATA.data_load_strategy == 'hybrid':
                workspace.log('Starting new trajectory {} at step {}'.format(batch['frame_id'][0][:-2], i), mode='train')
                workspace.log('Resetting grid for scene {} at step {}'.format(batch['frame_id'][0].split('/')[0], i), mode='train')
                train_database.reset(batch['frame_id'][0].split('/')[0])

            if config.TRAINING.optimization.reset_strategy:
                if np.random.random_sample() <= config.TRAINING.optimization.reset_prob:
                    workspace.log('Resetting randomly trajectory {} at step {}'.format(batch['frame_id'][0][:-2], i), mode='train')
                    workspace.log('Resetting grid for scene {} at step {}'.format(batch['frame_id'][0].split('/')[0], i), mode='train')
                    train_database.reset(batch['frame_id'][0].split('/')[0])

            # fusion pipeline
            output = pipeline.fuse_training(batch, train_database, device)

            tsdf_fused = output['tsdf_fused']
            tsdf_target = output['tsdf_target']

            # optimization
            loss = criterion(tsdf_fused, tsdf_target)
            if loss.grad_fn: # this is needed because when the mono mask filters out all pixels, this results in a failure
                loss.backward()
                train_loss += loss.item() # note that this loss is a moving average over the training window of log_freq steps

            if (i + 1) % config.SETTINGS.log_freq == 0:
                train_loss = train_loss / config.SETTINGS.log_freq
                workspace.writer.add_scalar('Train/loss', train_loss, global_step=i + 1 + epoch*n_batches)
                train_loss = 0

            if config.TRAINING.optimization.clipping:
                torch.nn.utils.clip_grad_norm_(pipeline._fusion_network.parameters(), max_norm=1., norm_type=2)

            # accumulate gradients for stability
            if (i + 1) % config.TRAINING.optimization.accumulation_steps == 0 or i == n_batches - 1:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            if (i + 1) % config.SETTINGS.eval_freq == 0 or i == n_batches - 1:

                optimizer.zero_grad()
                val_database.reset()

                # train_database.filter(value=1.) # when should I do the filtering? Since I do random resetting, it does not
                # make sense to do outlier filtering during training
                train_database.to_numpy()
                train_eval = train_database.evaluate(mode='train', workspace=workspace)
                workspace.writer.add_scalar('Train/mse', train_eval['mse'], global_step=i + 1 + epoch*n_batches)
                workspace.writer.add_scalar('Train/acc', train_eval['acc'], global_step=i + 1 + epoch*n_batches)
                workspace.writer.add_scalar('Train/iou', train_eval['iou'], global_step=i + 1 + epoch*n_batches)
                workspace.writer.add_scalar('Train/mad', train_eval['mad'], global_step=i + 1 + epoch*n_batches)

                # validation step - fusion
                pipeline.eval()
                with torch.no_grad():
                    for batch in tqdm(val_loader, total=len(val_dataset), mininterval=30):
                        if not torch.all(torch.isfinite(batch['extrinsics'])): continue
                        # fusion pipeline
                        batch = transform.to_device(batch, device)

                        pipeline.fuse(batch, val_database, device)

                val_database.to_numpy()
                val_database.filter(value=0.5) # the more frames you integrate, the higher can the value be
                val_eval = val_database.evaluate(mode='val', workspace=workspace)
                workspace.writer.add_scalar('Val/mse', val_eval['mse'], global_step=i + 1 + epoch*n_batches)
                workspace.writer.add_scalar('Val/acc', val_eval['acc'], global_step=i + 1 + epoch*n_batches)
                workspace.writer.add_scalar('Val/iou', val_eval['iou'], global_step=i + 1 + epoch*n_batches)
                workspace.writer.add_scalar('Val/mad', val_eval['mad'], global_step=i + 1 + epoch*n_batches)

                # Note that the train models are saved s.t. the integration takes place during training so the 
                # initial frames are not integrated using the same fusion network as the later frames.
                # This means that this is not the same as integrating these frames during testing.

                # check if current checkpoint is best
                if (val_eval['iou'] + val_eval['acc']) / 2 >= best_iou:
                    best_iou = (val_eval['iou'] + val_eval['acc']) / 2
                    workspace.log('Found new best model with score {} at epoch {}'.format(best_iou, epoch), mode='val')
                    
                    val_database.save_to_workspace(workspace, mode='best_val', save_mode=config.SETTINGS.save_mode)

                    checkpoint_best = {
                        'epoch': epoch + 1,
                        'model_state': pipeline._fusion_network.state_dict(),
                        'best_iou': best_iou
                    }
                    workspace.save_model_state(checkpoint_best, is_best=True, name='best.pth.tar')

                # save database
                val_database.save_to_workspace(workspace, mode='latest_val', save_mode=config.SETTINGS.save_mode)

                # save checkpoint
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state': pipeline._fusion_network.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                }
                workspace.save_model_state(checkpoint, is_best=False)

                pipeline._fusion_network.train()
                train_database.to_torch()
                val_database.to_torch()


if __name__ == '__main__':

    args = arg_parser()
    print(args['comment'])
    
    # get configs
    config = loading.load_config_from_yaml(args['config'])

    train_fusion(args, config)
