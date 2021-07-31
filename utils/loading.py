import yaml
import json
import os
import torch

from easydict import EasyDict


def load_config_from_yaml(path):
    """
    Method to load the config file for
    neural network training
    :param path: yaml-filepath with configs stored
    :return: easydict containing config
    """
    c = yaml.safe_load(open(path))
    config = EasyDict(c)

    return config


def load_config_from_json(path):
    """
    Method to load the config file
    from json files.
    :param path: path to json file
    :return: easydict containing config
    """
    with open(path, 'r') as file:
        data = json.load(file)
    config = EasyDict(data)
    return config


def load_experiment(path):
    """
    Method to load experiment from path
    :param path: path to experiment folder
    :return: easydict containing config
    """
    path = os.path.join(path, 'config.json')
    config = load_config_from_json(path)
    return config


def load_config(path):
    """
    Wrapper method around different methods
    loading config file based on file ending.
    """

    if path[-4:] == 'yaml':
        return load_config_from_yaml(path)
    elif path[-4:] == 'json':
        return load_config_from_json(path)
    else:
        raise ValueError('Unsupported file format for config')


def load_model(file, model):

    checkpoint = file

    if not os.path.exists(checkpoint):
        raise FileNotFoundError("File doesn't exist {}".format(checkpoint))
    try:
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint)
        else:
            checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
    except:
        print('loading model partly')
        pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model.state_dict()}
        model.state_dict().update(pretrained_dict)
        model.load_state_dict(model.state_dict())

def load_pipeline(file, model, name=''):

    print("Using pre-trained pipeline {}".format(file)) 
    checkpoint = file

    if not os.path.exists(checkpoint):
        raise FileNotFoundError("File doesn't exist {}".format(checkpoint))

    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint)
    else:
        checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))

    try:
        print("Loading entire pipeline from checkpoint: {}.".format(file))
        model.load_state_dict(checkpoint['model_state'])
    except:
        # model params and checkpoint params may have a different reference path
        # e.g. if the model is a part of the loaded pipeline

        model_keys = model.state_dict().keys()

        # get keys of model parameters with the checkpoint reference
        pretrained_keys = []
        for k in model_keys:
            for check_k in checkpoint['model_state'].keys():
                if name + '.' + k in check_k:
                    pretrained_keys.append(check_k)
                    break

        # get difference between checkpoint reference and model
        idx = pretrained_keys[0].split('.').index(list(model_keys)[0].split('.')[0]) - 1

        pretrained_dict = {k: v for k, v in checkpoint['model_state'].items() if k in pretrained_keys}
        main_keys = {k.split('.')[idx] for k in pretrained_dict.keys()}
        print("Loaded only {} from pipeline.".format(main_keys))
        model.state_dict().update(pretrained_dict)
        model.load_state_dict(model.state_dict())


def load_weights_partial(file, model):
    checkpoint = file
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint)
    else:
        checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))

    pretrained_dict = {k: v for k, v in checkpoint['model_state'].items() if k in model.state_dict()}
    main_keys = {k.split('.')[0] for k in pretrained_dict.keys()}
    print("Loading {} from pipeline.".format(main_keys))
    model.state_dict().update(pretrained_dict)
    model.load_state_dict(model.state_dict())   


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path.
    If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise FileNotFoundError("File doesn't exist {}".format(checkpoint))
    try:
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint)
        else:
            checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
    except:
        print('loading model partly')
        pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model.state_dict()}
        model.state_dict().update(pretrained_dict)
        model.load_state_dict(model.state_dict())

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def separate_pipeline(pipeline_path):
    checkpoint = torch.load(pipeline_path, map_location=torch.device('cpu'))

    main_keys = {k.split('.')[0] for k in checkpoint['model_state'].keys()}

    if '_fusion_network' in main_keys:
        separate = True
    else:
        separate = False

    if separate:
        # Save whole pipeline in another dir
        outdir, file = os.path.split(pipeline_path)
        path = os.path.join(outdir, 'pipeline', file)

        if not os.path.exists(os.path.dirname(path)):
            os.mkdir(os.path.dirname(path))

        torch.save(checkpoint, path)

        model_state = checkpoint['model_state']
        model_state = {k.replace('_fusion_network.', ''): v for k, v in model_state.items() if k.startswith('_fusion_network.')}

        checkpoint['model_state'] = model_state
        checkpoint.pop('pipeline_state', None)
        torch.save(checkpoint, pipeline_path)
        print("Extracted fusion model from pipeline and saved separately.")


def select_child(state_dict, string):
    if string[-1] != '.':
        string = string + '.'
    # select only keys containing the string and remove it --> select child module
    return {k.replace(string, ''): v for k, v in state_dict.items() if k.startswith(string)}


def remove_parent(state_dict, string):
    if string[-1] != '.':
        string = string + '.'
    # remove string only in keys containing it --> remove parent module
    return {(k.replace(string, '') if k.startswith(string) else k): v for k, v in state_dict.items()}