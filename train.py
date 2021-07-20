#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han modified by @indonoso'

import argparse
import json
import logging
import os

import torch
import torch.optim as optim
from tqdm import tqdm

from machine_comprehension.dataset.squad_dataset import SquadDataset
from machine_comprehension.models.loss import MyNLLLoss
from machine_comprehension.utils.eval import eval_on_model
from machine_comprehension.utils.load_config import init_logging, read_config
from .model import MatchLSTMModified
from .utils import get_input_size, get_embeddings

logger = logging.getLogger(__name__)


def train(config_path):
    logger.info('------------MODEL TRAIN--------------')
    logger.info('loading config file...')
    global_config = read_config(config_path)

    # set random seed
    seed = global_config['global']['random_seed']
    torch.manual_seed(seed)

    enable_cuda = global_config['train']['enable_cuda']
    device = torch.device("cuda" if enable_cuda else "cpu")
    if torch.cuda.is_available() and not enable_cuda:
        logger.warning("CUDA is avaliable, you can enable CUDA in config file")
    elif not torch.cuda.is_available() and enable_cuda:
        raise ValueError("CUDA is not abaliable, please unable CUDA in config file")

    logger.info('reading squad dataset...')
    dataset = SquadDataset(global_config)

    logger.info('constructing model...')

    network_input_size = global_config['preprocess']['word_embedding_size'] + \
                         get_input_size('kg', global_config['preprocess']) + \
                         get_input_size('pos', global_config['preprocess'])

    model = MatchLSTMModified(device, network_input_size, global_config['model']['match_lstm_input_size'],
                              hidden_size=global_config['model']['hidden_size'],
                              word_embedding=get_embeddings('word', global_config),
                              part_of_speech=get_embeddings('pos', global_config),
                              knowledge_graph=get_embeddings('kg', global_config))
    model = model.to(device)
    criterion = MyNLLLoss()

    # optimizer
    optimizer_choose = global_config['train']['optimizer']
    optimizer_lr = global_config['train']['learning_rate']
    optimizer_param = filter(lambda p: p.requires_grad, model.parameters())

    if optimizer_choose == 'adamax':
        optimizer = optim.Adamax(optimizer_param)
    elif optimizer_choose == 'adadelta':
        optimizer = optim.Adadelta(optimizer_param)
    elif optimizer_choose == 'adam':
        optimizer = optim.Adam(optimizer_param)
    elif optimizer_choose == 'sgd':
        optimizer = optim.SGD(optimizer_param, lr=optimizer_lr)
    else:
        raise ValueError('optimizer "%s" in config file not recoginized' % optimizer_choose)

    # check if exist model weight
    weight_path = global_config['data']['model_path']
    model, init_epoch = load_weights_if_available(model, weight_path, enable_cuda)

    # training arguments
    logger.info('start training...')
    train_batch_size = global_config['train']['batch_size']
    valid_batch_size = global_config['train']['valid_batch_size']

    num_workers = global_config['global']['num_data_workers']
    batch_train_data = dataset.get_dataloader_train(train_batch_size, num_workers)
    batch_dev_data = dataset.get_dataloader_dev(valid_batch_size, num_workers)

    clip_grad_max = global_config['train']['clip_grad_norm']

    best_avg = 0.
    model = model.to(device)
    # every epoch
    apply_clip = True
    last_loss = -1
    for epoch in range(init_epoch + 1, global_config['train']['epoch']):
        # train
        model.train()  # set training = True, make sure right dropout
        sum_loss = train_on_model(model=model,
                                  criterion=criterion,
                                  optimizer=optimizer,
                                  batch_data=batch_train_data,
                                  epoch=epoch,
                                  clip_grad_max=clip_grad_max,
                                  device=device, apply_clip=apply_clip)
        logger.info('epoch=%d, sum_loss=%.5f' % (epoch, sum_loss))
        # apply_clip = last_loss != -1 and (sum_loss - last_loss)/sum_loss < 0.15
        last_loss = sum_loss

        # evaluate
        with torch.no_grad():
            model.eval()  # let training = False, make sure right dropout
            valid_score_em, valid_score_f1, valid_loss = eval_on_model(model=model,
                                                                       criterion=criterion,
                                                                       batch_data=batch_dev_data,
                                                                       epoch=epoch,
                                                                       device=device)
            valid_avg = (valid_score_em + valid_score_f1) / 2
        logger.info("epoch=%d, ave_score_em=%.2f, ave_score_f1=%.2f, sum_loss=%.5f" %
                    (epoch, valid_score_em, valid_score_f1, valid_loss))

        save_metrics(checkpoint_path=global_config['data']['checkpoint_path'], metrics={'train_loss': sum_loss,
                                                                                        'validation_f1': valid_score_f1,
                                                                                        'validation_em': valid_score_em,
                                                                                        'epoch': epoch,
                                                                                        'apply_clip': apply_clip}, epoch=epoch)
        # save model when best avg score
        if valid_avg > best_avg:
            save_model(model,
                       epoch=epoch,
                       model_weight_path=global_config['data']['model_path'])
            logger.info("saving model weight on epoch=%d" % epoch)
            best_avg = valid_avg
    save_model(model, epoch=epoch,
               model_weight_path=global_config['data']['model_path'])
    logger.info('finished.')


def train_on_model(model, criterion, optimizer, batch_data, epoch, clip_grad_max, device, apply_clip):
    """
    train on every batch
    :param enable_char:
    :param batch_char_func:
    :param model:
    :param criterion:
    :param batch_data:
    :param optimizer:
    :param epoch:
    :param clip_grad_max:
    :param device:
    :return:
    """
    batch_cnt = len(batch_data)
    sum_loss = 0.
    for i, batch in enumerate(tqdm(batch_data, desc=f'Training epoch {epoch}', position=0, leave=True)):
        optimizer.zero_grad()

        # batch data
        batch = [x.to(device) if x is not None else x for x in batch]
        bat_answer_range = batch[-1]

        # forward
        batch_input = batch[:-1]
        ans_range_prop, _, _ = model.forward(*batch_input)

        # get loss
        loss = criterion.forward(ans_range_prop, bat_answer_range)
        loss.backward()
        if apply_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_max)  # fix gradient explosion
        optimizer.step()  # update parameters

        # logging
        batch_loss = loss.item()
        sum_loss += batch_loss * bat_answer_range.shape[0]

        # logger.info('epoch=%d, batch=%d/%d, loss=%.5f' % (epoch, i, batch_cnt, batch_loss))

        # manual release memory
        del batch, ans_range_prop, loss
        if device == 'cuda':
            torch.cuda.empty_cache()

    return sum_loss


def save_model(model, epoch, model_weight_path):
    """
    save model weight without embedding
    :param model:
    :param epoch:
    :param model_weight_path:
    :param checkpoint_path:
    :return:
    """
    # save model weight
    model_weight = model.state_dict()

    torch.save(model_weight, f"{model_weight_path}-{epoch}.pt")


def save_metrics(metrics, epoch, checkpoint_path):
    with open(f"{checkpoint_path}-{epoch}.json", 'w') as checkpoint_f:
        json.dump(metrics, checkpoint_f)


def load_weights_if_available(model, model_weight_path, use_cuda):
    folder = os.path.dirname(model_weight_path)
    filename = os.path.basename(model_weight_path)
    epochs = [int(p.split('-')[-1][:-3]) for p in os.listdir(folder) if p.startswith(filename)]
    max_epoch = 0
    if len(epochs) > 0:
        logger.info('loading existing weight...')
        max_epoch = max(epochs)
        weight = torch.load(f"{model_weight_path}-{max_epoch}.pt", map_location=lambda storage, loc: storage)
        if use_cuda:
            weight = torch.load(f"{model_weight_path}-{max_epoch}.pt", map_location=lambda storage, loc: storage.cuda())
        model.load_state_dict(weight, strict=False)

    return model, max_epoch


if __name__ == '__main__':
    init_logging()

    parser = argparse.ArgumentParser(description="train on the model")
    parser.add_argument('--config', '-c', required=False, dest='config_path', default='train_config.yaml')
    args = parser.parse_args()

    train(args.config_path)
