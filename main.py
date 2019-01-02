import os
import sys
import argparse
import numpy as np
import time
import torch
from torch import nn

from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import settings
from dataset import QuickdrawDataset, load_data
from model import ResBackbone, Delf_classification, Resnet_classification

logger = settings.logger
torch.cuda.manual_seed_all(66)
torch.manual_seed(66)
os.environ['CUDA_VISIBLE_DEVICES'] = settings.device_id


def load_partly(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

def set_track_running_stats(model, new_state):
    for child in model.children():
        try:
            for ii in range(len(child)):
                if type(child[ii])==nn.BatchNorm2d:
                    child[ii].track_running_stats = new_state
                    print('set track_running_stats false')

                try:
                    little_father = child[ii]
                    for childchild in little_father.children():
                        if type(childchild)==nn.BatchNorm2d:
                            childchild[iii].track_running_stats = new_state
                            print('set track_running_stats false')
                except:
                    None
        except:
            None

class Session:
    def __init__(self):
        self.log_dir = settings.log_dir
        self.model_dir = settings.model_dir
        ensure_dir(settings.log_dir)
        ensure_dir(settings.model_dir)
        logger.info('set log dir as %s' % settings.log_dir)
        logger.info('set model dir as %s' % settings.model_dir)

        self.backbone = ResBackbone().cuda()
        self.classification_model = Delf_classification(settings.num_classes).cuda() if settings.classification_model == 'delf' else Resnet_classification(settings.num_classes).cuda()

        if settings.classification_model == 'delf':
            for para in self.backbone.parameters():
                para.requires_grad = False

        self.backbone = nn.DataParallel(self.backbone, device_ids=range(settings.num_gpu))
        self.classification_model = nn.DataParallel(self.classification_model, device_ids=range(settings.num_gpu))

        self.crit = nn.CrossEntropyLoss().cuda()

        self.epoch_count = 0
        self.step = 0
        self.save_steps = settings.save_steps
        self.num_workers = settings.num_workers
        self.batch_size = settings.batch_size
        self.writers = {}
        self.dataloaders = {}

        if settings.classification_model == 'delf':
            parameters = list(self.classification_model.parameters())
        elif settings.classification_model == 'res':
            parameters= list(self.backbone.parameters()) + list(self.classification_model.parameters())

        self.opt = Adam(parameters, lr=settings.lr, weight_decay= 1,amsgrad=True)
        self.sche = MultiStepLR(self.opt, milestones=settings.iter_sche, gamma=0.1)

    def tensorboard(self, name):
        self.writers[name] = SummaryWriter(os.path.join(self.log_dir, name + '.events'))
        return self.writers[name]

    def write(self, name, out):
        for k, v in out.items():
            self.writers[name].add_scalar(name + '/' + k, v, self.step)

        out['lr'] = self.opt.param_groups[0]['lr']
        out['step'] = self.step
        out['eooch_count'] = self.epoch_count
        outputs = [
            "{}:{:.4g}".format(k, v) 
            for k, v in out.items()
        ]
        logger.info(name + '--' + ' '.join(outputs))

    def get_dataloader(self, dataset_name, keyid_cat_catindex, words, use_iter=True):
        dataset = QuickdrawDataset(dataset_name, keyid_cat_catindex, words) 

        dataloader_this = \
                    DataLoader(dataset,
                               batch_size=self.batch_size,
                               shuffle=True,
                               num_workers=self.num_workers,
                               drop_last=False if dataset_name == 'test' else True)
        if use_iter:
            return iter(dataloader_this)
        else:
            return dataloader_this

    def save_checkpoints(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        obj = {
            'backbone': self.backbone.state_dict(),
            'classification_model': self.classification_model.state_dict(),
            'clock': self.step,
            'epoch_count': self.epoch_count,
            'opt': self.opt.state_dict(),
        }
        torch.save(obj, ckp_path)

    def load_checkpoints(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            obj = torch.load(ckp_path)
            print('load checkpoint: %s' %ckp_path)
        except FileNotFoundError:
            print('Find no checkpoint, reinitialize one!')
            return
        self.backbone.load_state_dict(obj['backbone'])
        
        self.classification_model.load_state_dict(obj['classification_model'])
        self.opt.load_state_dict(obj['opt'])
        self.step = obj['clock']
        self.epoch_count = obj['epoch_count']
        self.sche.last_epoch = self.step


    def load_checkpoints_delf_init(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        obj = torch.load(ckp_path)
        self.backbone.load_state_dict(obj['backbone'])

    def inf_batch(self, batch, return_correct=False):
        drawing, label = batch['drawing'], batch['word']
        drawing, label = drawing.cuda(), label.cuda()

        x = self.backbone(drawing)

        if settings.classification_model == 'res':
            pred = self.classification_model(x)
        elif settings.classification_model == 'delf':
            pred, attention_prob = self.classification_model(x)

        loss = self.crit(pred, label)

        _, pred_word = torch.max(pred, 1)
        total = len(label)
        correct = (pred_word == label).sum()
        accuracy = 100 * correct / total
        total = pred.shape[0]

        if return_correct == False:
            return loss, accuracy
        else:
            return loss, accuracy, correct, total


def run_train_val(ckp_name='res_latest' if settings.classification_model=='res' else 'delf_latest'):
    sess = Session()
    if settings.delf_init == False:
        sess.load_checkpoints(ckp_name)
    else:
        sess.load_checkpoints_delf_init(ckp_name)

    sess.tensorboard(settings.classification_model + '_train')
    sess.tensorboard(settings.classification_model + '_val')

    keyid_cat_catindex_train, words_train = load_data('train')
    dt_train = sess.get_dataloader('train', keyid_cat_catindex_train, words_train)
    keyid_cat_catindex_val, words_val = load_data('val')
    dt_val = sess.get_dataloader('val', keyid_cat_catindex_val, words_val)

    while sess.step < settings.iter_sche[-1]:
        sess.sche.step()
        sess.backbone.train() if settings.classification_model == 'res' else  None #sess.backbone.eval()
        sess.classification_model.train()
        sess.backbone.zero_grad() if settings.classification_model == 'res' else None
        sess.classification_model.zero_grad()

        try:
            batch_t = next(dt_train)
        except StopIteration:
            dt_train = sess.get_dataloader('train', keyid_cat_catindex_train, words_train)
            batch_t = next(dt_train)
            sess.epoch_count += 1
        loss_t, precision_t = sess.inf_batch(batch_t)
        sess.write(settings.classification_model + '_train', {'loss_t': loss_t, 'precision_t': precision_t})
        loss_t.backward()
        sess.opt.step()

        if sess.step % int(settings.latest_steps) == 0:
            sess.save_checkpoints(settings.classification_model + '_latest')
            sess.save_checkpoints(settings.classification_model + '_latest_backup')

        if sess.step % settings.val_step ==0:
            try:
                batch_v = next(dt_val)
            except StopIteration:
                dt_val = sess.get_dataloader('val', keyid_cat_catindex_val, words_val)
                batch_v = next(dt_val)
            loss_v, precision_v = sess.inf_batch(batch_v)
            sess.write(settings.classification_model + '_val', {'loss_v': loss_v, 'precision_v': precision_v})

        if sess.step % sess.save_steps == 0:
            sess.save_checkpoints(settings.classification_model + '_step_%d' % sess.step)
            logger.info('save model as step_%d' % sess.step)
        sess.step += 1


def run_test(ckp_name):
    sess = Session()
    sess.load_checkpoints(ckp_name)
    keyid_cat_catindex_test, words_test = load_data('test')
    dt = sess.get_dataloader('test', keyid_cat_catindex_test, words_test, use_iter=False)

    all_num = 0
    correct = 0

    for i, batch in enumerate(dt):
        loss, precision, batch_correct, batch_total = sess.inf_batch(batch, return_correct=True)
        batch_size = batch_total
        all_num += batch_size
        correct += batch_correct
        logger.info('batch %d loss: %f, precision: %f' % (i, loss, precision))

    logger.info('total loss: %f, precision: %f' % (correct / all_num))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--action', default='train')
    parser.add_argument('-m', '--model', default='latest')

    args = parser.parse_args(sys.argv[1:])

    if args.action == 'train':
        run_train_val(args.model)
    elif args.action == 'test':
        run_test(args.model)

