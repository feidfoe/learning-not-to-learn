# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import models

import time
import os
import math

from utils import logger_setting, Timer


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * 0.1

def grad_reverse(x):
    return GradReverse.apply(x)



class Trainer(object):
    def __init__(self, option):
        self.option = option

        self._build_model()
        self._set_optimizer()
        self.logger = logger_setting(option.exp_name, option.save_dir, option.debug)

    def _build_model(self):
        self.n_color_cls = 8

        self.net = models.convnet(num_classes=self.option.n_class)
        self.pred_net_r = models.Predictor(input_ch=32, num_classes=self.n_color_cls)
        self.pred_net_g = models.Predictor(input_ch=32, num_classes=self.n_color_cls)
        self.pred_net_b = models.Predictor(input_ch=32, num_classes=self.n_color_cls)

        self.loss = nn.CrossEntropyLoss(ignore_index=255)
        self.color_loss = nn.CrossEntropyLoss(ignore_index=255)


        if self.option.cuda:
            self.net.cuda()
            self.pred_net_r.cuda()
            self.pred_net_g.cuda()
            self.pred_net_b.cuda()
            self.loss.cuda()
            self.color_loss.cuda()

    def _set_optimizer(self):
        self.optim = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.option.lr, momentum=self.option.momentum, weight_decay=self.option.weight_decay)
        self.optim_r = optim.SGD(self.pred_net_r.parameters(), lr=self.option.lr, momentum=self.option.momentum, weight_decay=self.option.weight_decay)
        self.optim_g = optim.SGD(self.pred_net_g.parameters(), lr=self.option.lr, momentum=self.option.momentum, weight_decay=self.option.weight_decay)
        self.optim_b = optim.SGD(self.pred_net_b.parameters(), lr=self.option.lr, momentum=self.option.momentum, weight_decay=self.option.weight_decay)

        #TODO: last_epoch should be the last step of loaded model
        lr_lambda = lambda step: self.option.lr_decay_rate ** (step // self.option.lr_decay_period)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lr_lambda, last_epoch=-1)
        self.scheduler_r = optim.lr_scheduler.LambdaLR(self.optim_r, lr_lambda=lr_lambda, last_epoch=-1)
        self.scheduler_g = optim.lr_scheduler.LambdaLR(self.optim_g, lr_lambda=lr_lambda, last_epoch=-1)
        self.scheduler_b = optim.lr_scheduler.LambdaLR(self.optim_b, lr_lambda=lr_lambda, last_epoch=-1)

    @staticmethod
    def _weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def _initialization(self):
        self.net.apply(self._weights_init)

        if self.option.is_train:
            model_dict = self.net.state_dict()
            if self.option.use_pretrain:
                pretrained_state = torch.load('pretrained/seed%d/%.03fmnist.pth'%(self.option.seed,self.option.color_var))['net_state_dict']
                self.logger.info('pretrained/seed%d/%.03fmnist.pth'%(self.option.seed,self.option.color_var))

                #if len(self.option.gpu_ids) > 1:
                #    module_pretrained_state = {'module.video_%s'%k:v for k,v in pretrained_state.items()}
                #else:
                #    module_pretrained_state = {k:v for k,v in pretrained_state.items()}
                #pretrained_state = {k:v for k,v in module_pretrained_state.items() if k in model_dict}

                for k in pretrained_state.keys():
                    msg = "Weight named %s resumed"%k
                    self.logger.info(msg)

                model_dict.update(pretrained_state)
                self.net.load_state_dict(model_dict)



    def _mode_setting(self, is_train=True):
        if is_train:
            self.net.train()
            self.pred_net_r.train()
            self.pred_net_g.train()
            self.pred_net_b.train()
        else:
            self.net.eval()
            self.pred_net_r.eval()
            self.pred_net_g.eval()
            self.pred_net_b.eval()

    def _train_step(self, data_loader, step):
        _lambda = 0.01


        for i, (images,color_labels,labels) in enumerate(data_loader):
            
            start_time = time.time()
            images = self._get_variable(images)
            color_labels = self._get_variable(color_labels)
            labels = self._get_variable(labels)

            self.optim.zero_grad()
            self.optim_r.zero_grad()
            self.optim_g.zero_grad()
            self.optim_b.zero_grad()
            feat_label, pred_label = self.net(images)


            # predict colors from feat_label. Their prediction should be uniform.
            _,pseudo_pred_r = self.pred_net_r(feat_label)
            _,pseudo_pred_g = self.pred_net_g(feat_label)
            _,pseudo_pred_b = self.pred_net_b(feat_label)


            # loss for self.net
            loss_pred = self.loss(pred_label, torch.squeeze(labels))

            loss_pseudo_pred_r = torch.mean(torch.sum(pseudo_pred_r*torch.log(pseudo_pred_r),1))
            loss_pseudo_pred_g = torch.mean(torch.sum(pseudo_pred_g*torch.log(pseudo_pred_g),1))
            loss_pseudo_pred_b = torch.mean(torch.sum(pseudo_pred_b*torch.log(pseudo_pred_b),1))

            loss_pred_ps_color = (loss_pseudo_pred_r + loss_pseudo_pred_g + loss_pseudo_pred_b) / 3.

            loss = loss_pred + loss_pred_ps_color*_lambda

            loss.backward()
            self.optim.step()


            self.optim.zero_grad()
            self.optim_r.zero_grad()
            self.optim_g.zero_grad()
            self.optim_b.zero_grad()

            feat_label, pred_label = self.net(images)

            feat_color = grad_reverse(feat_label)
            pred_r,_ = self.pred_net_r(feat_color)
            pred_g,_ = self.pred_net_g(feat_color)
            pred_b,_ = self.pred_net_b(feat_color)

            # loss for rgb predictors
            loss_pred_r = self.color_loss(pred_r, color_labels[:,0])
            loss_pred_g = self.color_loss(pred_g, color_labels[:,1])
            loss_pred_b = self.color_loss(pred_b, color_labels[:,2])

            loss_pred_color = loss_pred_r + loss_pred_g + loss_pred_b

            loss_pred_color.backward()
            self.optim.step()
            self.optim_r.step()
            self.optim_g.step()
            self.optim_b.step()

            single_iter_time = time.time() - start_time
            # TODO: print elapsed time for iteration
            if i % self.option.log_step == 0:
                msg = "[TRAIN] cls loss : %.6f, rgb : %.6f, MI : %.6f  (epoch %d.%02d)" \
                       % (loss_pred,loss_pred_color/3.,loss_pred_ps_color,step,int(100*i/data_loader.__len__()))
                self.logger.info(msg)


    def _validate(self, data_loader):
        self._mode_setting(is_train=False)
        self._initialization()
        if self.option.checkpoint is not None:
            self._load_model()
        else:
            print("No trained model for evaluation provided")
            import sys
            sys.exit()

        num_test = 10000

        total_num_correct = 0.
        total_num_test = 0.
        total_loss = 0.
        for i, (images,color_labels,labels) in enumerate(data_loader):
            
            start_time = time.time()
            images = self._get_variable(images)
            colro_labels = self._get_variable(color_labels)
            labels = self._get_variable(labels)

            self.optim.zero_grad()
            _, pred_label = self.net(images)


            loss = self.loss(pred_label, torch.squeeze(labels))
            
            batch_size = images.shape[0]
            total_num_correct += self._num_correct(pred_label,labels,topk=1).data[0]
            total_loss += loss.data[0]*batch_size
            total_num_test += batch_size
               
        avg_loss = total_loss/total_num_test
        avg_acc = total_num_correct/total_num_test
        msg = "EVALUATION LOSS  %.4f, ACCURACY : %.4f (%d/%d)" % \
                        (avg_loss,avg_acc,int(total_num_correct),total_num_test)
        self.logger.info(msg)



    def _num_correct(self,outputs,labels,topk=1):
        _, preds = outputs.topk(k=topk, dim=1)
        preds = preds.t()
        correct = preds.eq(labels.view(1, -1).expand_as(preds))
        correct = correct.view(-1).sum()
        return correct
        


    def _accuracy(self, outputs, labels):
        batch_size = labels.size(0)
        _, preds = outputs.topk(k=1, dim=1)
        preds = preds.t()
        correct = preds.eq(labels.view(1, -1).expand_as(preds))
        correct = correct.view(-1).float().sum(0, keepdim=True)
        accuracy = correct.mul_(100.0 / batch_size)
        return accuracy

    def _save_model(self, step):
        torch.save({
            'step': step,
            'optim_state_dict': self.optim.state_dict(),
            'net_state_dict': self.net.state_dict()
        }, os.path.join(self.option.save_dir,self.option.exp_name, 'checkpoint_step_%04d.pth' % step))
        print('checkpoint saved. step : %d'%step)

    def _load_model(self):
        ckp_path = os.path.join(self.option.save_dir,self.option.exp_name,'checkpoint_step_%04d.pth'%self.option.checkpoint)
        ckpt = torch.load(ckp_path)
        assert ckpt['step'] == self.option.checkpoint
        self.net.load_state_dict(ckpt['net_state_dict'])
        self.optim.load_state_dict(ckpt['optim_state_dict'])

    def train(self, train_loader, val_loader=None):
        self._initialization()
        if self.option.checkpoint is not None:
            self._load_model()

        self._mode_setting(is_train=True)
        timer = Timer(self.logger, self.option.max_step)
        if self.option.checkpoint is None:
            start_epoch = 0
        else:
            start_epoch = self.option.checkpoint
        for step in range(start_epoch+1, self.option.max_step+1):
            #data = next(train_loader)
            self._train_step(train_loader,step)
            self.scheduler.step()
            self.scheduler_r.step()
            self.scheduler_g.step()
            self.scheduler_b.step()
            timer()

            if step == 1 or step % self.option.save_step == 0:
                if val_loader is not None:
                    self._validate(step, val_loader)
                self._save_model(step)


    def _get_variable(self, inputs):
        if self.option.cuda:
            return Variable(inputs.cuda())
        return Variable(inputs)
