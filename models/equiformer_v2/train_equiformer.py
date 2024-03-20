# %%


import ast
import json
import argparse
import logging
import time
import torch
import numpy as np
import os
from accelerate import Accelerator

from pathlib import Path
from typing  import Iterable, Optional

import torch_geometric

from torch_geometric.loader import DataLoader
from torch_cluster import radius_graph

from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
from timm.optim.nadam import Nadam
from timm.optim.novograd import NovoGrad
from timm.optim.nvnovograd import NvNovoGrad
from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP
from timm.optim.adabelief import AdaBelief
from timm.utils import ModelEmaV2, get_state_dict, dispatch_clip_grad
from timm.scheduler import create_scheduler

from nets import EquiformerV2_OC20

from mace_data.hdf5_dataset import HDF5Dataset
from mace_data.tools import get_atomic_number_table_from_zs


ModelEma = ModelEmaV2

# %%

def get_dataloaders(args):

    with open(f'data-r{args.radius}/statistics.json', "r") as f:
        statistics = json.load(f)

    zs_list = ast.literal_eval(statistics["atomic_numbers"])
    z_table = get_atomic_number_table_from_zs(zs_list)


    train_set = HDF5Dataset(
        f"data-r{args.radius}/train.h5", r_max=args.radius, z_table=z_table
    )
    valid_set = HDF5Dataset(
        f"data-r{args.radius}/valid.h5", r_max=args.radius, z_table=z_table
    )

    train_loader = torch_geometric.loader.DataLoader(
        dataset=train_set,
        batch_size=10,
        shuffle=True,
        drop_last=False,
        pin_memory=False,
        num_workers=10,
    )
    valid_loader = torch_geometric.loader.DataLoader(
        dataset=valid_set,
        batch_size=1,
        shuffle=True,
        drop_last=False,
        pin_memory=False,
        num_workers=10,
    )
    return train_loader, valid_loader, valid_loader

# %%

class FileLogger:
    def __init__(self, is_master=False, is_rank0=False, output_dir=None, logger_name='training'):
        # only call by master 
        # checked outside the class
        self.output_dir = output_dir
        if is_rank0:
            self.logger_name = logger_name
            self.logger = self.get_logger(output_dir, log_to_file=is_master)
        else:
            self.logger_name = None
            self.logger = NoOp()
        
        
    def get_logger(self, output_dir, log_to_file):
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')

        if output_dir and log_to_file:
            
            time_formatter = logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(message)s')
            debuglog = logging.FileHandler(output_dir+'/debug.log')
            debuglog.setLevel(logging.DEBUG)
            debuglog.setFormatter(time_formatter)
            logger.addHandler(debuglog)

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(logging.DEBUG)
        logger.addHandler(console)
        
        # Reference: https://stackoverflow.com/questions/21127360/python-2-7-log-displayed-twice-when-logging-module-is-used-in-two-python-scri
        logger.propagate = False

        return logger

    def console(self, *args):
        self.logger.debug(*args)

    def event(self, *args):
        self.logger.warn(*args)

    def verbose(self, *args):
        self.logger.info(*args)

    def info(self, *args):
        self.logger.info(*args)

# %%

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def evaluate(model, norm_factor, target, data_loader, amp_autocast=None, 
    print_freq=100, logger=None):
    
    model.eval()
    
    loss_metric = AverageMeter()
    mae_metric = AverageMeter()
    criterion = torch.nn.L1Loss()
    criterion.eval()
    
    task_mean = norm_factor[0] #model.task_mean
    task_std  = norm_factor[1] #model.task_std
    
    with torch.no_grad():
            
        for data in data_loader:
            #data.edge_d_index = radius_graph(data.pos, r=10.0, batch=data.batch, loop=True)
            #data.edge_d_attr = data.edge_attr
            
            with amp_autocast():
                pred = model(f_in=data.x, pos=data.pos, batch=data.batch, 
                    node_atom=data.z,
                    edge_d_index=data.edge_d_index, edge_d_attr=data.edge_d_attr)
                pred = pred.squeeze()
            
            loss = criterion(pred, (data.y[:, target] - task_mean) / task_std)
            loss_metric.update(loss.item(), n=pred.shape[0])
            err = pred.detach() * task_std + task_mean - data.y[:, target]
            mae_metric.update(torch.mean(torch.abs(err)).item(), n=pred.shape[0])
        
    return mae_metric.avg, loss_metric.avg


def compute_stats(data_loader, max_radius, logger, print_freq=1000):
    '''
        Compute mean of numbers of nodes and edges
    '''
    log_str = '\nCalculating statistics with '
    log_str = log_str + 'max_radius={}\n'.format(max_radius)
    logger.info(log_str)
        
    avg_node = AverageMeter()
    avg_edge = AverageMeter()
    avg_degree = AverageMeter()
    
    for step, data in enumerate(data_loader):
        
        pos = data.pos
        batch = data.batch
        edge_src, edge_dst = radius_graph(pos, r=max_radius, batch=batch,
            max_num_neighbors=1000)
        batch_size = float(batch.max() + 1)
        num_nodes = pos.shape[0]
        num_edges = edge_src.shape[0]
        num_degree = torch_geometric.utils.degree(edge_src, num_nodes)
        num_degree = torch.sum(num_degree)
            
        avg_node.update(num_nodes / batch_size, batch_size)
        avg_edge.update(num_edges / batch_size, batch_size)
        avg_degree.update(num_degree / (num_nodes), num_nodes)
            
        if step % print_freq == 0 or step == (len(data_loader) - 1):
            log_str = '[{}/{}]\tavg node: {}, '.format(step, len(data_loader), avg_node.avg)
            log_str += 'avg edge: {}, '.format(avg_edge.avg)
            log_str += 'avg degree: {}, '.format(avg_degree.avg)
            logger.info(log_str)

#%%
def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (name.endswith(".bias") or name.endswith(".affine_weight")  
            or name.endswith(".affine_bias") or name.endswith('.mean_shift')
            or 'bias.' in name 
            or name in skip_list):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def optimizer_kwargs(cfg):
    """ cfg/argparse to kwargs helper
    Convert optimizer args in argparse args or cfg like object to keyword args for updated create fn.
    """
    kwargs = dict(
        optimizer_name=cfg.opt,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        momentum=cfg.momentum)
    if getattr(cfg, 'opt_eps', None) is not None:
        kwargs['eps'] = cfg.opt_eps
    if getattr(cfg, 'opt_betas', None) is not None:
        kwargs['betas'] = cfg.opt_betas
    if getattr(cfg, 'opt_args', None) is not None:
        kwargs.update(cfg.opt_args)
    return kwargs


def create_optimizer(args, model, filter_bias_and_bn=True):
    """ Legacy optimizer factory for backwards compatibility.
    NOTE: Use create_optimizer_v2 for new code.
    """
    return create_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=args),
        filter_bias_and_bn=filter_bias_and_bn,
    )


def create_optimizer_v2(
        model: torch.nn.Module,
        optimizer_name: str = 'sgd',
        learning_rate: Optional[float] = None,
        weight_decay: float = 0.,
        momentum: float = 0.9,
        filter_bias_and_bn: bool = True,
        **kwargs):
    """ Create an optimizer.

    TODO currently the model is passed in and all parameters are selected for optimization.
    For more general use an interface that allows selection of parameters to optimize and lr groups, one of:
      * a filter fn interface that further breaks params into groups in a weight_decay compatible fashion
      * expose the parameters interface and leave it up to caller

    Args:
        model (nn.Module): model containing parameters to optimize
        optimizer_name: name of optimizer to create
        learning_rate: initial learning rate
        weight_decay: weight decay to apply in optimizer
        momentum:  momentum for momentum based optimizers (others may use betas via kwargs)
        filter_bias_and_bn:  filter out bias, bn and other 1d params from weight decay
        **kwargs: extra optimizer specific kwargs to pass through

    Returns:
        Optimizer
    """
    opt_lower = optimizer_name.lower()
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = add_weight_decay(model, weight_decay, skip)
        weight_decay = 0.
    else:
        parameters = model.parameters()
    #if 'fused' in opt_lower:
    #    assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=learning_rate, weight_decay=weight_decay, **kwargs)
    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = torch.optim.SGD(parameters, momentum=momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = torch.optim.SGD(parameters, momentum=momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = torch.optim.Adam(parameters, **opt_args) 
    elif opt_lower == 'adabelief':
        optimizer = AdaBelief(parameters, rectify=False, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = torch.optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam':
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamp':        
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'sgdp':
        optimizer = SGDP(parameters, momentum=momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = torch.optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not learning_rate:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = torch.optim.RMSprop(parameters, alpha=0.9, momentum=momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=momentum, **opt_args)
    elif opt_lower == 'novograd':
        optimizer = NovoGrad(parameters, **opt_args)
    elif opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    #elif opt_lower == 'fusedsgd':
    #    opt_args.pop('eps', None)
    #    optimizer = FusedSGD(parameters, momentum=momentum, nesterov=True, **opt_args)
    #elif opt_lower == 'fusedmomentum':
    #    opt_args.pop('eps', None)
    #    optimizer = FusedSGD(parameters, momentum=momentum, nesterov=False, **opt_args)
    #elif opt_lower == 'fusedadam':
    #    optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    #elif opt_lower == 'fusedadamw':
    #    optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    #elif opt_lower == 'fusedlamb':
    #    optimizer = FusedLAMB(parameters, **opt_args)
    #elif opt_lower == 'fusednovograd':
    #    opt_args.setdefault('betas', (0.95, 0.98))
    #    optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer

# %%

def get_args_parser():
    parser = argparse.ArgumentParser('Training equivariant networks', add_help=False)
    parser.add_argument('--output-dir', type=str, default='result')
    parser.add_argument('--radius', type=float, default=4.5)
    # training hyper-parameters
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=24)
    parser.add_argument('--model-ema', action='store_true')
    parser.set_defaults(model_ema=False)
    parser.add_argument('--model-ema-decay', type=float, default=0.9999, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')
    # regularization
    parser.add_argument('--drop-path', type=float, default=0.0)
    # optimizer (timm)
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-3,
                        help='weight decay (default: 5e-3)')
    # learning rate schedule parameters (timm)
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    # logging
    parser.add_argument("--print-freq", type=int, default=100)
    # task and dataset
    parser.add_argument("--target", type=str, default='aspirin')
    parser.add_argument("--data-path", type=str, default='datasets/md17')
    parser.add_argument("--train-size", type=int, default=950)
    parser.add_argument("--val-size", type=int, default=50)
    parser.add_argument('--compute-stats', action='store_true', dest='compute_stats')
    parser.set_defaults(compute_stats=False)
    parser.add_argument('--test-interval', type=int, default=10, 
                        help='epoch interval to evaluate on the testing set')
    parser.add_argument('--test-max-iter', type=int, default=1000, 
                        help='max iteration to evaluate on the testing set')
    parser.add_argument('--energy-weight', type=float, default=0.2)
    parser.add_argument('--force-weight', type=float, default=0.8)
    # random
    parser.add_argument("--seed", type=int, default=1)
    # data loader config
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    # evaluation
    parser.add_argument('--checkpoint-path', type=str, default=None)
    parser.add_argument('--evaluate', action='store_true', dest='evaluate')
    parser.set_defaults(evaluate=False)
    return parser


# from https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/modules/loss.py#L7
class L2MAELoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        assert reduction in ["mean", "sum"]

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        dists = torch.norm(input - target, p=2, dim=-1)
        if self.reduction == "mean":
            return torch.mean(dists)
        elif self.reduction == "sum":
            return torch.sum(dists)


def main(args):
    
    _log = FileLogger(is_master=True, is_rank0=True, output_dir=args.output_dir)
    _log.info(args)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # since dataset needs random 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    accelerator = Accelerator()

    ''' Network '''
    model = EquiformerV2_OC20(
        # First three arguments are not used
        None, None, None,
        max_radius=args.radius,
        max_num_elements=95)
    _log.info(model)

    if args.checkpoint_path is not None:
        state_dict = torch.load(args.checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict['state_dict'])
    
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay)
        model_ema = accelerator.prepare(model_ema)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log.info('Number of params: {}'.format(n_parameters))
    
    ''' Optimizer and LR Scheduler '''
    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    criterion = L2MAELoss() #torch.nn.L1Loss()  #torch.nn.MSELoss() # torch.nn.L1Loss() 

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    ''' Data Loader '''
    train_loader, val_loader, test_loader = get_dataloaders(args)
    train_loader, val_loader, test_loader = accelerator.prepare(train_loader, val_loader, test_loader)

    ''' Compute stats '''
    if args.compute_stats:
        compute_stats(train_loader, max_radius=args.radius, logger=_log, print_freq=args.print_freq)
        return
    
    # record the best validation and testing errors and corresponding epochs
    best_metrics = {'val_epoch': 0, 'test_epoch': 0, 
        'val_force_err': float('inf'),  'val_energy_err': float('inf'), 
        'test_force_err': float('inf'), 'test_energy_err': float('inf')}
    best_ema_metrics = {'val_epoch': 0, 'test_epoch': 0, 
        'val_force_err': float('inf'),  'val_energy_err': float('inf'), 
        'test_force_err': float('inf'), 'test_energy_err': float('inf')}

    if args.evaluate:
        test_err, test_loss = evaluate(args=args, model=model, criterion=criterion, 
            data_loader=test_loader,
            print_freq=args.print_freq, logger=_log, print_progress=True, max_iter=-1)
        return

    for epoch in range(args.epochs):
        
        epoch_start_time = time.perf_counter()
        
        lr_scheduler.step(epoch)
        
        train_err, train_loss = train_one_epoch(args=args, model=model, accelerator=accelerator, criterion=criterion,
            data_loader=train_loader, optimizer=optimizer,
            epoch=epoch, model_ema=model_ema,
            print_freq=args.print_freq, logger=_log)
        
        val_err, val_loss = evaluate(args=args, model=model, criterion=criterion, 
            data_loader=val_loader,
            print_freq=args.print_freq, logger=_log, print_progress=False)
        
        if (epoch + 1) % args.test_interval == 0:
            test_err, test_loss = evaluate(args=args, model=model, criterion=criterion, 
            data_loader=test_loader,
            print_freq=args.print_freq, logger=_log, print_progress=True, max_iter=args.test_max_iter)
        else:
            test_err, test_loss = None, None

        # Only main process should save model
        if accelerator.process_index == 0:

            update_val_result, update_test_result = update_best_results(args, best_metrics, val_err, test_err, epoch)
            if update_val_result:
                torch.save(
                    {'state_dict': model.state_dict()}, 
                    os.path.join(args.output_dir, 
                        'best_val_epochs@{}_e@{:.4f}_f@{:.4f}.pth.tar'.format(epoch, val_err['energy'].avg, val_err['force'].avg))
                )
            if update_test_result:
                torch.save(
                    {'state_dict': model.state_dict()}, 
                    os.path.join(args.output_dir, 
                        'best_test_epochs@{}_e@{:.4f}_f@{:.4f}.pth.tar'.format(epoch, test_err['energy'].avg, test_err['force'].avg))
                )
            if (epoch + 1) % args.test_interval == 0 and (not update_val_result) and (not update_test_result):
                torch.save(
                    {'state_dict': model.state_dict()}, 
                    os.path.join(args.output_dir, 
                        'epochs@{}_e@{:.4f}_f@{:.4f}.pth.tar'.format(epoch, test_err['energy'].avg, test_err['force'].avg))
                )

            info_str = 'Epoch: [{epoch}] Target: [{target}] train_e_MAE: {train_e_mae:.5f}, train_f_MAE: {train_f_mae:.5f}, '.format(
                epoch=epoch, target=args.target, train_e_mae=train_err['energy'].avg, train_f_mae=train_err['force'].avg)
            info_str += 'val_e_MAE: {:.5f}, val_f_MAE: {:.5f}, '.format(val_err['energy'].avg, val_err['force'].avg)
            if (epoch + 1) % args.test_interval == 0:
                info_str += 'test_e_MAE: {:.5f}, test_f_MAE: {:.5f}, '.format(test_err['energy'].avg, test_err['force'].avg)
            info_str += 'Time: {:.2f}s'.format(time.perf_counter() - epoch_start_time)
            _log.info(info_str)
            
            info_str = 'Best -- val_epoch={}, test_epoch={}, '.format(best_metrics['val_epoch'], best_metrics['test_epoch'])
            info_str += 'val_e_MAE: {:.5f}, val_f_MAE: {:.5f}, '.format(best_metrics['val_energy_err'], best_metrics['val_force_err'])
            info_str += 'test_e_MAE: {:.5f}, test_f_MAE: {:.5f}\n'.format(best_metrics['test_energy_err'], best_metrics['test_force_err'])
            _log.info(info_str)
        
        # evaluation with EMA
        if model_ema is not None:
            ema_val_err, _ = evaluate(args=args, model=model_ema.module, criterion=criterion, 
                data_loader=val_loader,
                print_freq=args.print_freq, logger=_log, print_progress=False)
            
            if (epoch + 1) % args.test_interval == 0:
                ema_test_err, _ = evaluate(args=args, model=model_ema.module, criterion=criterion, 
                    data_loader=test_loader,
                    print_freq=args.print_freq, logger=_log, print_progress=True, max_iter=args.test_max_iter)
            else:
                ema_test_err, ema_test_loss = None, None
                
            update_val_result, update_test_result = update_best_results(args, best_ema_metrics, ema_val_err, ema_test_err, epoch)

            if accelerator.process_index == 0:

                if update_val_result:
                    torch.save(
                        {'state_dict': get_state_dict(model_ema)}, 
                        os.path.join(args.output_dir, 
                            'best_ema_val_epochs@{}_e@{:.4f}_f@{:.4f}.pth.tar'.format(epoch, ema_val_err['energy'].avg, ema_val_err['force'].avg))
                    )
                if update_test_result:
                    torch.save(
                        {'state_dict': get_state_dict(model_ema)}, 
                        os.path.join(args.output_dir, 
                            'best_ema_test_epochs@{}_e@{:.4f}_f@{:.4f}.pth.tar'.format(epoch, ema_test_err['energy'].avg, ema_test_err['force'].avg))
                    )
                if (epoch + 1) % args.test_interval == 0 and (not update_val_result) and (not update_test_result):
                    torch.save(
                        {'state_dict': get_state_dict(model_ema)}, 
                        os.path.join(args.output_dir, 
                            'ema_epochs@{}_e@{:.4f}_f@{:.4f}.pth.tar'.format(epoch, test_err['energy'].avg, test_err['force'].avg))
                    )

                info_str = 'EMA '
                info_str += 'val_e_MAE: {:.5f}, val_f_MAE: {:.5f}, '.format(ema_val_err['energy'].avg, ema_val_err['force'].avg)
                if (epoch + 1) % args.test_interval == 0:
                    info_str += 'test_e_MAE: {:.5f}, test_f_MAE: {:.5f}, '.format(ema_test_err['energy'].avg, ema_test_err['force'].avg)
                info_str += 'Time: {:.2f}s'.format(time.perf_counter() - epoch_start_time)
                _log.info(info_str)
                
                info_str = 'Best EMA -- val_epoch={}, test_epoch={}, '.format(best_ema_metrics['val_epoch'], best_ema_metrics['test_epoch'])
                info_str += 'val_e_MAE: {:.5f}, val_f_MAE: {:.5f}, '.format(best_ema_metrics['val_energy_err'], best_ema_metrics['val_force_err'])
                info_str += 'test_e_MAE: {:.5f}, test_f_MAE: {:.5f}\n'.format(best_ema_metrics['test_energy_err'], best_ema_metrics['test_force_err'])
                _log.info(info_str)

    # evaluate on the whole testing set
    test_err, test_loss = evaluate(args=args, model=model, criterion=criterion, 
        data_loader=test_loader,
        print_freq=args.print_freq, logger=_log, print_progress=True, max_iter=-1)
        

def update_best_results(args, best_metrics, val_err, test_err, epoch):

    def _compute_weighted_error(args, energy_err, force_err):
        return args.energy_weight * energy_err + args.force_weight * force_err 

    update_val_result, update_test_result = False, False 

    new_loss  = _compute_weighted_error(args, val_err['energy'].avg, val_err['force'].avg)
    prev_loss = _compute_weighted_error(args, best_metrics['val_energy_err'], best_metrics['val_force_err'])
    if new_loss < prev_loss:
        best_metrics['val_energy_err'] = val_err['energy'].avg
        best_metrics['val_force_err']  = val_err['force'].avg
        best_metrics['val_epoch'] = epoch
        update_val_result = True

    if test_err is None:
        return update_val_result, update_test_result

    new_loss  = _compute_weighted_error(args, test_err['energy'].avg, test_err['force'].avg)
    prev_loss = _compute_weighted_error(args, best_metrics['test_energy_err'], best_metrics['test_force_err'])
    if new_loss < prev_loss:
        best_metrics['test_energy_err'] = test_err['energy'].avg
        best_metrics['test_force_err']  = test_err['force'].avg
        best_metrics['test_epoch'] = epoch
        update_test_result = True

    return update_val_result, update_test_result


def train_one_epoch(args, 
                    model: torch.nn.Module, accelerator: Accelerator, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    epoch: int, 
                    model_ema: Optional[ModelEma] = None,  
                    print_freq: int = 100, 
                    logger=None):
    
    model.train()
    criterion.train()
    
    loss_metrics = {'energy': AverageMeter(), 'force': AverageMeter()}
    mae_metrics  = {'energy': AverageMeter(), 'force': AverageMeter()}
    
    start_time = time.perf_counter()

    for step, data in enumerate(data_loader):

        pred_y, pred_dy = model(data)

        loss_e = criterion(pred_y, data.y)
        loss_f = criterion(pred_dy, data['force'])
        loss = args.energy_weight * loss_e + args.force_weight * loss_f

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        
        loss_metrics['energy'].update(loss_e.item(), n=pred_y.shape[0])
        loss_metrics['force'].update(loss_f.item(), n=pred_dy.shape[0])
        
        energy_err = pred_y.detach() - data.y
        energy_err = torch.mean(torch.abs(energy_err)).item()
        mae_metrics['energy'].update(energy_err, n=pred_y.shape[0])
        force_err = pred_dy.detach()- data['force']
        force_err = torch.mean(torch.abs(force_err)).item()     # based on OC20 and TorchMD-Net, they average over x, y, z
        mae_metrics['force'].update(force_err, n=pred_dy.shape[0])
        
        if model_ema is not None:
            model_ema.update(model)
        
        if accelerator.process_index == 0:

            # logging
            if step % print_freq == 0 or step == len(data_loader) - 1: 
                w = time.perf_counter() - start_time
                e = (step + 1) / len(data_loader)
                info_str = 'Epoch: [{epoch}][{step}/{length}] \t'.format(epoch=epoch, step=step, length=len(data_loader))
                info_str +=  'loss_e: {loss_e:.5f}, loss_f: {loss_f:.5f}, e_MAE: {e_mae:.5f}, f_MAE: {f_mae:.5f}, '.format(
                    loss_e=loss_metrics['energy'].avg, loss_f=loss_metrics['force'].avg, 
                    e_mae=mae_metrics['energy'].avg, f_mae=mae_metrics['force'].avg, 
                )
                info_str += 'time/step={time_per_step:.0f}ms, '.format( 
                    time_per_step=(1e3 * w / e / len(data_loader))
                )
                info_str += 'lr={:.2e}'.format(optimizer.param_groups[0]["lr"])
                logger.info(info_str)
        
    return mae_metrics, loss_metrics


def evaluate(args, 
            model: torch.nn.Module, criterion: torch.nn.Module,
            data_loader: Iterable, 
            print_freq: int = 100, 
            logger=None, 
            print_progress=False, 
            max_iter=-1):

    model.eval()
    criterion.eval()
    loss_metrics = {'energy': AverageMeter(), 'force': AverageMeter()}
    mae_metrics  = {'energy': AverageMeter(), 'force': AverageMeter()}
    
    start_time = time.perf_counter()
    
    with torch.no_grad():
            
        for step, data in enumerate(data_loader):

            pred_y, pred_dy = model(node_atom=data.z, pos=data.pos, batch=data.batch)

            loss_e = criterion(pred_y, data.y)
            loss_f = criterion(pred_dy, data['force'])
            
            loss_metrics['energy'].update(loss_e.item(), n=pred_y.shape[0])
            loss_metrics['force'].update(loss_f.item(), n=pred_dy.shape[0])
            
            energy_err = pred_y.detach() - data.y
            energy_err = torch.mean(torch.abs(energy_err)).item()
            mae_metrics['energy'].update(energy_err, n=pred_y.shape[0])
            force_err = pred_dy.detach() - data['force']
            force_err = torch.mean(torch.abs(force_err)).item()     # based on OC20 and TorchMD-Net, they average over x, y, z
            mae_metrics['force'].update(force_err, n=pred_dy.shape[0])
            
            if accelerator.process_index == 0:
                # logging
                if (step % print_freq == 0 or step == len(data_loader) - 1) and print_progress: 
                    w = time.perf_counter() - start_time
                    e = (step + 1) / len(data_loader)
                    info_str = '[{step}/{length}] \t'.format(step=step, length=len(data_loader))
                    info_str +=  'e_MAE: {e_mae:.5f}, f_MAE: {f_mae:.5f}, '.format(
                        e_mae=mae_metrics['energy'].avg, f_mae=mae_metrics['force'].avg, 
                    )
                    info_str += 'time/step={time_per_step:.0f}ms'.format( 
                        time_per_step=(1e3 * w / e / len(data_loader))
                    )
                    logger.info(info_str)
            
            if ((step + 1) >= max_iter) and (max_iter != -1):
                break

    return mae_metrics, loss_metrics
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser('Training equivariant networks', parents=[get_args_parser()])
    args = parser.parse_args()  
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
