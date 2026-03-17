from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import argparse
import torch.utils.data as data
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
from data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_re50
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from models.retinaface import RetinaFace

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--training_dataset', default='./data/widerface/train/label.txt', help='Training dataset directory')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--num_workers', default=24, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default='./weights/mobilenet0.25_Final.pth', help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./weights/weight_grhnc3_da/', help='Location to save checkpoint models')
parser.add_argument('--distributed', action='store_true', help='Use DistributedDataParallel training')
parser.add_argument('--dist_backend', default='nccl', type=str, help='Distributed backend')
parser.add_argument('--dist_url', default='env://', type=str, help='URL used to set up distributed training')
parser.add_argument('--rank', default=0, type=int, help='Node rank for distributed training')
parser.add_argument('--world_size', default=1, type=int, help='Number of processes for distributed training')
parser.add_argument('--local_rank', default=-1, type=int, help='Local rank for distributed training')
parser.add_argument('--empty_label_ratio', default=0.0, type=float,
                    help='Ratio of empty-label samples in each batch, range [0, 1). 0 means disabled.')

args = parser.parse_args()


class BalancedEmptyLabelBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, empty_label_ratio=0.0, drop_last=False,
                 distributed=False, num_replicas=1, rank=0):
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        if empty_label_ratio < 0 or empty_label_ratio >= 1:
            raise ValueError('empty_label_ratio must be in [0, 1).')

        self.dataset = dataset
        self.batch_size = batch_size
        self.empty_label_ratio = empty_label_ratio
        self.drop_last = drop_last
        self.distributed = distributed
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        self.empty_per_batch = min(batch_size - 1, int(round(batch_size * empty_label_ratio)))
        self.non_empty_per_batch = batch_size - self.empty_per_batch

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _split_for_rank(self, indices):
        if not self.distributed:
            return indices
        return indices[self.rank::self.num_replicas]

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.epoch + 17)

        empty_indices = list(self.dataset.get_empty_label_indices())
        non_empty_indices = list(self.dataset.get_non_empty_label_indices())

        if len(non_empty_indices) == 0:
            raise RuntimeError('Dataset has no non-empty labels, cannot build mixed batches.')

        if len(empty_indices) == 0 or self.empty_per_batch == 0:
            all_indices = non_empty_indices + empty_indices
            perm = torch.randperm(len(all_indices), generator=generator).tolist()
            all_indices = [all_indices[i] for i in perm]
            all_indices = self._split_for_rank(all_indices)
            for i in range(0, len(all_indices), self.batch_size):
                batch = all_indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or (len(batch) > 0 and not self.drop_last):
                    yield batch
            return

        empty_perm = torch.randperm(len(empty_indices), generator=generator).tolist()
        non_empty_perm = torch.randperm(len(non_empty_indices), generator=generator).tolist()
        empty_pool = [empty_indices[i] for i in empty_perm]
        non_empty_pool = [non_empty_indices[i] for i in non_empty_perm]

        empty_pool = self._split_for_rank(empty_pool)
        non_empty_pool = self._split_for_rank(non_empty_pool)

        empty_ptr = 0
        non_empty_ptr = 0
        while non_empty_ptr < len(non_empty_pool):
            next_non_empty = non_empty_pool[non_empty_ptr:non_empty_ptr + self.non_empty_per_batch]
            non_empty_ptr += self.non_empty_per_batch
            if len(next_non_empty) < self.non_empty_per_batch and self.drop_last:
                break

            batch = list(next_non_empty)
            target_empty = self.batch_size - len(batch)
            if len(empty_pool) > 0 and target_empty > 0:
                for _ in range(target_empty):
                    batch.append(empty_pool[empty_ptr])
                    empty_ptr = (empty_ptr + 1) % len(empty_pool)

            if len(batch) == self.batch_size or (len(batch) > 0 and not self.drop_last):
                batch_perm = torch.randperm(len(batch), generator=generator).tolist()
                batch = [batch[i] for i in batch_perm]
                yield batch

    def __len__(self):
        non_empty_indices = list(self.dataset.get_non_empty_label_indices())
        if self.distributed:
            non_empty_indices = self._split_for_rank(non_empty_indices)
        if self.drop_last:
            return len(non_empty_indices) // self.non_empty_per_batch
        return math.ceil(len(non_empty_indices) / self.non_empty_per_batch)


def init_distributed_mode(opt):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        opt.rank = int(os.environ['RANK'])
        opt.world_size = int(os.environ['WORLD_SIZE'])
        opt.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    elif opt.local_rank != -1:
        opt.rank = opt.local_rank
    else:
        opt.distributed = False
        return False

    opt.distributed = True
    torch.cuda.set_device(opt.local_rank)
    dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                            world_size=opt.world_size, rank=opt.rank)
    dist.barrier()
    return True

def is_main_process():
    return (not is_distributed) or args.rank == 0


def train():
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    dataset = WiderFaceDetection(training_dataset,preproc(img_dim, rgb_mean))

    train_sampler = None
    batch_sampler = None
    if args.empty_label_ratio > 0:
        batch_sampler = BalancedEmptyLabelBatchSampler(
            dataset,
            batch_size=batch_size,
            empty_label_ratio=args.empty_label_ratio,
            drop_last=False,
            distributed=is_distributed,
            num_replicas=dist.get_world_size() if is_distributed else 1,
            rank=args.rank if is_distributed else 0,
        )
        print('Using BalancedEmptyLabelBatchSampler, empty_label_ratio={:.2f}'.format(args.empty_label_ratio))
    else:
        train_sampler = DistributedSampler(dataset, shuffle=True) if is_distributed else None

    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size if batch_sampler is None else 1,
        shuffle=(train_sampler is None and batch_sampler is None),
        num_workers=num_workers,
        sampler=train_sampler,
        batch_sampler=batch_sampler,
        collate_fn=detection_collate,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    epoch_size = len(dataloader)

    max_iter = max_epoch * epoch_size

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0
    epoch_t0 = time.perf_counter()
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 19 and is_main_process():
            epoch_t0 = time.perf_counter()
        if iteration % epoch_size == 0:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if batch_sampler is not None and hasattr(batch_sampler, 'set_epoch'):
                batch_sampler.set_epoch(epoch)

            # create batch iterator
            batch_iterator = iter(dataloader)
            # if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
            if is_main_process():
                epoch_t1 = time.perf_counter()
                epoch_time = epoch_t1 - epoch_t0
                state_dict = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
                torch.save(state_dict, save_folder + cfg['name'] + '_epoch_' + str(epoch) + '.pth')
                print('Epoch {}  Time {:.3f}'.format(epoch, epoch_time))
            epoch += 1



        load_t0 = time.perf_counter()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        t0 = time.perf_counter()
        images, targets = next(batch_iterator)
        if is_distributed:
            images = images.cuda(args.local_rank, non_blocking=True)
            targets = [anno.cuda(args.local_rank, non_blocking=True) for anno in targets]
        else:
            images = images.cuda(non_blocking=True)
            targets = [anno.cuda(non_blocking=True) for anno in targets]
        torch.cuda.synchronize()
        data_time = time.perf_counter() - t0

        # forward
        t1 = time.perf_counter()
        out = net(images)
        torch.cuda.synchronize()
        forward_time = time.perf_counter() - t1

        # backprop
        t2 = time.perf_counter()

        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        torch.cuda.synchronize()
        loss_time = time.perf_counter() - t2

        t3 = time.perf_counter()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        backward_time = time.perf_counter() - t3
        batch_time = time.perf_counter() - load_t0
        eta = int(batch_time * (max_iter - iteration))

        if iteration % 20 == 0 and is_main_process():
            print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Datatime: {:.4f} s ||forwardtime: {:.4f} s ||losstime: {:.4f} s ||backwardtime: {:.4f} s ||Batchtime: {:.4f} s || ETA: {}'
                .format(epoch, max_epoch, (iteration % epoch_size) + 1,
                epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), lr, data_time, forward_time, loss_time, backward_time, batch_time, str(datetime.timedelta(seconds=eta))))
    if is_main_process():
        state_dict = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
        torch.save(state_dict, save_folder + cfg['name'] + '_Final.pth')
    # torch.save(net.state_dict(), save_folder + 'Final_Retinaface.pth')


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    is_distributed = init_distributed_mode(args)

    if not os.path.exists(args.save_folder) and is_main_process():
        os.mkdir(args.save_folder)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50

    rgb_mean = (104, 117, 123)  # bgr order
    num_classes = 2
    img_dim = cfg['image_size']
    num_gpu = cfg['ngpu']
    batch_size = cfg['batch_size']
    max_epoch = cfg['epoch']
    gpu_train = cfg['gpu_train']

    num_workers = args.num_workers
    momentum = args.momentum
    weight_decay = args.weight_decay
    initial_lr = args.lr
    gamma = args.gamma
    training_dataset = args.training_dataset
    save_folder = args.save_folder

    net = RetinaFace(cfg=cfg)
    if is_main_process():
        print("Printing net...")
        print(net)

    if args.resume_net is not None:
        print('Loading resume network...')
        state_dict = torch.load(args.resume_net, weights_only=True)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)

    if is_distributed:
        num_gpu = dist.get_world_size()
        if batch_size % num_gpu != 0:
            raise ValueError('For distributed training, cfg[\'batch_size\'] must be divisible by world_size.')
        batch_size = batch_size // num_gpu
        net = net.cuda(args.local_rank)
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank],
                                                        output_device=args.local_rank)
    elif num_gpu > 1 and gpu_train:
        net = torch.nn.DataParallel(net).cuda()
    else:
        net = net.cuda()

    cudnn.benchmark = True

    optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
    criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

    priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))

    with torch.no_grad():
        priors = priorbox.forward()
        if is_distributed:
            priors = priors.cuda(args.local_rank)
        else:
            priors = priors.cuda()

    train()
