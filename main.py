import configargparse
import data_loader
import os
import torch
import models
import utils
from utils import str2bool
import numpy as np
import random


def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add("--config", is_config_file=True, help="config file path")
    parser.add("--seed", type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--json_path', type=str, default='')
    parser.add_argument('--n_class', type=int, default=3)
    
    # network related
    parser.add_argument('--use_bottleneck', type=str2bool, default=True)
    
    # training related
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=0, help="Early stopping")
    parser.add_argument('--epoch_based_training', type=str2bool, default=False, help="Epoch-based training / Iteration-based training")
    parser.add_argument("--n_iter_per_epoch", type=int, default=20, help="Used in Iteration-based training")

    # optimizer related
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # learning rate scheduler related
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=str2bool, default=True)
    
    parser.add_argument('--transfer_loss1_weight', type=float, default=0.5)
    parser.add_argument('--transfer_loss2_weight', type=float, default=1.0)

    return parser


def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(args):
    source_loader = data_loader.load_data(
        args.json_path, args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True, num_workers=args.num_workers, domain='source')
    target_train_loader = data_loader.load_data(
        args.json_path, args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True, num_workers=args.num_workers, domain='target')
    target_test_loader = data_loader.load_data(
        args.json_path, args.batch_size, infinite_data_loader=False, train=False, num_workers=args.num_workers, domain='target')
    return source_loader, target_train_loader, target_test_loader


def get_model(args):
    model = models.myTransferNet(
        args.n_class, transfer_loss1='mmd', transfer_loss2 ='adv', max_iter=args.max_iter, use_bottleneck=args.use_bottleneck).to(args.device)
    return model


def get_optimizer(model, args):
    initial_lr = args.lr if not args.lr_scheduler else 1.0
    params = model.get_parameters(initial_lr=initial_lr)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    return optimizer


def get_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    return scheduler


def test(model, target_test_loader, args):
    model.eval()
    test_loss = utils.AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.to(args.device), target.to(args.device)
            s_output = model.predict(data)
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)
    acc = 100. * correct / len_target_dataset
    return acc, test_loss.avg


def train(source_loader, target_train_loader, target_test_loader, model, optimizer, lr_scheduler, args):
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    n_batch = min(len_source_loader, len_target_loader)
    if n_batch == 0:
        n_batch = args.n_iter_per_epoch 
    
    iter_source, iter_target = iter(source_loader), iter(target_train_loader)

    best_acc = 0
    stop = 0
    log = []
    for e in range(1, args.n_epoch+1):
        model.train()
        train_loss_clf = utils.AverageMeter()
        train_loss1_transfer = utils.AverageMeter()
        train_loss2_transfer = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        model.epoch_based_processing(n_batch)
        
        if max(len_target_loader, len_source_loader) != 0:
            iter_source, iter_target = iter(source_loader), iter(target_train_loader)
        
        for _ in range(n_batch):
            data_source, label_source = next(iter_source) # .next()
            data_target, _ = next(iter_target) # .next()
            data_source, label_source = data_source.to(args.device), label_source.to(args.device)
            data_target = data_target.to(args.device)
            
            clf_loss, transfer_loss1, transfer_loss2 = model(data_source, data_target, label_source)
            loss = clf_loss + args.transfer_loss1_weight * transfer_loss1 + args.transfer_loss2_weight * transfer_loss2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()

            train_loss_clf.update(clf_loss.item())
            train_loss1_transfer.update(transfer_loss1.item())
            train_loss2_transfer.update(transfer_loss2.item())
            train_loss_total.update(loss.item())

        
        info = 'Epoch: [{:2d}/{}], cls_loss: {:.4f}, transfer_loss1: {:.4f}, transfer_loss2: {:.4f}, total_Loss: {:.4f}'.format(
                        e, args.n_epoch, train_loss_clf.avg, train_loss1_transfer.avg, train_loss2_transfer.avg, train_loss_total.avg)
        
        # save model
        # torch.save(model.state_dict(), f'/opt/data/private/TBdetection/log/mymodel/model_epoch_{e}.pth') 

        
        # Test
        stop += 1
        test_acc, test_loss = test(model, target_test_loader, args)
        info += ', test_loss {:4f}, test_acc: {:.4f}'.format(test_loss, test_acc)
        
        if best_acc < test_acc:
            best_acc = test_acc
            stop = 0
        if args.early_stop > 0 and stop >= args.early_stop:
            print(info)
            break
        print(info)
    print('Transfer result: {:.4f}'.format(best_acc))


def main():
    parser = get_parser()
    args = parser.parse_args()
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(args)
    set_random_seed(args.seed)
    source_loader, target_train_loader, target_test_loader = load_data(args)
    if args.epoch_based_training:
        setattr(args, "max_iter", args.n_epoch * min(len(source_loader), len(target_train_loader)))
    else:
        setattr(args, "max_iter", args.n_epoch * args.n_iter_per_epoch)
        
    model = get_model(args)
    optimizer = get_optimizer(model, args)
    
    if args.lr_scheduler:
        scheduler = get_scheduler(optimizer, args)
    else:
        scheduler = None
    train(source_loader, target_train_loader, target_test_loader, model, optimizer, scheduler, args)
    

if __name__ == "__main__":
    main()
