import datetime
import logging
import os
import shutil
from pprint import pformat
from random import randint

import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from sklearn.metrics import average_precision_score
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from model.vit_crop_knowbert import Net
from utils.dataset import ImageText
from utils.globals import global_dict
from utils.log import log, logger
from utils.save_n_load import save_model, load_model
from utils.config import cfg
from utils.seed import set_random_seed

try:
    import moxing as mox
    run_on_remote = True
except:
    run_on_remote = False

logger.setLevel(logging.INFO)


def test_model(model, device, loader):
    logger.info('testing model...')
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        y_true = []
        y_scores = []
        loss = 0

        for datas in tqdm(loader):
            img = datas[0].to(device)
            texts = datas[1]
            targets = datas[2].to(device)
            y_true.extend(list(targets))
            targets = targets.to(device)

            outputs = model(img, texts)
            if isinstance(model, DDP):
                loss += model.module.criterion(outputs, targets)
            else:
                loss += model.criterion(outputs, targets)

            y_scores.append(outputs)
            _, index = torch.max(outputs, dim=1)
            total += targets.shape[0]
            correct += (index == targets).sum().item()

    y_scores = torch.nn.functional.softmax(torch.cat(y_scores), dim=1)
    y_label = y_true
    tmp = torch.zeros(y_scores.shape, dtype=torch.bool)
    tmp[range(y_scores.shape[0]), y_true] = 1
    y_true = tmp.numpy()
    mAP = average_precision_score(y_true, y_scores.cpu().numpy(), average='macro')
    weighted_AP = average_precision_score(y_true, y_scores.cpu().numpy(), average='weighted')

    # mAP in the paper
    precision_per_class = torch.zeros(y_scores.shape[1])
    for i in tqdm(range(y_scores.shape[0])):
        precision_per_class[y_label[i]] += average_precision_score(y_true[i], y_scores.cpu().numpy()[i])
    class_count = [(torch.Tensor(y_label) == i).sum().item() for i in range(cfg.NUM_CLASSES)]
    total_precision = [precision_per_class[i] / class_count[i] for i in range(cfg.NUM_CLASSES)]
    mAP_paper = torch.Tensor(total_precision).mean().item()
    acc = correct / total
    loss = (loss / total).item()
    total_precision = [x.item() for x in total_precision]

    logger.info(f'paper AP for every class:\n{total_precision}')
    logger.info('Accuracy: ' + str(acc))
    logger.info('mAP: ' + str(mAP))
    logger.info('weighted AP: ' + str(weighted_AP))
    logger.info('In the paper, "mAP": ' + str(mAP_paper))
    logger.info(f'Loss: {loss:.6f}')

    return (acc, mAP), loss


def get_dataloader():
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop((224, 224), scale=(0.05, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    logger.info('loading datasets...')
    trainset = ImageText(
        dataset_name=cfg.DATASET_NAME,
        mode='train',
        transform=transform_train,
    )
    valset = ImageText(
        dataset_name=cfg.DATASET_NAME,
        mode='test',
        transform=transform_val,
    )

    logger.info('len(trainset): ' + str(len(trainset)))
    logger.info('len(valset): ' + str(len(valset)))

    train_datasampler = DistributedSampler(
        trainset,
        num_replicas=cfg.WORLD_SIZE,
        rank=cfg.RANK,
        shuffle=True,
        drop_last=True,
    )
    val_datasampler = DistributedSampler(
        valset,
        num_replicas=cfg.WORLD_SIZE,
        rank=cfg.RANK,
        shuffle=True,
        drop_last=False,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=cfg.BATCH_SIZE_PER_GPU,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
        sampler=train_datasampler,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=valset,
        batch_size=cfg.BATCH_SIZE_PER_GPU,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
        sampler=val_datasampler,
    )

    logger.info('len(train_loader): ' + str(len(train_loader)))
    logger.info('len(val_loader): ' + str(len(val_loader)))
    return train_loader, val_loader


def prepare_output(local_output_dir):

    shutil.rmtree(local_output_dir, ignore_errors=True)
    os.makedirs(os.path.join(local_output_dir, "others/codes/"), exist_ok=True)

    if cfg.RANK == 0:
        os.makedirs(cfg.TB_DIR, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=cfg.TB_DIR)
    else:

        class dummy_tb_writer(object):

            def __init__(self):
                super().__init__()

            def add_scalar(self, *args):
                pass

            def add_scalars(self, *args):
                pass

            def flush(self, *args):
                pass

        tb_writer = dummy_tb_writer()

    shutil.copyfile(__file__, os.path.join(local_output_dir, "others/codes/", __file__.split('/')[-1]))
    shutil.copytree('model', os.path.join(local_output_dir, 'others/codes/model'))
    shutil.copytree('utils', os.path.join(local_output_dir, 'others/codes/utils'))

    if run_on_remote:
        mox.file.copy_parallel(os.path.join(local_output_dir, 'others'),
                               os.path.join(cfg.OUTPUT_DIR, 'others'))

    if cfg.RANK == 0:
        fh = logging.FileHandler(os.path.join(local_output_dir, 'others/trainval.log'), mode='w')
        fh.setLevel(logging.INFO)
        fh.setFormatter(global_dict["logging_formatter"])
        logger.addHandler(fh)

    logger.info('start logging')
    logger.info('OUTPUT_DIR: ' + str(cfg.OUTPUT_DIR))
    logger.info('TB_DIR: ' + str(cfg.TB_DIR))
    logger.info('configs:\n' + pformat(cfg.as_dict()))
    logger.info(f"cfg.LOCAL_RANK={cfg.LOCAL_RANK}, cfg.RANK={cfg.RANK}, cfg.WORLD_SIZE={cfg.WORLD_SIZE}, cfg.distributed={cfg.distributed}")
    return tb_writer


def main():
    # setup
    if cfg.distributed:
        torch.cuda.set_device(cfg.LOCAL_RANK)
        dist.init_process_group(
            backend='nccl',
            # init_method="env://",
            # world_size=cfg.WORLD_SIZE,
            # rank=cfg.RANK,
        )

    if cfg.SEED == 'random':
        cfg.SEED = randint(0, 10**10)
    set_random_seed(cfg.SEED)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if cfg.OUTPUT_DIR.startswith('s3://'):
        cfg.TB_DIR = './outputs/tmp/others/tb_logs'
        local_output_dir = './outputs/tmp'
    else:
        tmp = datetime.datetime.now()
        tmp = tmp.strftime('%m%d%H%M%S')
        tmp = f"_{tmp}"
        tmp = "vit_knowbert_" + cfg.DATASET_NAME + tmp
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, tmp)
        cfg.TB_DIR = os.path.join(cfg.OUTPUT_DIR, 'others/tb_logs')
        local_output_dir = cfg.OUTPUT_DIR

    tb_writer = prepare_output(local_output_dir)

    train_loader, val_loader = get_dataloader()
    step_per_epoch = len(train_loader)
    if cfg.USE_NUM_T:
        num_T_0 = (cfg.LR_COSINE_T_MULT ** cfg.NUM_T - 1) // (cfg.LR_COSINE_T_MULT - 1) \
            if cfg.LR_COSINE_T_MULT > 1 else cfg.NUM_T
        cfg.LR_COSINE_T_0 = (step_per_epoch * cfg.NUM_EPOCHS - cfg.LR_WARMUP_STEP) // num_T_0
        cfg.VALIDATE_EVERY_STEP = step_per_epoch // 2
        cfg.SAVE_MODEL_EVERY_STEP = step_per_epoch
        logger.info(f"new cfg.LR_COSINE_T_0 = {cfg.LR_COSINE_T_0}")

    model = Net(cfg)
    model.to(device)

    if cfg.TEST_ONLY:
        load_model(model, cfg.CHECKPOINT_PATH, device, logger)
        logger.info(str(test_model(model, device, val_loader)))
        logger.info('configs:\n' + pformat(cfg.as_dict()))
        return 0

    if cfg.distributed:
        model = DDP(
            model,
            device_ids=[cfg.LOCAL_RANK],
            output_device=cfg.LOCAL_RANK,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )

    logger.info('Model:\n' + pformat(model))

    # Loss and optimizer
    if cfg.TEST_LR_RANGE:
        cfg.LR = 1e-6
        cfg.NUM_EPOCHS = 1
        cfg.LOG_EVERY_STEP = 10
        cfg.VALIDATE_EVERY_STEP = step_per_epoch // 8
        cfg.SAVE_MODEL_EVERY_STEP = float('inf')

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.USE_AMP)

    if not cfg.TEST_LR_RANGE:
        lr_scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(step / cfg.LR_WARMUP_STEP, 1),
        )
        if cfg.USE_MULTISTEP:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[200, 900],
                gamma=0.3,
            )
        else:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=cfg.LR_COSINE_T_0,
                T_mult=cfg.LR_COSINE_T_MULT,
                eta_min=1e-6,
            )
    else:
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            (1e-1 / cfg.LR) ** (1 / (len(train_loader) * cfg.NUM_EPOCHS)),
        )

    # Train the model
    logger.info('start training')
    max_val_res = torch.zeros(2)
    save_model(model, 'init_model.pth', cfg, logger)

    for epoch in range(cfg.NUM_EPOCHS):
        train_loader.sampler.set_epoch(epoch)

        for i, datas in enumerate(train_loader):
            model.train()
            global_step = epoch * step_per_epoch + i + 1

            with torch.cuda.amp.autocast(enabled=cfg.USE_AMP):
                img = datas[0].to(device)
                texts = datas[1]
                targets = datas[2].to(device)

                outputs = model(img, texts, targets)
                loss = outputs

            # Backward and optimize
            scaler.scale(loss).backward()
            if global_step % cfg.ACCUM_STEP == 0 or global_step == cfg.num_epochs * step_per_epoch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                model.zero_grad(set_to_none=True)

            # Update learning rate
            curr_lr = optimizer.param_groups[0]['lr']
            if not cfg.TEST_LR_RANGE:
                if global_step < cfg.LR_WARMUP_STEP:
                    lr_scheduler_warmup.step()
                else:
                    lr_scheduler.step()
            else:
                lr_scheduler.step()

            # Log
            if global_step % cfg.LOG_EVERY_STEP == 0:
                logger.info(
                    f'Epoch [{epoch + 1}/{cfg.NUM_EPOCHS}], Step [{i + 1}/{step_per_epoch}], '
                    f'Loss: {loss.item():.4f}, LR: {curr_lr:.3e}, MEM: {torch.cuda.memory_reserved() / 1024 / 1024:.0f} MB'
                )
                if cfg.RANK == 0:
                    tb_writer.add_scalar('train/loss', loss.item(), global_step)
                    tb_writer.add_scalar('train/lr', curr_lr, global_step)
                if cfg.TEST_LR_RANGE:
                    tb_writer.add_scalars('train/loss_lr', dict(curr_lr=curr_lr, loss=loss.item()), global_step)

            # Calculate Metrics
            if global_step % cfg.VALIDATE_EVERY_STEP == 0 and cfg.RANK == 0:
                res, loss = test_model(model, device, val_loader)
                res = torch.Tensor(res)
                if res[-1] > max_val_res[-1]:
                    save_model(model, 'best_trained_model.pth', cfg, logger,
                               '"best_trained_model.pth" saved. Current global_step is ' + str(global_step))
                max_val_res = torch.max(res, max_val_res)

                logger.info('Current max ACC and mAP: ' + str(max_val_res))
                tb_writer.add_scalar('val/max_mAP', max_val_res[-1], global_step)
                tb_writer.add_scalar('val/acc', res[0], global_step)
                tb_writer.add_scalar('val/mAP', res[-1], global_step)
                tb_writer.add_scalar('val/loss', loss, global_step)

                if cfg.RANK == 0 and cfg.OUTPUT_DIR.startswith('s3://'):
                    tb_writer.flush()
                    mox.file.copy_parallel(
                        cfg.TB_DIR,
                        os.path.join(cfg.OUTPUT_DIR, 'others/tb_logs'),
                    )
                    mox.file.copy_parallel(
                        os.path.join(local_output_dir, 'others/trainval.log'),
                        os.path.join(cfg.OUTPUT_DIR, 'others/trainval.log'),
                    )

            # Save model
            if global_step % cfg.SAVE_MODEL_EVERY_STEP == 0:
                save_model(model, f'step_{global_step}.pth', cfg, logger)

    # Finish training
    save_model(model, 'final.pth', cfg, logger)
    logger.info('max_val_res (acc, mAP): ' + str(max_val_res))

    # print cfg again
    logger.info('OUTPUT_DIR: ' + cfg.OUTPUT_DIR)
    logger.info('TB_DIR: ' + cfg.TB_DIR)
    logger.info('configs:\n' + pformat(cfg.as_dict()))
