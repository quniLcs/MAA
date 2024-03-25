import argparse
import os
import random
import re
from pprint import pformat

from utils.config import cfg
from utils.globals import global_dict
from utils.log import log, logger, logging

try:
    import moxing as mox
    global_dict["run_on_remote"] = True
except:
    global_dict["run_on_remote"] = False


def main():
    if 'RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(random.randint(50000, 60000))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config_path',
        type=str,
        help='Path to the config file.',
        default='configs/train_context.toml',
    )
    parser.add_argument(
        '--cfgs',
        default=None,
        nargs="*",
    )
    args, unknown_args = parser.parse_known_args()

    from dynaconf import Dynaconf, loaders
    init_cfg = Dynaconf(
        settings_files=[
            args.config_path,
        ],
        environments=True,
        load_dotenv=True,
        envvar_prefix='DYNACONF',
        env_switcher='ENV_FOR_DYNACONF',
        dotenv_path='configs/.env',
    )

    cfg.clean()
    cfg.update(init_cfg)
    if args.cfgs:
        if len(args.cfgs) == 1 and type(args.cfgs[0]) == str:
            args.cfgs = re.split('\s+', args.cfgs[0].strip())
        if len(args.cfgs) % 2 != 0:
            raise ValueError(f"nargs of --cfgs should be divisible by 2. args.cfgs: {args.cfgs}")
        str2bool_true = lambda x: True if type(x) == str and x.lower() in ['true', 't', 'y', 'yes'] else x
        str2bool_false = lambda x: False if type(x) == str and x.lower() in ['false', 'f', 'n', 'no'] else x
        str2bool = lambda x: str2bool_true(str2bool_false(x))
        args.cfgs = list(map(str2bool, args.cfgs))
        if len(args.cfgs) > 0:
            cfgs_from_cmd = {x[0]: type(cfg[x[0]])(x[1]) for x in zip(args.cfgs[0::2], args.cfgs[1::2])}
            print(f"cfgs_from_cmd:\n{pformat(cfgs_from_cmd)}")
            cfg.update(cfgs_from_cmd)

    cfg.LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    cfg.RANK = int(os.environ['RANK'])
    cfg.WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    cfg.MASTER_ADDR = os.environ['MASTER_ADDR']

    if cfg.WORLD_SIZE > 1:
        cfg.distributed = True
    else:
        cfg.distributed = False

    batch_size_per_step = cfg.BATCH_SIZE_PER_GPU * cfg.WORLD_SIZE
    cfg.EFFECTIVE_BATCH_SIZE = batch_size_per_step * cfg.ACCUM_STEP

    if not cfg.TEST_ONLY:
        cfg.LR_WARMUP_STEP = max(1, cfg.LR_WARMUP_STEP * 16 // batch_size_per_step)
        cfg.LR_COSINE_T_0 = max(1, cfg.LR_COSINE_T_0 * 16 // batch_size_per_step)
        cfg.LOG_EVERY_STEP = max(1, cfg.LOG_EVERY_STEP * 16 // batch_size_per_step)
        cfg.VALIDATE_EVERY_STEP = max(1, cfg.VALIDATE_EVERY_STEP * 16 // batch_size_per_step)
        cfg.SAVE_MODEL_EVERY_STEP = max(1, cfg.SAVE_MODEL_EVERY_STEP * 16 // batch_size_per_step)

    global_dict["logging_formatter"] = logging.Formatter(
        f'[%(asctime)s][RANK={cfg.RANK:02d}][%(levelname).1s]: %(message)s \t[%(pathname)s:%(lineno)d]',
    )

    logger.handlers.clear()
    if cfg.RANK <= 99:
        sh = logging.StreamHandler()
        sh.setFormatter(global_dict["logging_formatter"])
        logger.addHandler(sh)

    logger.info(f"unknown_args={unknown_args}")

    import train
    return train.main()


if __name__ == '__main__':
    main()
