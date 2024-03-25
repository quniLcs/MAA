import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.propagate = False


def log(*obj):
    for x in obj:
        logger.info(str(x))
