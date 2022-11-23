import logging
import os
import sys

def setup_logger(logname, logfile, log_to_stdout):
    logger = logging.getLogger(logname)

    logger.propagate = False
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(fmt='%(asctime)s.%(msecs)03d: %(name)s: %(message)s', datefmt='%d/%m/%y %H:%M:%S')

    if logfile:
        handler = logging.FileHandler(logfile, 'a')
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    if log_to_stdout:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    return logger
