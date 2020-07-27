import logging
import sys

# logFileMode='a' or 'w', same as used in open(filename, *)

def configLogger(logFile, logFileMode='a', handlerStdoutLevel = logging.INFO, handlerFileLevel = logging.DEBUG, format=None, printToScreen=False):
    logger = logging.getLogger("logger")
    if format is None:
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    else:
        formatter = logging.Formatter(format)

    if printToScreen:
        handlerStdout = logging.StreamHandler(sys.stdout)
        handlerStdout.setLevel(handlerStdoutLevel)
        handlerStdout.setFormatter(formatter)
        logger.addHandler(handlerStdout)

    handlerFile = logging.FileHandler(filename=logFile, mode=logFileMode)

    logger.setLevel(logging.DEBUG)
    handlerFile.setLevel(handlerFileLevel)

    handlerFile.setFormatter(formatter)

    logger.addHandler(handlerFile)

    return logger

