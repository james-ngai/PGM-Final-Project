import torch 


def constantlr():
    return lambda step: 1


def warmup():
    return lambda step: step / 1000