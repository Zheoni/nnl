import math


def identity(x: float) -> float:
    return x


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def binary(x: float) -> float:
    return 1.0 if x >= 0.0 else 0.0
