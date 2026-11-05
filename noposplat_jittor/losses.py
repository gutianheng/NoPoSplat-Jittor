"""Photometric losses: FP32 pixels/gradients, stable scalar accumulation."""


def mse_loss(prediction,target):
    """Accumulate the FP32 squared residuals in FP64, return an FP32 scalar.

    Large flat FP32 reductions in Jittor 1.3.11 lose small residuals in the
    diagnostic images. Only scalar accumulation uses FP64; this does not turn
    the network, renderer, optimizer or pixel arithmetic into FP64.
    """
    if tuple(prediction.shape)!=tuple(target.shape):raise ValueError('MSE image shapes differ')
    if prediction.dtype!='float32' or target.dtype!='float32':raise TypeError('MSE expects FP32 images')
    squared=(prediction-target)**2
    return squared.float64().mean().float32()
