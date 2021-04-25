import chainer
import chainer.functions as F


def focal_loss(x, t, class_num=10, alpha=0.2, gamma=2, eps=1e-7, reduce='mean'):
    xp = chainer.cuda.get_array_module(t)

    logit = F.softmax(x)
    logit = F.clip(logit, x_min=eps, x_max=1-eps)

    t_onehot = xp.eye(class_num)[t]

    loss_ce = -1 * t_onehot * F.log(logit)
    loss_focal = loss_ce * alpha * (1 - logit) ** gamma

    loss_focal = F.mean(loss_focal, axis=1)
    if reduce == 'mean':
        loss_focal = F.mean(loss_focal)

    return loss_focal
