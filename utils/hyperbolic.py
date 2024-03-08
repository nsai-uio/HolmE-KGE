"""Hyperbolic operations utils functions."""

import torch

MIN_NORM = 1e-15
BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5}


# ################# MATH FUNCTIONS ########################

class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        dtype = x.dtype
        x = x.double()
        return (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5).to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


def artanh(x):
    return Artanh.apply(x)


def tanh(x):
    return x.clamp(-15, 15).tanh()


# ################# HYP OPS ########################

def expmap0(u, c):
    """Exponential map taken at the origin of the Poincare ball with curvature c.

    Args:
        u: torch.Tensor of size B x d with hyperbolic points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

    Returns:
        torch.Tensor with tangent points.
    """
    sqrt_c = c ** 0.5
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return project(gamma_1, c)


def expmap(u, x, c):
    """Exponential map taken at the origin of the Poincare ball with curvature c.

    Args:
        u: torch.Tensor of size B x d with hyperbolic points, point
        x: tangent point of size B x d with hyperbolic points, point
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

    Returns:
        torch.Tensor with tangent points.
    """
    sqrt_c = c ** 0.5
    lbd = 1 / (1 - c * torch.sum(x * x, dim=-1, keepdim=True)).clamp_min(MIN_NORM)
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    gamma_1 = tanh(sqrt_c * lbd * u_norm) * u / (sqrt_c * u_norm)
    exp = mobius_add(x, gamma_1, c)
    return project(exp, c)

def expmap1(u, x, c):
    """Exponential map taken at the origin of the Poincare ball with curvature c.

    Args:
        u: torch.Tensor of size B x d with hyperbolic points, point
        x: tangent point of size B x d with hyperbolic points, point
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

    Returns:
        torch.Tensor with tangent points.
    """
    sqrt_c = c ** 0.5
    lbd = 1 / (1 - c * torch.sum(x * x, dim=-1, keepdim=True)).clamp_min(MIN_NORM)
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    gamma_1 = tanh(sqrt_c * lbd * u_norm) * u / (sqrt_c)
    exp = mobius_add(x, gamma_1, c)
    return project(exp, c)


def logmap0(y, c):
    """Logarithmic map taken at the origin of the Poincare ball with curvature c.

    Args:
        y: torch.Tensor of size B x d with tangent points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

    Returns:
        torch.Tensor with hyperbolic points.
    """
    sqrt_c = c ** 0.5
    y_norm = y.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)


def project(x, c):
    """Project points to Poincare ball with curvature c.

    Args:
        x: torch.Tensor of size B x d with hyperbolic points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

    Returns:
        torch.Tensor with projected hyperbolic points.
    """
    norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    eps = BALL_EPS[x.dtype]
    maxnorm = (1 - eps) / (c ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def mobius_add(x, y, c):
    """Mobius addition of points in the Poincare ball with curvature c.

    Args:
        x: torch.Tensor of size B x d with hyperbolic points
        y: torch.Tensor of size B x d with hyperbolic points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

    Returns:
        Tensor of shape B x d representing the element-wise Mobius addition of x and y.
    """
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / denom.clamp_min(MIN_NORM)


# ################# HYP DISTANCES ########################

def hyp_distance(x, y, c, eval_mode=False):
    """Hyperbolic distance on the Poincare ball with curvature c.

    Args:
        x: torch.Tensor of size B x d with hyperbolic queries
        y: torch.Tensor with hyperbolic queries, shape [n, d] if eval_mode is true else [B, d]
        c: torch.Tensor of size 1 with absolute hyperbolic curvature

    Returns: torch,Tensor with hyperbolic distances, size B x 1 if eval_mode is False
            else B x n_entities matrix with all pairs distances
    """
    sqrt_c = c ** 0.5
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    if eval_mode:
        y2 = torch.sum(y * y, dim=-1, keepdim=True).transpose(0, 1)
        xy = x @ y.transpose(0, 1)
    else:
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
    c1 = 1 - 2 * c * xy + c * y2
    c2 = 1 - c * x2
    num = torch.sqrt((c1 ** 2) * x2 + (c2 ** 2) * y2 - (2 * c1 * c2) * xy)
    denom = 1 - 2 * c * xy + c ** 2 * x2 * y2
    pairwise_norm = num / denom.clamp_min(MIN_NORM)
    dist = artanh(sqrt_c * pairwise_norm)
    return 2 * dist / sqrt_c


def hyp_distance_multi_c(x, v, c, eval_mode=False):
    """Hyperbolic distance on Poincare balls with varying curvatures c.

    Args:
        x: torch.Tensor of size B x d with hyperbolic queries
        v: torch.Tensor with hyperbolic queries, shape [n, d] if eval_mode is true else [B, d] 
        c: torch.Tensor of size B x 1 with absolute hyperbolic curvatures

    Return: torch,Tensor with hyperbolic distances, size B x 1 if eval_mode is False
            else B x n_entities matrix with all pairs distances
    """
    # sqrt_c = c ** 0.5 #[B, 1]
    # if eval_mode:
    #     vnorm = torch.norm(v, p=2, dim=-1, keepdim=True).transpose(0, 1) #[1, n]
    #     xv = x @ v.transpose(0, 1) / vnorm #[B, n]
    # else:
    #     vnorm = torch.norm(v, p=2, dim=-1, keepdim=True) #[B, 1]
    #     xv = torch.sum(x * v / vnorm, dim=-1, keepdim=True) #[B, 1]
    # gamma = tanh(sqrt_c * vnorm) / sqrt_c #[B, 1] / [B, n]
    # x2 = torch.sum(x * x, dim=-1, keepdim=True) #[B, 1]
    # c1 = 1 - 2 * c * gamma * xv + c * gamma ** 2 #[B, 1] / [B, n]
    # c2 = 1 - c * x2 #[B, 1]
    # num = torch.sqrt((c1 ** 2) * x2 + (c2 ** 2) * (gamma ** 2) - (2 * c1 * c2) * gamma * xv) #[B, 1] / [B, n]
    # denom = 1 - 2 * c * gamma * xv + (c ** 2) * (gamma ** 2) * x2 #[B, 1] / [B, n] 
    # pairwise_norm = num / denom.clamp_min(MIN_NORM) #[B, 1] / [B, n]
    # dist = artanh(sqrt_c * pairwise_norm) #[B, 1] / [B, n]
    # return 2 * dist / sqrt_c #[B, 1] / [B, n]

    sqrt_c = c ** 0.5 #[B, 1]
    B = sqrt_c.shape[0]
    n = v.shape[0]
    if eval_mode:
        vnorm = torch.norm(v, p=2, dim=-1, keepdim=True).transpose(0, 1) #[1, n]
        gamma = tanh(sqrt_c * vnorm) / (sqrt_c * vnorm) #[B, n]
        gamma = gamma.reshape(B, -1, 1)  #[B, n, 1]
        y = gamma * v.reshape(1, n, -1)  #[B, n, d]
        y = y.reshape(B*n, -1)
        x_ = torch.stack([x]*n,dim=1).view(B*n, -1) #[B*n, d]
        expand_c = torch.stack([c]*n,dim=1).view(-1,1) #[B*n, 1]
        dist = hyp_distance(x_, y, expand_c).reshape(B, -1) #[B, n]
    else:
        vnorm = torch.norm(v, p=2, dim=-1, keepdim=True) #[B, 1]
        gamma = tanh(sqrt_c * vnorm) / sqrt_c #[B, 1]
        y = (gamma * v) / vnorm  #[B, d]   
        dist = hyp_distance(x, y, c) #[B, 1]  
    return dist


    # x2 = torch.sum(x * x, dim=-1, keepdim=True) #[B, 1]
    # y2 = torch.sum(y * y, dim=-1, keepdim=True) #[B, 1]
    # xy = torch.sum(x * y, dim=-1, keepdim=True) #[B, 1]
    # c1 = 1 - 2 * c * xy + c * y2
    # c2 = 1- c * x2
    # num = torch.sqrt((c1 ** 2) * x2 + (c2 ** 2) * y2 - (2 * c1 * c2) * xy) 
    # denom = 1 - 2 * c * xy + (c **2) * x2 * y2
    # pairwise_norm = num / denom.clamp_min(MIN_NORM)


def hyp_distance_multi_t(x, v, c, tang, eval_mode=False):
    """Hyperbolic distance on Poincare balls with varying curvatures c.

    Args:
        x: torch.Tensor of size [B, d] with hyperbolic queries
        v: torch.Tensor with hyperbolic queries, shape [n, d] if eval_mode is true else (B x d)
        c: torch.Tensor of size [B, 1] with absolute hyperbolic curvatures
        tang: [B, d]

    Return: torch,Tensor with hyperbolic distances, size B x 1 if eval_mode is False
            else B x n_entities matrix with all pairs distances
    """
    batchSize = x.shape[0]
    validSize = v.shape[0] # B or n
    rank = x.shape[1] # d
    sqrt_c = c ** 0.5  #[B, 1]
    tnorm = torch.sum(tang * tang, dim=-1, keepdim=True)  #[B, 1]
    # print(c.shape, tnorm.shape)
    lbd = 1 / (1 - c * tnorm).clamp_min(MIN_NORM)  #[B, 1]
    if eval_mode:
        vnorm = torch.norm(v, p=2, dim=-1, keepdim=True).transpose(0, 1) # [1, n]
        gammar_ =  tanh(sqrt_c * lbd * vnorm) / (sqrt_c * vnorm).clamp_min(MIN_NORM) # [B, n]
        # gammar_ =  tanh(sqrt_c * lbd * vnorm) / sqrt_c.clamp_min(MIN_NORM) # [B, n]
        gammar_ = gammar_.reshape(batchSize, -1, 1) #[B, n, 1]
        y = gammar_ * v.reshape(1, validSize, -1) #[B, n, d]
        y = y.reshape(-1, rank) #[B*n, d]
        t_ = torch.stack([tang]*validSize, dim=1).view(-1,rank) #[B*n, d]
        extend_c = torch.stack([c]*validSize,dim=1).view(-1,1) #[B*n, 1]
        y = mobius_add(t_, y, extend_c) #[B*n, d]

        x_ = torch.stack([x]*validSize,dim=1).view(-1,rank) #[B*n, d]
        # assert False, 'batchSize{}, validSize{}, ranfk{}, {} {}'.format(batchSize, validSize, rank, x_.shape, y.shape)
        dist = hyp_distance(x_, y, extend_c).reshape(batchSize, -1)  #[B*n, 1] -> [B, n]
    else:
        vnorm = torch.norm(v, p=2, dim=-1, keepdim=True) # [B, 1]
        y = tanh(sqrt_c * lbd * vnorm) * v / (sqrt_c * vnorm).clamp_min(MIN_NORM) # [B, d]
        y = mobius_add(tang, y, c) #[B, d]

        dist = hyp_distance(x, y, c)
    return dist

    

    # u_norm = torch.norm(dim=-1, p=2, keepdim=True)

    
    # exp = mobius_add(x, gamma_1, c)
    # return project(exp, c)

    # lbd = 1 / (1 - c * torch.sum(tang * tang, dim=-1, keepdim=True)).clamp_min(MIN_NORM) #[B, 1]
    # sqrt_c = c ** 0.5 #[B, 1]
    # if eval_mode:
    #     vnorm = torch.norm(v, p=2, dim=-1, keepdim=True).transpose(0, 1) # [1, n]
    #     tv = tang @ v.transpose(0, 1) / vnorm # [B, n]
    # else:
    #     vnorm = torch.norm(v, p=2, dim=-1, keepdim=True) # [B, 1]
    #     tv = torch.sum(tang * v / vnorm, dim=-1, keepdim=True) # [B, 1]
    # gamma_ = tanh(sqrt_c * lbd * vnorm) / sqrt_c #[B, 1] / [B, n]
    # y = gamma_ * v / vnorm  #[B, d] / [B, d]
    # t2 = torch.sum(tang * tang, dim=-1, keepdim=True) #[B, 1]
    # c1 = 1 + 2 * c * gamma_ * tv + c * gamma_ ** 2 #[B, 1] / [B, n]
    # c2 = 1 - c * t2 #[B, 1]
    # num = torch.sqrt((c1 ** 2) * t2 + (c2 ** 2) * (gamma_ ** 2) + (2 * c1 * c2) * gamma_ * tv) #[B, 1] / [B, n]
    # denom = 1 + 2 * c * gamma_ * tv + (c **2) * t2 * (gamma_ ** 2) #[B, 1] / [B, n]
    # y = num / denom.clamp_min(MIN_NORM) #[B, 1] / [B, n]
    
    # #[B, d] / [n, d]
    
    # c12 = 1 - 2 * c * gamma * xv + c * gamma ** 2

    # tgamma_ = torch.sum(tang * gamma_, dim=-1, keepdim=True) #[B, 1]
    # num = (1 + 2 * c * tgamma_ + c * gamma_2) * tang + (1 - c * t2) * gamma_
    # denom = 1 + 2 * c * tgamma_ + c ** 2 * t2 * gamma_2
    # tail = num / denom.clamp_min(MIN_NORM) # [B, rank] / [n, rank]

    # x2 = torch.sum(x * x, dim=-1, keepdim=True) #[B, 1]
    # tail2 = torch.sum(tail * tail, dim=-1, keepdim=True)
    # xtail = torch.sum(x * tail, dim=-1, keepdim=True)
    # num = - (1 - 2 * c * xtail + c * tail2) * x + (1 - c * x2) * tail
    # denom = 1 - 2 * c * xtail + c ** 2 * x2 * tail2
    # pairwise_norm = num / denom.clamp_min(MIN_NORM)

    # dist = artanh(sqrt_c * pairwise_norm)
    # return 2 * dist / sqrt_c





    


    # if eval_mode:
    #     vnorm = torch.norm(v, p=2, dim=-1, keepdim=True).transpose(0, 1) # [1, n]
    #     xv = x @ v.transpose(0, 1) / vnorm # # [B, n]
    # else:
    #     vnorm = torch.norm(v, p=2, dim=-1, keepdim=True) # [B, 1]
    #     xv = torch.sum(x * v / vnorm, dim=-1, keepdim=True) # [B, 1]
    # gamma = tanh(sqrt_c * vnorm) / sqrt_c
    # x2 = torch.sum(x * x, dim=-1, keepdim=True)
    # c1 = 1 - 2 * c * gamma * xv + c * gamma ** 2
    # c2 = 1 - c * x2
    # num = torch.sqrt((c1 ** 2) * x2 + (c2 ** 2) * (gamma ** 2) - (2 * c1 * c2) * gamma * xv)
    # denom = 1 - 2 * c * gamma * xv + (c ** 2) * (gamma ** 2) * x2
    # pairwise_norm = num / denom.clamp_min(MIN_NORM)
    # dist = artanh(sqrt_c * pairwise_norm)
    # return 2 * dist / sqrt_c


def hyp_distance_multi_t2(x, v, c, tang, eval_mode=False):
    """Hyperbolic distance on Poincare balls with varying curvatures c.

    Args:
        x: torch.Tensor of size [B, d] with hyperbolic queries
        v: torch.Tensor with hyperbolic queries, shape [n, d] if eval_mode is true else (B x d)
        c: torch.Tensor of size [B, 1] with absolute hyperbolic curvatures
        tang: [B, d]

    Return: torch,Tensor with hyperbolic distances, size B x 1 if eval_mode is False
            else B x n_entities matrix with all pairs distances
    """
    batchSize = x.shape[0]
    validSize = v.shape[0] # B or n
    rank = x.shape[1] # d
    sqrt_c = c ** 0.5  #[B, 1]
    tnorm = torch.sum(tang * tang, dim=-1, keepdim=True)  #[B, 1]
    # print(c.shape, tnorm.shape)
    lbd = 1 / (1 - c * tnorm).clamp_min(MIN_NORM)  #[B, 1]
    if eval_mode:
        vnorm = torch.norm(v, p=2, dim=-1, keepdim=True).transpose(0, 1) # [1, n]
        gammar_ =  tanh(sqrt_c * lbd * vnorm) / (sqrt_c * vnorm).clamp_min(MIN_NORM) # [B, n]
        # gammar_ =  tanh(sqrt_c * lbd * vnorm) / sqrt_c.clamp_min(MIN_NORM) # [B, n]
        gammar_ = gammar_.reshape(batchSize, -1, 1) #[B, n, 1]
        y = gammar_ * v.reshape(1, validSize, -1) #[B, n, d]
        y = y.reshape(-1, rank) #[B*n, d]
        t_ = torch.stack([tang]*validSize, dim=1).view(-1,rank) #[B*n, d]
        extend_c = torch.stack([c]*validSize,dim=1).view(-1,1) #[B*n, 1]
        y = mobius_add(t_, y, extend_c) #[B*n, d]

        x_ = torch.stack([x]*validSize,dim=1).view(-1,rank) #[B*n, d]
        # assert False, 'batchSize{}, validSize{}, ranfk{}, {} {}'.format(batchSize, validSize, rank, x_.shape, y.shape)
        dist = hyp_distance(x_, y, extend_c).reshape(batchSize, -1)  #[B*n, 1] -> [B, n]
    else:
        vnorm = torch.norm(v, p=2, dim=-1, keepdim=True) # [B, 1]
        y = tanh(sqrt_c * lbd * vnorm) * v / (sqrt_c * vnorm).clamp_min(MIN_NORM) # [B, d]
        y = mobius_add(tang, y, c) #[B, d]

        dist = hyp_distance(x, y, c)

    return dist


def hyp_distance_multi_t3(x, v, c, tang, eval_mode=False):
    """Hyperbolic distance on Poincare balls with varying curvatures c.

    Args:
        x: torch.Tensor of size [B, d] with hyperbolic queries
        v: torch.Tensor with hyperbolic queries, shape [n, d] if eval_mode is true else (B x d)
        c: torch.Tensor of size [B, 1] with absolute hyperbolic curvatures
        tang: [B, d]

    Return: torch,Tensor with hyperbolic distances, size B x 1 if eval_mode is False
            else B x n_entities matrix with all pairs distances
    """
    batchSize = x.shape[0]
    validSize = v.shape[0] # B or n
    rank = x.shape[1] # d
    sqrt_c = c ** 0.5  #[B, 1]
    tnorm = torch.sum(tang * tang, dim=-1, keepdim=True)  #[B, 1]
    # print(c.shape, tnorm.shape)
    lbd = 1 / (1 - c * tnorm).clamp_min(MIN_NORM)  #[B, 1]
    if eval_mode:
        vnorm = torch.norm(v, p=2, dim=-1, keepdim=True).transpose(0, 1) # [1, n]
        # gammar_ =  tanh(sqrt_c * lbd * vnorm) / (sqrt_c * vnorm).clamp_min(MIN_NORM) # [B, n]
        gammar_ =  tanh(sqrt_c * lbd * vnorm) / sqrt_c.clamp_min(MIN_NORM) # [B, n]
        gammar_ = gammar_.reshape(batchSize, -1, 1) #[B, n, 1]
        y = gammar_ * v.reshape(1, validSize, -1) #[B, n, d]
        y = y.reshape(-1, rank) #[B*n, d]
        t_ = torch.stack([tang]*validSize, dim=1).view(-1,rank) #[B*n, d]
        extend_c = torch.stack([c]*validSize,dim=1).view(-1,1) #[B*n, 1]
        y = mobius_add(t_, y, extend_c) #[B*n, d]

        x_ = torch.stack([x]*validSize,dim=1).view(-1,rank) #[B*n, d]
        # assert False, 'batchSize{}, validSize{}, ranfk{}, {} {}'.format(batchSize, validSize, rank, x_.shape, y.shape)
        dist = hyp_distance(x_, y, extend_c).reshape(batchSize, -1)  #[B*n, 1] -> [B, n]
    else:
        vnorm = torch.norm(v, p=2, dim=-1, keepdim=True) # [B, 1]
        # y = tanh(sqrt_c * lbd * vnorm) * v / (sqrt_c * vnorm).clamp_min(MIN_NORM) # [B, d]
        y = tanh(sqrt_c * lbd * vnorm) * v / sqrt_c.clamp_min(MIN_NORM) # [B, d]
        y = mobius_add(tang, y, c) #[B, d]

        dist = hyp_distance(x, y, c)

    return dist
