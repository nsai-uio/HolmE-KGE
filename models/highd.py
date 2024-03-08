import numpy as np
import torch
import torch.nn.functional as F

from torch import nn

from models.base import KGModel
from utils.euclidean import givens_rotations, givens_reflection
from utils.hyperbolic import mobius_add, expmap0, project, hyp_distance_multi_c

MIN_NORM = 1e-15
HD_MODELS = ["HighDHolmE"]

class BaseH(KGModel):
    """Trainable curvature for each relationship."""

    def __init__(self, args):
        super(BaseH, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        # self.dim = args.dim # dim=4,rank=32
        self.dim = 4
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type) # 32 = 4 * 8
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank * self.dim), dtype=self.data_type) #rotation 
        self.rel_trans = nn.Embedding(self.sizes[1], self.rank)
        self.rel_trans.weight.data = self.init_size * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) #translation
        self.multi_c = args.multi_c
        c_init = torch.ones((self.sizes[1], 1), dtype=self.data_type)
        if self.multi_c:
            self.c = nn.Parameter(c_init, requires_grad=True)
        else:
            self.c = nn.Parameter(c_init, requires_grad=False)

        # self.multi_c = args.multi_c
        # if self.multi_c:
        #     c_init = torch.ones((self.sizes[1], 1), dtype=self.data_type)
        # else:
        #     c_init = torch.ones((1, 1), dtype=self.data_type)
        # self.c = nn.Parameter(c_init, requires_grad=True)

    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities."""
        if eval_mode:
            return self.entity.weight, self.bt.weight
        else:
            return self.entity(queries[:, 2]), self.bt(queries[:, 2])


class HighDHolmE(BaseH):
    """Hyperbolic 2x2 Givens rotations"""
    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        batch_size = queries.shape[0]
        highD_rank = int(self.rank/self.dim) #32/4 = 8
        r_dim = self.dim
        c = F.softplus(self.c[queries[:, 1]]) # batchSize x 1
        head = self.entity(queries[:, 0])
        relt = self.rel_trans(queries[:, 1]) # translation
        relR = self.orthogo_tensor(self.rel(queries[:, 1]), highD_rank) #[relR[0], relR[1], ... relR[highD_rank]] , [batchSize, dim, dim]

        # R*h = t
        # [dim, dim] * [dim ,1] = [dim, 1]

        # product space: [highD_rank, dim, dim] * [highD_rank, dim ,1] = [highD_rank, dim, 1]
        # diag([dim, dim], high_rank)

        # [batchSize, highD_rank, dim, dim] * [batchSize, highD_rank, dim, 1] -> [batchSize, highD_rank, dim, 1]
        # diag([dim, dim], high_rank*batchSize)


        head = torch.chunk(head, highD_rank, dim=1) # [head[0], head[1] ... head[highD_rank]], [batchSize, dim]
        relt = torch.chunk(relt, highD_rank, dim=1) # [relt[0], relt[1] ... relt[highD_rank]], [batchSize, dim]

        res = [] # [res[0], res[1] ... res[highD_rank]], batchSize x dim
        for i in range(highD_rank):
            tmp_head = expmap0(head[i], c) # [batchSize, dim]
            tmp_relt = expmap0(relt[i], c) # [batchSize, dim]
            # tmp_rot_head = torch.mul(relR[i], tmp_head)  # [batch, dim, dim] * [batch, dim] => [batch, dim]
            tmp_rot_head = []
            for j in range(batch_size):
                mult_tmp = torch.mm(relR[i][j], tmp_head[j].view(r_dim,1))
                # print(relR[i][j].shape, tmp_head[j].view(r_dim,1).shape, mult_tmp.shape)
                mult_tmp = mult_tmp.view(1, r_dim) # [1, dim]
                tmp_rot_head.append(mult_tmp)
            tmp_rot_head = torch.cat(tmp_rot_head, dim=0) #[batch, dim]
                        
            res.append(project(mobius_add(tmp_relt, tmp_rot_head, c), c))
        res = torch.cat(res, dim=1)

        return (res, c), self.bh(queries[:, 0])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""      
        lhs_e, c = lhs_e
        batch_size = lhs_e.shape[0]
        highD_rank = int(self.rank/self.dim) #32/4 = 8
        lhs_e = torch.chunk(lhs_e, highD_rank, dim=1)
        rhs_e = torch.chunk(rhs_e, highD_rank, dim=1)

        for i in range(highD_rank):
            if i == 0:
                dist = - hyp_distance_multi_c(lhs_e[i], rhs_e[i], c, eval_mode) ** 2
            else:
                dist -= hyp_distance_multi_c(lhs_e[i], rhs_e[i], c, eval_mode) ** 2
        dist /= highD_rank

        return dist

    def orthogo_tensor(self, relr, highD_rank):
        '''
        x: [batchSize, rank*dim] = [batchSize, dim*dim*highD_rank]
        highD_rank: highD_rank
        output [relR[0], relR[1], ... relR[highD_rank]] , batchSize x dim x dim
        '''
        rel_l = torch.chunk(relr, highD_rank, dim=1) #[x[0], x[1] ... x[highD_rank]], [batchSize, dim*dim]
        rel_dim =self.dim
        relR_l = []
        for rel in rel_l: #[batchSize, dim*dim]
            relr = torch.chunk(rel, rel_dim, dim=1) #[batchSize, dim] * dim
            ort_relr = []
            for i in range(rel_dim):
                cur_row = relr[i] #[batchSize, dim]
                ort_relr_tmp = cur_row #[batchSize, dim]
                for r in ort_relr:
                    norm = torch.sum(r * cur_row, dim=-1, keepdim=True) #[batchSize, 1]
                    denorm = torch.sum(r * r, dim=-1, keepdim=True).clamp_min(MIN_NORM) #[batchSize, 1]
                    sub = norm/denorm *r #[batchSize, dim]
                    ort_relr_tmp = ort_relr_tmp - sub  #[batchSize, dim]
                relr_norm = ort_relr_tmp.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM) #[batchSize, 1]
                ort_relr_tmp = ort_relr_tmp/relr_norm  #[batchSize, dim]
                # assert False, ort_relr_tmp.shape
                ort_relr.append(ort_relr_tmp)
            relR = torch.stack(ort_relr, dim=2) #[batchSize, dim, dim]
            
            relR_l.append(relR)            
        return relR_l # #[batchSize, dim, dim] * highD_rank
