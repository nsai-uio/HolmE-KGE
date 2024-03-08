"""Hyperbolic Knowledge Graph embedding models where all parameters are defined in tangent spaces."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.base import KGModel
from utils.euclidean import givens_rotations, givens_reflection
from utils.hyperbolic import mobius_add, expmap0, expmap, expmap1, project, hyp_distance_multi_c, hyp_distance, hyp_distance_multi_t, hyp_distance_multi_t2, hyp_distance_multi_t3

HYP_MODELS = ["RotH", "RefH", "AttH", 'HolmE', 'HolmE3D', 'GIE']


class BaseH(KGModel):
    """Trainable curvature for each relationship."""

    def __init__(self, args):
        super(BaseH, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], 2 * self.rank), dtype=self.data_type)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
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
            return self.entity.weight, self.bt.weight # [entity_n, 32] -> [n*16, 2] -> expmap_new ->  [n*16, 2] -> [n, 32]
        else:
            return self.entity(queries[:, 2]), self.bt(queries[:, 2]) # [batch, 32]

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e, c = lhs_e
        return - hyp_distance_multi_c(lhs_e, rhs_e, c, eval_mode) ** 2


class RotH(BaseH):
    """Hyperbolic 2x2 Givens rotations"""

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c[queries[:, 1]])
        head = expmap0(self.entity(queries[:, 0]), c)
        rel1, rel2 = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel1 = expmap0(rel1, c)
        rel2 = expmap0(rel2, c)
        lhs = project(mobius_add(head, rel1, c), c)
        res1 = givens_rotations(self.rel_diag(queries[:, 1]), lhs)
        res2 = mobius_add(res1, rel2, c)
        return (res2, c), self.bh(queries[:, 0])


class RefH(BaseH):
    """Hyperbolic 2x2 Givens reflections"""

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c[queries[:, 1]])
        rel, _ = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel = expmap0(rel, c)
        lhs = givens_reflection(self.rel_diag(queries[:, 1]), self.entity(queries[:, 0]))
        lhs = expmap0(lhs, c)
        res = project(mobius_add(lhs, rel, c), c)
        return (res, c), self.bh(queries[:, 0])

    def get_queries_2rel(self, lhs_e, queries):
        (res, _), lhs_biases = lhs_e
        c = F.softplus(self.c[queries[:, 1]])
        rel, _ = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel = expmap0(rel, c)
        lhs = givens_reflection(self.rel_diag(queries[:, 1]), res)
        lhs = expmap0(lhs, c)
        res = project(mobius_add(lhs, rel, c), c)
        return (res, c), self.bh(queries[:, 0])


class AttH(BaseH):
    """Hyperbolic attention model combining translations, reflections and rotations"""

    def __init__(self, args):
        super(AttH, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], 2 * self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], 2 * self.rank), dtype=self.data_type) - 1.0
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.context_vec.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.act = nn.Softmax(dim=1)
        if args.dtype == "double":
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).double().cuda()
        else:
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c[queries[:, 1]])
        head = self.entity(queries[:, 0])
        rot_mat, ref_mat = torch.chunk(self.rel_diag(queries[:, 1]), 2, dim=1)
        rot_q = givens_rotations(rot_mat, head).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head).view((-1, 1, self.rank))
        cands = torch.cat([ref_q, rot_q], dim=1)
        context_vec = self.context_vec(queries[:, 1]).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        lhs = expmap0(att_q, c)
        rel, _ = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel = expmap0(rel, c)
        res = project(mobius_add(lhs, rel, c), c)
        return (res, c), self.bh(queries[:, 0])
    
    def get_queries_2rel(self, lhs_e, queries):
        (res, _), lhs_biases = lhs_e
        c = F.softplus(self.c[queries[:, 1]])
        # head = self.entity(queries[:, 0])
        rot_mat, ref_mat = torch.chunk(self.rel_diag(queries[:, 1]), 2, dim=1)
        rot_q = givens_rotations(rot_mat, res).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, res).view((-1, 1, self.rank))
        cands = torch.cat([ref_q, rot_q], dim=1)
        context_vec = self.context_vec(queries[:, 1]).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        lhs = expmap0(att_q, c)
        rel, _ = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel = expmap0(rel, c)
        res = project(mobius_add(lhs, rel, c), c)
        return (res, c), lhs_biases

class HolmE(BaseH):
    """Hyperbolic 2x2 Givens rotations"""
    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e, c = lhs_e
        batch_size = lhs_e.shape[0]
        compl_rank = int(self.rank/2)
        expand_c = torch.stack([c]*compl_rank,dim=1).view(-1,1)
        lhs_e = lhs_e[:, :compl_rank], lhs_e[:, compl_rank:]
        lhs_e = torch.cat([lhs_e[0].reshape(-1,1), lhs_e[1].reshape(-1,1)], dim=1) #[batchSize*rank/2, 2]
        rhs_e = rhs_e[:, :compl_rank], rhs_e[:, compl_rank:] #[batchSize, rank/2] / [entitySize, rank/2]
        rhs_e = torch.cat([rhs_e[0].reshape(-1,1), rhs_e[1].reshape(-1,1)], dim=1) #[batchSize*rank/2, 2] / [entitySize*rank/2, 2]
        
        if not eval_mode:
            dist = - hyp_distance_multi_c(lhs_e, rhs_e, expand_c, eval_mode) ** 2
            dist = torch.sum(dist.view(batch_size,-1), dim=1, keepdim=True)
        else:
            dist = False
            lhs_e = lhs_e.view(batch_size, compl_rank, 2).chunk(compl_rank, dim=1)
            rhs_e = rhs_e.view(-1, compl_rank, 2).chunk(compl_rank, dim=1)
            for i in range(compl_rank):
                if i==0:
                    dist = - hyp_distance_multi_c(lhs_e[i].squeeze(), rhs_e[i].squeeze(), c, eval_mode) ** 2
                else:
                    dist -= hyp_distance_multi_c(lhs_e[i].squeeze(), rhs_e[i].squeeze(), c, eval_mode) ** 2        
        return dist

    def get_queries(self, queries):
        batch_size = queries.shape[0]
        compl_rank = int(self.rank/2)
        c = F.softplus(self.c[queries[:, 1]])
        expand_c = torch.stack([c]*compl_rank,dim=1).view(-1,1)
        head = self.entity(queries[:, 0]) #[batchSize, self.rank] -> [batchSize, self.rank/4, 4]
        relt, _ = torch.chunk(self.rel(queries[:, 1]), 2, dim=1) # translation
        head = torch.cat([head[:, :compl_rank].reshape(-1,1), head[:, compl_rank:].reshape(-1,1)],dim=1) #[batchSize*self.rank/2, 2]
        head = expmap0(head, expand_c) #[batchSize*rank/2, 2]
        # head = head[:,0], head[:,1] #[batchSize*rank/2 x 1]

        relt = torch.cat([relt[:, :compl_rank].reshape(-1,1), relt[:, compl_rank:].reshape(-1,1)],dim=1) #[batchSize*self.rank/2 x 2]
        relt = expmap0(relt, expand_c) #[batchSize*rank/2 x 2]
        # relt = relt[:,0], relt[:,1] #[batchSize*rank/2 x 1]

        relr = self.rel_diag(queries[:, 1])
        relr = relr[:, :compl_rank].reshape(-1,1), relr[:, compl_rank:].reshape(-1,1)
        
        lhs = project(mobius_add(head, relt, expand_c), expand_c) #[batchSize*rank/2, 2]    
        lhs = lhs[:,:1], lhs[:,1:] #[batchSize*rank/2, 1] *2

        rel_norm = torch.sqrt(relr[0] ** 2 + relr[1] ** 2)
        cos = relr[0] / rel_norm
        sin = relr[1] / rel_norm

        try:
            x = (lhs[0] * cos - lhs[1] * sin).reshape(batch_size,-1) #[batchSize, rank/2]
            y = (lhs[0] * sin + lhs[1] * cos).reshape(batch_size,-1) #[batchSize, rank/2]
        except:
            print(lhs[0].shape, lhs[1].shape)
            print(cos.shape, sin.shape)

        res = torch.concat([x,y], dim=1)

        # res = givens_rotations(relr, lhs).reshape(queries[:, 1].shape[0],-1,2)
        # res = torch.cat([res[:,0], res[:,1]], dim=1)
        return (res, c), self.bh(queries[:, 0])
    
    def get_queries_2rel(self, lhs_e, queries):
        (res, _), lhs_biases = lhs_e
        batch_size = queries.shape[0]
        compl_rank = int(self.rank/2)
        c = F.softplus(self.c[queries[:, 1]])
        expand_c = torch.stack([c]*compl_rank,dim=1).view(-1,1)
        # head = self.entity(queries[:, 0]) #[batchSize, self.rank] -> [batchSize, self.rank/4, 4]
        relt, _ = torch.chunk(self.rel(queries[:, 1]), 2, dim=1) # translation
        head = torch.cat([res[:, :compl_rank].reshape(-1,1), res[:, compl_rank:].reshape(-1,1)],dim=1) #[batchSize*self.rank/2, 2]
        head = expmap0(head, expand_c) #[batchSize*rank/2, 2]
        # head = head[:,0], head[:,1] #[batchSize*rank/2 x 1]

        relt = torch.cat([relt[:, :compl_rank].reshape(-1,1), relt[:, compl_rank:].reshape(-1,1)],dim=1) #[batchSize*self.rank/2 x 2]
        relt = expmap0(relt, expand_c) #[batchSize*rank/2 x 2]
        # relt = relt[:,0], relt[:,1] #[batchSize*rank/2 x 1]

        relr = self.rel_diag(queries[:, 1])
        relr = relr[:, :compl_rank].reshape(-1,1), relr[:, compl_rank:].reshape(-1,1)
        
        lhs = project(mobius_add(head, relt, expand_c), expand_c) #[batchSize*rank/2, 2]    
        lhs = lhs[:,:1], lhs[:,1:] #[batchSize*rank/2, 1] *2

        rel_norm = torch.sqrt(relr[0] ** 2 + relr[1] ** 2)
        cos = relr[0] / rel_norm
        sin = relr[1] / rel_norm

        x = (lhs[0] * cos - lhs[1] * sin).reshape(batch_size,-1) #[batchSize, rank/2]
        y = (lhs[0] * sin + lhs[1] * cos).reshape(batch_size,-1) #[batchSize, rank/2]

        res = torch.concat([x,y], dim=1)

        # res = givens_rotations(relr, lhs).reshape(queries[:, 1].shape[0],-1,2)
        # res = torch.cat([res[:,0], res[:,1]], dim=1)
        return (res, c), lhs_biases

class HolmE3D(BaseH):
    """Hyperbolic 2x2 Givens rotations"""
    def __init__(self, args):
        super(HolmE3D, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], 2 * self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], 2 * self.rank), dtype=self.data_type) - 1.0

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e, c = lhs_e
        batch_size = lhs_e.shape[0]
        prod_rank = int(self.rank/3)
        expand_c = torch.stack([c]*prod_rank,dim=1).view(-1,1)
        lhs_e = torch.chunk(lhs_e, 3, dim=1)
        lhs_e = torch.cat([lhs_e[0].reshape(-1,1), lhs_e[1].reshape(-1,1), lhs_e[2].reshape(-1,1)], dim=1) #[batchSize*rank/3, 3]
        rhs_e = torch.chunk(rhs_e, 3, dim=1) #[batchSize, rank/3] / [entitySize, rank/3]
        rhs_e = torch.cat([rhs_e[0].reshape(-1,1), rhs_e[1].reshape(-1,1), rhs_e[2].reshape(-1,1)], dim=1) #[batchSize*rank/3, 3] / [entitySize*rank/3, 3]        
        if not eval_mode:
            dist = - hyp_distance_multi_c(lhs_e, rhs_e, expand_c, eval_mode) ** 2
            dist = torch.sum(dist.view(batch_size,-1), dim=1, keepdim=True)
        else:
            dist = False
            lhs_e = lhs_e.view(batch_size, prod_rank, 3).chunk(prod_rank, dim=1)
            rhs_e = rhs_e.view(-1, prod_rank, 3).chunk(prod_rank, dim=1)
            for i in range(prod_rank):
                if i==0:
                    dist = - hyp_distance_multi_c(lhs_e[i].squeeze(), rhs_e[i].squeeze(), c, eval_mode) ** 2
                else:
                    dist -= hyp_distance_multi_c(lhs_e[i].squeeze(), rhs_e[i].squeeze(), c, eval_mode) ** 2        
        return dist

    def rotation_3d(self, relr, entity):
        relr1 = torch.chunk(relr[0], 3, dim=1) #[batchSize, rank/3] *3
        relr2 = torch.chunk(relr[1], 3, dim=1) #[batchSize, rank/3] *3
        x, y, z = torch.chunk(entity, 3, dim=1) #[batchSize*rank/3, 1] *3

        # xr = torch.cos(yaw)*torch.cos(pitch) * x + \
        #     (torch.cos(yaw)*torch.sin(pitch)*torch.sin(roll) - torch.sin(yaw)*torch.cos(roll)) * y +\
        #     (torch.cos(yaw)*torch.sin(pitch)*torch.cos(roll) + torch.sin(yaw)*torch.sin(roll)) * z #[batchSize*rank/3, 1]
        # yr = torch.sin(yaw)*torch.cos(pitch) * x + \
        #     (torch.sin(yaw)*torch.sin(pitch)*torch.sin(roll) + torch.cos(yaw)*torch.cos(roll)) * y +\
        #     (torch.sin(yaw)*torch.sin(pitch)*torch.cos(roll) - torch.cos(yaw)*torch.sin(roll)) * z #[batchSize*rank/3, 1]
        # zr = -torch.sin(pitch) * x + \
        #     torch.cos(pitch)*torch.sin(roll) * y +\
        #     torch.cos(pitch)*torch.cos(roll) * z  #[batchSize*rank/3, 1]  

        yaw1, pitch1, roll1 = relr1[0].reshape(-1,1), relr1[1].reshape(-1,1), relr1[2].reshape(-1,1) #[batchSize*rank/3, 1] *3
        yaw2, pitch2, roll2 = relr2[0].reshape(-1,1), relr2[1].reshape(-1,1), relr2[2].reshape(-1,1) #[batchSize*rank/3, 1] *3

        yaw_norm = torch.sqrt(yaw1 ** 2 + yaw2 ** 2)
        pitch_norm = torch.sqrt(pitch1 ** 2 + pitch2 ** 2)
        roll_norm = torch.sqrt(roll1 ** 2 + roll2 ** 2)

        yaw_cos = yaw1 / yaw_norm
        yaw_sin = yaw2 / yaw_norm
        pitch_cos = pitch1 / pitch_norm
        pitch_sin = pitch2 / pitch_norm
        roll_cos = roll1 / roll_norm
        roll_sin = roll2 / roll_norm
        xr = yaw_cos*pitch_cos * x + \
            (yaw_cos*pitch_sin*roll_sin - yaw_sin*roll_cos) * y +\
            (yaw_cos*pitch_sin*roll_cos + yaw_sin*roll_sin) * z #[batchSize*rank/3, 1]
        yr = yaw_sin*pitch_cos * x + \
            (yaw_sin*pitch_sin*roll_sin + yaw_cos*roll_cos) * y +\
            (yaw_sin*pitch_sin*roll_cos - yaw_cos*roll_sin) * z #[batchSize*rank/3, 1]
        zr = -pitch_sin * x + \
            pitch_cos*roll_sin * y +\
            pitch_cos*roll_cos * z  #[batchSize*rank/3, 1]       
        head_r = torch.concat([xr, yr, zr], dim=1) #[batchSize*rank/3, 3]
        return head_r

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        batch_size = queries.shape[0]
        prod_rank = int(self.rank/3)
        c = F.softplus(self.c[queries[:, 1]])
        expand_c = torch.stack([c]*prod_rank,dim=1).view(-1,1)
        head = self.entity(queries[:, 0]) #[batchSize, self.rank] 
        relt, _ = torch.chunk(self.rel(queries[:, 1]), 2, dim=1) # translation [batchSize, self.rank] 

        head = torch.chunk(head, 3, dim=1)
        head = torch.cat([head[0].reshape(-1,1), head[1].reshape(-1,1), head[2].reshape(-1,1)],dim=1) #[batchSize*self.rank/3, 3]
        head = expmap0(head, expand_c) #[batchSize*rank/3, 3]
        
        relt = torch.chunk(relt, 3, dim=1)
        relt = torch.cat([relt[0].reshape(-1,1), relt[1].reshape(-1,1), relt[2].reshape(-1,1)],dim=1) #[batchSize*self.rank/3 x 3]
        relt = expmap0(relt, expand_c) #[batchSize*rank/3, 3]

        relr = torch.chunk(self.rel_diag(queries[:, 1]), 2, dim=1)
 
        head_r = self.rotation_3d(relr, head)       
        
        lhs = project(mobius_add(relt, head_r, expand_c), expand_c) #[batchSize*rank/3, 3]   
        lhs = torch.chunk(lhs, 3, dim=1) #[batchSize*rank/3, 1] *3
        lhs = lhs[0].reshape(batch_size, -1), lhs[1].reshape(batch_size, -1), lhs[2].reshape(batch_size, -1) #[batchSize, rank/3] *3
        res = torch.cat(lhs, dim=1) #[batchSize, rank]    
        return (res, c), self.bh(queries[:, 0])
    
class GIE(BaseH):
    def __init__(self, args):
        super(BaseH, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], 2 * self.rank), dtype=self.data_type)
        # self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        # self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.multi_c = args.multi_c
        c_init = torch.ones((self.sizes[1], 1), dtype=self.data_type)
        c_init1 = torch.ones((self.sizes[1], 1), dtype=self.data_type)
        c_init2 = torch.ones((self.sizes[1], 1), dtype=self.data_type)
        if self.multi_c:
            self.c = nn.Parameter(c_init, requires_grad=True)
            self.c1= nn.Parameter(c_init1, requires_grad=True)
            self.c2= nn.Parameter(c_init2, requires_grad=True)
        else:
            self.c = nn.Parameter(c_init, requires_grad=False)
            self.c1 = nn.Parameter(c_init1, requires_grad=False)
            self.c2 = nn.Parameter(c_init2, requires_grad=False)

        # super(GIE, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], 2 * self.rank)
        self.rel_diag1 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag2 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], 2 * self.rank), dtype=self.data_type) - 1.0
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.context_vec.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.act = nn.Softmax(dim=1)
        if args.dtype == "double":
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).double().cuda()
        else:
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()

    def get_queries(self, queries):
        c1 = F.softplus(self.c1[queries[:, 1]])
        head1 = expmap0(self.entity(queries[:, 0]), c1)
        rel1, rel2 = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel1 = expmap0(rel1, c1)
        rel2 = expmap0(rel2, c1)
        lhs = project(mobius_add(head1, rel1, c1), c1)
        res1 = givens_rotations(self.rel_diag1(queries[:, 1]), lhs).view(-1, 1, self.rank)

        c2 = F.softplus(self.c2[queries[:, 1]])
        head2 = expmap0(self.entity(queries[:, 0]), c2)
        rel21, rel22 = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel21 = expmap0(rel21, c2)
        rel22 = expmap0(rel22, c2)
        lhss = project(mobius_add(head2, rel21, c2), c2)
        res2 = givens_rotations(self.rel_diag2(queries[:, 1]), lhss).view(-1, 1, self.rank)

        c = F.softplus(self.c[queries[:, 1]])
        head = self.entity(queries[:, 0])
        rot_mat, _ = torch.chunk(self.rel_diag(queries[:, 1]), 2, dim=1)
        rot_q = givens_rotations(rot_mat, head).view(-1, 1, self.rank)

        cands = torch.cat([res1, res2, rot_q], dim=1)
        context_vec = self.context_vec(queries[:, 1]).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)

        lhs = expmap0(att_q, c)
        rel, _ = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel = expmap0(rel, c)
        res = project(mobius_add(lhs, rel, c), c)
        return (res, c), self.bh(queries[:, 0])
    
    def get_queries_2rel(self, lhs_e, queries):
        (res, _), lhs_biases = lhs_e
        c1 = F.softplus(self.c1[queries[:, 1]])
        head1 = expmap0(res, c1)
        rel1, rel2 = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel1 = expmap0(rel1, c1)
        rel2 = expmap0(rel2, c1)
        lhs = project(mobius_add(head1, rel1, c1), c1)
        res1 = givens_rotations(self.rel_diag1(queries[:, 1]), lhs).view(-1, 1, self.rank)

        c2 = F.softplus(self.c2[queries[:, 1]])
        head2 = expmap0(res, c2)
        rel21, rel22 = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel21 = expmap0(rel21, c2)
        rel22 = expmap0(rel22, c2)
        lhss = project(mobius_add(head2, rel21, c2), c2)
        res2 = givens_rotations(self.rel_diag2(queries[:, 1]), lhss).view(-1, 1, self.rank)

        c = F.softplus(self.c[queries[:, 1]])
        head = res
        rot_mat, _ = torch.chunk(self.rel_diag(queries[:, 1]), 2, dim=1)
        rot_q = givens_rotations(rot_mat, head).view(-1, 1, self.rank)

        cands = torch.cat([res1, res2, rot_q], dim=1)
        context_vec = self.context_vec(queries[:, 1]).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)

        lhs = expmap0(att_q, c)
        rel, _ = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel = expmap0(rel, c)
        res = project(mobius_add(lhs, rel, c), c)
        return (res, c), lhs_biases

