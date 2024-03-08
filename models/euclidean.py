"""Euclidean Knowledge Graph embedding models where embeddings are in real space."""
import numpy as np
import torch
from torch import nn

from models.base import KGModel
from utils.euclidean import euc_sqdistance, givens_rotations, givens_reflection

EUC_MODELS = ["TransE", "CP", "MurE", "RotE", "RefE", "AttE", 'HolmEE']


class BaseE(KGModel):
    """Euclidean Knowledge Graph Embedding models.

    Attributes:
        sim: similarity metric to use (dist for distance and dot for dot product)
    """

    def __init__(self, args):
        super(BaseE, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)

    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities."""
        if eval_mode:
            return self.entity.weight, self.bt.weight
        else:
            return self.entity(queries[:, 2]), self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        if self.sim == "dot":
            if eval_mode:
                score = lhs_e @ rhs_e.transpose(0, 1)
            else:
                score = torch.sum(lhs_e * rhs_e, dim=-1, keepdim=True)
        else:
            score = - euc_sqdistance(lhs_e, rhs_e, eval_mode)
        return score


class TransE(BaseE):
    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""

    def __init__(self, args):
        super(TransE, self).__init__(args)
        self.sim = "dist"

    def get_queries(self, queries):
        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])
        lhs_e = head_e + rel_e
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases
    
    def get_queries_2rel(self, lhs_e, queries):
        lhs_e, lhs_biases = lhs_e
        rel2_e = self.rel(queries[:, 1])
        lhs_e = lhs_e + rel2_e
        return lhs_e, lhs_biases


class CP(BaseE):
    """Canonical tensor decomposition https://arxiv.org/pdf/1806.07297.pdf"""

    def __init__(self, args):
        super(CP, self).__init__(args)
        self.sim = "dot"

    def get_queries(self, queries: torch.Tensor):
        """Compute embedding and biases of queries."""
        return self.entity(queries[:, 0]) * self.rel(queries[:, 1]), self.bh(queries[:, 0])


class MurE(BaseE):
    """Diagonal scaling https://arxiv.org/pdf/1905.09791.pdf"""

    def __init__(self, args):
        super(MurE, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.sim = "dist"

    def get_queries(self, queries: torch.Tensor):
        """Compute embedding and biases of queries."""
        lhs_e = self.rel_diag(queries[:, 1]) * self.entity(queries[:, 0]) + self.rel(queries[:, 1])
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases


class RotE(BaseE):
    """Euclidean 2x2 Givens rotations"""

    def __init__(self, args):
        super(RotE, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.sim = "dist"

    def get_queries(self, queries: torch.Tensor):
        """Compute embedding and biases of queries."""
        lhs_e = givens_rotations(self.rel_diag(queries[:, 1]), self.entity(queries[:, 0])) + self.rel(queries[:, 1])
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases


class RefE(BaseE):
    """Euclidean 2x2 Givens reflections"""

    def __init__(self, args):
        super(RefE, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.sim = "dist"

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        lhs = givens_reflection(self.rel_diag(queries[:, 1]), self.entity(queries[:, 0]))
        rel = self.rel(queries[:, 1])
        lhs_biases = self.bh(queries[:, 0])
        return lhs + rel, lhs_biases


class AttE(BaseE):
    """Euclidean attention model combining translations, reflections and rotations"""

    def __init__(self, args):
        super(AttE, self).__init__(args)
        self.sim = "dist"

        # reflection
        self.ref = nn.Embedding(self.sizes[1], self.rank)
        self.ref.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0

        # rotation
        self.rot = nn.Embedding(self.sizes[1], self.rank)
        self.rot.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0

        # attention
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.act = nn.Softmax(dim=1)
        self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()

    def get_reflection_queries(self, queries):
        lhs_ref_e = givens_reflection(
            self.ref(queries[:, 1]), self.entity(queries[:, 0])
        )
        return lhs_ref_e

    def get_rotation_queries(self, queries):
        lhs_rot_e = givens_rotations(
            self.rot(queries[:, 1]), self.entity(queries[:, 0])
        )
        return lhs_rot_e

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        lhs_ref_e = self.get_reflection_queries(queries).view((-1, 1, self.rank))
        lhs_rot_e = self.get_rotation_queries(queries).view((-1, 1, self.rank))

        # self-attention mechanism
        cands = torch.cat([lhs_ref_e, lhs_rot_e], dim=1)
        context_vec = self.context_vec(queries[:, 1]).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        lhs_e = torch.sum(att_weights * cands, dim=1) + self.rel(queries[:, 1])
        return lhs_e, self.bh(queries[:, 0])
    
    def get_queries_2rel(self, lhs_e, queries):
        res, lhs_biases = lhs_e
        lhs_ref_e = givens_reflection(
            self.ref(queries[:, 1]), res
        ).view((-1, 1, self.rank))
        lhs_rot_e = givens_rotations(
            self.rot(queries[:, 1]), res
        ).view((-1, 1, self.rank))

        # self-attention mechanism
        cands = torch.cat([lhs_ref_e, lhs_rot_e], dim=1)
        context_vec = self.context_vec(queries[:, 1]).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        lhs_e = torch.sum(att_weights * cands, dim=1) + self.rel(queries[:, 1])
        return lhs_e, lhs_biases


class HolmEE(BaseE):
    """Hyperbolic 2x2 Givens rotations"""
    def __init__(self, args):
        super(HolmEE, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.sim = "dist"

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        batch_size = lhs_e.shape[0]
        compl_rank = int(self.rank/2)
        # expand_c = torch.stack([c]*compl_rank,dim=1).view(-1,1)
        lhs_e = lhs_e[:, :compl_rank], lhs_e[:, compl_rank:]
        lhs_e = torch.cat([lhs_e[0].reshape(-1,1), lhs_e[1].reshape(-1,1)], dim=1) #[batchSize*rank/2, 2]
        rhs_e = rhs_e[:, :compl_rank], rhs_e[:, compl_rank:] #[batchSize, rank/2] / [entitySize, rank/2]
        rhs_e = torch.cat([rhs_e[0].reshape(-1,1), rhs_e[1].reshape(-1,1)], dim=1) #[batchSize*rank/2, 2] / [entitySize*rank/2, 2]
        
        if not eval_mode:
            dist = - euc_sqdistance(lhs_e, rhs_e, eval_mode) 
            dist = torch.sum(dist.view(batch_size,-1), dim=1, keepdim=True)
        else:
            dist = False
            lhs_e = lhs_e.view(batch_size, compl_rank, 2).chunk(compl_rank, dim=1)
            rhs_e = rhs_e.view(-1, compl_rank, 2).chunk(compl_rank, dim=1)
            for i in range(compl_rank):
                if i==0:
                    dist = - euc_sqdistance(lhs_e[i].squeeze(), rhs_e[i].squeeze(), eval_mode) 
                else:
                    dist -= euc_sqdistance(lhs_e[i].squeeze(), rhs_e[i].squeeze(), eval_mode)    
        return dist

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        batch_size = queries.shape[0]
        compl_rank = int(self.rank/2)
        # c = F.softplus(self.c[queries[:, 1]])
        # expand_c = torch.stack([c]*compl_rank,dim=1).view(-1,1)
        head = self.entity(queries[:, 0]) #[batchSize, self.rank] -> [batchSize, self.rank/4, 4]
        # relt, _ = torch.chunk(self.rel(queries[:, 1]), 2, dim=1) # translation
        relt = self.rel(queries[:, 1])
        head = torch.cat([head[:, :compl_rank].reshape(-1,1), head[:, compl_rank:].reshape(-1,1)],dim=1) #[batchSize*self.rank/2, 2]
        # head = expmap0(head, expand_c) #[batchSize*rank/2, 2]
        # head = head[:,0], head[:,1] #[batchSize*rank/2 x 1]

        relt = torch.cat([relt[:, :compl_rank].reshape(-1,1), relt[:, compl_rank:].reshape(-1,1)],dim=1) #[batchSize*self.rank/2 x 2]
        # relt = expmap0(relt, expand_c) #[batchSize*rank/2 x 2]
        # relt = relt[:,0], relt[:,1] #[batchSize*rank/2 x 1]

        relr = self.rel_diag(queries[:, 1])
        relr = relr[:, :compl_rank].reshape(-1,1), relr[:, compl_rank:].reshape(-1,1)
        
        # lhs = project(mobius_add(head, relt, expand_c), expand_c) #[batchSize*rank/2, 2]    
        # lhs = project(mobius_add(relt, head, expand_c), expand_c) #[batchSize*rank/2, 2]    
        lhs = relt + head
        lhs = lhs[:,:1], lhs[:,1:] #[batchSize*rank/2, 1] *2

        rel_norm = torch.sqrt(relr[0] ** 2 + relr[1] ** 2)
        cos = relr[0] / rel_norm
        sin = relr[1] / rel_norm


        x = (lhs[0] * cos - lhs[1] * sin).reshape(batch_size,-1) #[batchSize, rank/2]
        y = (lhs[0] * sin + lhs[1] * cos).reshape(batch_size,-1) #[batchSize, rank/2]

        res = torch.concat([x,y], dim=1)

        # res = givens_rotations(relr, lhs).reshape(queries[:, 1].shape[0],-1,2)
        # res = torch.cat([res[:,0], res[:,1]], dim=1)
        return res, self.bh(queries[:, 0])
    
    def get_queries_2rel(self, lhs_e, queries):
        res, lhs_biases = lhs_e
        batch_size = queries.shape[0]
        compl_rank = int(self.rank/2)
        # c = F.softplus(self.c[queries[:, 1]])
        # expand_c = torch.stack([c]*compl_rank,dim=1).view(-1,1)
        head = res
        # relt, _ = torch.chunk(self.rel(queries[:, 1]), 2, dim=1) # translation
        relt = self.rel(queries[:, 1])
        head = torch.cat([head[:, :compl_rank].reshape(-1,1), head[:, compl_rank:].reshape(-1,1)],dim=1) #[batchSize*self.rank/2, 2]
        # head = expmap0(head, expand_c) #[batchSize*rank/2, 2]
        # head = head[:,0], head[:,1] #[batchSize*rank/2 x 1]

        relt = torch.cat([relt[:, :compl_rank].reshape(-1,1), relt[:, compl_rank:].reshape(-1,1)],dim=1) #[batchSize*self.rank/2 x 2]
        # relt = expmap0(relt, expand_c) #[batchSize*rank/2 x 2]
        # relt = relt[:,0], relt[:,1] #[batchSize*rank/2 x 1]

        relr = self.rel_diag(queries[:, 1])
        relr = relr[:, :compl_rank].reshape(-1,1), relr[:, compl_rank:].reshape(-1,1)
        
        # lhs = project(mobius_add(head, relt, expand_c), expand_c) #[batchSize*rank/2, 2]    
        # lhs = project(mobius_add(relt, head, expand_c), expand_c) #[batchSize*rank/2, 2]    
        lhs = relt + head
        lhs = lhs[:,:1], lhs[:,1:] #[batchSize*rank/2, 1] *2

        rel_norm = torch.sqrt(relr[0] ** 2 + relr[1] ** 2)
        cos = relr[0] / rel_norm
        sin = relr[1] / rel_norm


        x = (lhs[0] * cos - lhs[1] * sin).reshape(batch_size,-1) #[batchSize, rank/2]
        y = (lhs[0] * sin + lhs[1] * cos).reshape(batch_size,-1) #[batchSize, rank/2]

        res = torch.concat([x,y], dim=1)

        # res = givens_rotations(relr, lhs).reshape(queries[:, 1].shape[0],-1,2)
        # res = torch.cat([res[:,0], res[:,1]], dim=1)
        return res, lhs_biases