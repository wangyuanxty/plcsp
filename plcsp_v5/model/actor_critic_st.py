"""Set Transformer variants of the E2E-MAPPO actors and global critic.

Two independent Set Transformers:
  * operation side: 36 task tokens -> per-task embeddings + pooled context
  * machine side:   one token per machine -> per-machine embeddings + pooled context

Interfaces match ``model/actor_critic.py`` so ``PPOAlgorithm`` can swap them.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer.train_device import device
from model.gcn_mlp import Encoder, MLPActor, MLPCritic, aggr_obs
from model.gat import GATLayer
from model.set_transformer import SetTransformerEncoder
from algorithm.agent_func import select_operation_action, greedy_select_action


def _as_tensor(x, dtype):
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.from_numpy(np.copy(x)).to(device=device, dtype=dtype)


TOKEN_CONTEXT_DIM = 8
EDGE_CONTEXT_DIM = 4
TASK_CONTEXT_DIM = 13


def _token_context_features(x, adj, n_machine, trans=None, idle=None,
                            task_ctx=None):
    """Node-level consequences of the precedence edges.

    Returns a concatenation of:
      * 8 base features (delay aggregates, latest predecessor finish, slack,
        remaining operations/duration);
      * 4 edge features (max/sum transport, max/sum idle) when provided;
      * 13 task-context features when provided.
    """
    if adj is None:
        base = torch.zeros(x.size(0), x.size(1), TOKEN_CONTEXT_DIM,
                           device=x.device, dtype=x.dtype)
    else:
        a = _as_tensor(adj, torch.float32)
        b, n, _ = a.shape
        eye = torch.eye(n, device=a.device, dtype=torch.bool).unsqueeze(0)
        edge = (a > 0) & (~eye)
        delay = torch.where(edge, a - 1.0, torch.zeros_like(a))
        max_delay = delay.amax(dim=-1)
        sum_delay = delay.sum(dim=-1)
        count = edge.float().sum(dim=-1)
        mean_delay = sum_delay / count.clamp(min=1.0)

        ft = x[:, :, 1]
        pred_ft = torch.where(edge, ft.unsqueeze(1).expand_as(edge),
                              torch.zeros_like(a))
        max_pred_ft = pred_ft.amax(dim=-1)
        st = x[:, :, 0]
        slack = st - max_pred_ft

        job = x[:, :, 8].long()
        sched = x[:, :, 3]
        dur = (ft - st).clamp(min=0.0)
        same_job = (job.unsqueeze(-1) == job.unsqueeze(-2)).float()
        remaining_ops = ((1.0 - sched).unsqueeze(1) * same_job).sum(dim=-1)
        remaining_dur = (dur.unsqueeze(1) * (1.0 - sched).unsqueeze(1)
                         * same_job).sum(dim=-1)

        base = torch.stack([
            torch.log1p(max_delay.clamp(min=0.0)),
            torch.log1p(sum_delay.clamp(min=0.0)),
            torch.log1p(mean_delay.clamp(min=0.0)),
            count,
            torch.log1p(max_pred_ft.clamp(min=0.0)),
            slack,
            remaining_ops / max(1, n_machine),
            torch.log1p(remaining_dur.clamp(min=0.0)),
        ], dim=-1)

    if trans is not None and idle is not None:
        t = _as_tensor(trans, torch.float32)
        d = _as_tensor(idle, torch.float32)
        t_count = (t > 0).float().sum(dim=-1).clamp(min=1.0)
        d_count = (d > 0).float().sum(dim=-1).clamp(min=1.0)
        edge_feats = torch.stack([
            torch.log1p(t.amax(dim=-1).clamp(min=0.0)),
            torch.log1p((t.sum(dim=-1) / t_count).clamp(min=0.0)),
            torch.log1p(d.amax(dim=-1).clamp(min=0.0)),
            torch.log1p((d.sum(dim=-1) / d_count).clamp(min=0.0)),
        ], dim=-1)
        base = torch.cat([base, edge_feats], dim=-1)

    if task_ctx is not None:
        base = torch.cat([base, _as_tensor(task_ctx, torch.float32)], dim=-1)
    return base


class Operation_Actor_SetTransformer(nn.Module):
    """Job/operation actor using a Set Transformer over task tokens."""

    def __init__(self, configs):
        super().__init__()
        self.n_job = configs['n_job']
        self.n_total_task = configs['n_job'] * configs['n_machine']
        self.n_machine = configs['n_machine']
        self.use_token_context = bool(configs.get('st_token_context', 0))
        self.use_full_context = bool(configs.get('st_full_context', 0))
        self.use_task_features = bool(configs.get('st_task_features', 0))
        self.use_edge_features = bool(configs.get('st_edge_features', 0))
        hidden = configs['gcn_hidden_dim']
        ctx_dim = 0
        if self.use_token_context:
            ctx_dim = TOKEN_CONTEXT_DIM
            if self.use_edge_features:
                ctx_dim += EDGE_CONTEXT_DIM
            if self.use_task_features:
                ctx_dim += TASK_CONTEXT_DIM
        in_dim = configs['gcn_input_dim'] + ctx_dim
        self.encoder = SetTransformerEncoder(
            input_dim=in_dim,
            hidden_dim=hidden,
            num_layers=max(1, int(configs['gcn_layer']) - 1),
            num_heads=4,
            num_seeds=1,
        ).to(device)
        self.num_heads = 4
        self.use_graph_bias = bool(configs.get('st_graph_bias', 0))
        if self.use_graph_bias:
            self.edge_bias = nn.Parameter(torch.zeros(self.num_heads))
            self.none_bias = nn.Parameter(torch.zeros(self.num_heads))
        self.m_pool_init = nn.Parameter(torch.zeros(hidden)).to(device)
        self.o_policy = MLPActor(
            configs['mlp_actor_layer'], hidden * 3, hidden, 1).to(device)
        self.job_critic = MLPCritic(
            configs['mlp_critic_layer'], configs['critic_input_dim'],
            configs['critic_hidden_dim'], 2).to(device)

    def forward(self, x_fea, graph_pool_avg, padded_nei, adj, candidate,
                h_g_m_pooled, mask_operation, use_greedy=False,
                trans_wrk=None, idle_wrk=None, task_ctx=None,
                score_bias=None, pref_emb=None):
        x = _as_tensor(x_fea, torch.float32)
        candidate = _as_tensor(candidate, torch.long)
        mask = _as_tensor(mask_operation, torch.bool)
        batch = candidate.size(0)
        x = x.reshape(batch, self.n_total_task, -1)
        if self.use_token_context:
            x = torch.cat(
                [x, _token_context_features(
                    x, adj, self.n_machine,
                    trans_wrk, idle_wrk, task_ctx)], dim=-1)
        if self.use_graph_bias and adj is not None:
            adj_t = (_as_tensor(adj, torch.float32) > 0).float()
            b, n, _ = adj_t.shape
            bias = (adj_t.unsqueeze(1) * self.edge_bias.view(1, -1, 1, 1)
                    + (1.0 - adj_t).unsqueeze(1)
                    * self.none_bias.view(1, -1, 1, 1))
            attn_bias = bias.reshape(b * self.num_heads, n, n)
        else:
            attn_bias = None
        h_nodes, h_pool = self.encoder(x, attn_bias=attn_bias)

        idx = candidate.unsqueeze(-1).expand(-1, -1, h_nodes.size(-1))
        candidate_feature = torch.gather(h_nodes, 1, idx)       # (B, n_job, H)
        h_pool_rep = h_pool.unsqueeze(1).expand_as(candidate_feature)
        if h_g_m_pooled is None:
            m_pool = self.m_pool_init.unsqueeze(0).expand(batch, -1)
        else:
            m_pool = _as_tensor(h_g_m_pooled, torch.float32)
        m_pool_rep = m_pool.unsqueeze(1).expand_as(candidate_feature)

        feat = torch.cat([candidate_feature, h_pool_rep, m_pool_rep], dim=-1)
        scores = self.o_policy(feat).squeeze(-1)                 # (B, n_job)
        if score_bias is not None:  # [v5 hier] fixed top-level rule bias
            scores = scores + score_bias.to(
                device=scores.device, dtype=scores.dtype)
        scores = scores.masked_fill(mask, float('-inf'))
        prob = F.softmax(scores, dim=-1)

        if use_greedy:
            task_index, action_index, log_a = greedy_select_action(prob, candidate)
        else:
            task_index, action_index, log_a = select_operation_action(prob, candidate)
        job_v = self.job_critic(h_pool)
        return task_index, action_index, log_a, prob, h_pool, job_v


class Machine_Actor_SetTransformer(nn.Module):
    """Machine actor using a Set Transformer over machine tokens."""

    def __init__(self, configs):
        super().__init__()
        hidden = configs['machine_hidden_dim']
        self.n_machine = configs['n_machine']
        self.machine_ctx_dim = 8 if configs.get('st_full_context') else 3
        self.use_machine_id = bool(configs.get('machine_id_emb', 0))
        self.no_attn = bool(configs.get('machine_no_attn', 0))
        self.logit_scale = float(configs.get('machine_logit_scale', 1.0))
        self.use_bn = bool(configs.get('machine_bn', 0))
        self.use_intra = bool(configs.get('machine_intra_attn', 1))
        if self.use_bn:
            self.machine_bn = nn.BatchNorm1d(hidden).to(device)
        if self.use_machine_id:
            self.machine_id_emb = nn.Embedding(
                self.n_machine, hidden).to(device)
        if self.use_intra:
            self.m1_proj = nn.Linear(
                10 if configs.get('use_agv') else 6, hidden, bias=False).to(device)
            self.m2_proj = nn.Linear(8, hidden, bias=False).to(device)
            self.m3_proj = nn.Linear(
                self.machine_ctx_dim, hidden, bias=False).to(device)
            self.token_encoder = SetTransformerEncoder(
                input_dim=hidden,
                hidden_dim=hidden,
                num_layers=1,
                num_heads=2,
                num_seeds=1,
                project_input=False,
            ).to(device)
        else:
            self.input_proj = nn.Sequential(
                nn.Linear((10 if configs.get('use_agv') else 6) + 8
                          + self.machine_ctx_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            ).to(device)
        self.encoder = SetTransformerEncoder(
            input_dim=hidden,
            hidden_dim=hidden,
            num_layers=(0 if configs.get('machine_no_sab') else 3),
            num_heads=4,
            num_seeds=1,
            project_input=False,
        ).to(device)
        self.m_policy = MLPActor(
            configs['mlp_actor_layer'], hidden * 3, hidden, 1).to(device)
        self.machine_critic = MLPCritic(
            configs['mlp_critic_layer'], configs['critic_input_dim'],
            configs['critic_hidden_dim'], 2).to(device)

    def forward(self, machine_fea_1, machine_fea_2, h_pooled_o, machine_mask,
                machine_fea_3=None, score_bias=None, pref_emb=None):
        m1 = _as_tensor(machine_fea_1, torch.float32)
        m2 = _as_tensor(machine_fea_2, torch.float32)
        mask = _as_tensor(machine_mask, torch.bool).squeeze(1)   # (B, m)

        if machine_fea_3 is None:
            m3 = torch.zeros(m1.size(0), m1.size(1), self.machine_ctx_dim,
                             device=m1.device, dtype=m1.dtype)
        else:
            m3 = _as_tensor(machine_fea_3, torch.float32)
        if self.use_intra:
            tokens = torch.stack(
                [self.m1_proj(m1), self.m2_proj(m2), self.m3_proj(m3)],
                dim=2)                                   # (B, m, 3, H)
            b, m, t, h = tokens.shape
            _, fused = self.token_encoder(tokens.reshape(b * m, t, h))
            feat = fused.reshape(b, m, h)
        else:
            feat = self.input_proj(torch.cat([m1, m2, m3], dim=-1))
        if self.use_machine_id:
            ids = torch.arange(self.n_machine, device=feat.device)
            feat = feat + self.machine_id_emb(ids).unsqueeze(0)
        if self.no_attn:
            h_nodes = feat
            h_pool = feat.mean(dim=1)
        else:
            h_nodes, h_pool = self.encoder(feat)                     # (B,m,H), (B,H)
        if self.use_bn:
            b, m, h = h_nodes.shape
            h_nodes = self.machine_bn(h_nodes.reshape(-1, h)).reshape(b, m, h)
        h_pool_rep = h_pool.unsqueeze(1).expand_as(h_nodes)
        o_pool = _as_tensor(h_pooled_o, torch.float32)
        o_pool_rep = o_pool.unsqueeze(1).expand_as(h_nodes)

        feat2 = torch.cat([h_nodes, h_pool_rep, o_pool_rep], dim=-1)
        scores = self.m_policy(feat2).squeeze(-1)                # (B, m)
        scores = scores * self.logit_scale
        if score_bias is not None:  # [v5 hier] fixed top-level rule bias
            scores = scores + score_bias.to(
                device=scores.device, dtype=scores.dtype)
        scores = scores.masked_fill(mask, float('-inf'))
        prob = F.softmax(scores, dim=-1)
        machine_v = self.machine_critic(h_pool)
        return prob, h_pool, machine_v


class Global_Critic_SetTransformer(nn.Module):
    """Centralized 4-target critic built from two Set Transformers."""

    def __init__(self, configs):
        super().__init__()
        self.n_total_task = configs['n_job'] * configs['n_machine']
        self.n_machine = configs['n_machine']
        self.use_token_context = bool(configs.get('st_token_context', 0))
        self.use_full_context = bool(configs.get('st_full_context', 0))
        self.use_task_features = bool(configs.get('st_task_features', 0))
        self.use_edge_features = bool(configs.get('st_edge_features', 0))
        hidden = configs['gcn_hidden_dim']
        ctx_dim = 0
        if self.use_token_context:
            ctx_dim = TOKEN_CONTEXT_DIM
            if self.use_edge_features:
                ctx_dim += EDGE_CONTEXT_DIM
            if self.use_task_features:
                ctx_dim += TASK_CONTEXT_DIM
        in_dim = configs['gcn_input_dim'] + ctx_dim
        self.task_enc = SetTransformerEncoder(
            input_dim=in_dim,
            hidden_dim=hidden,
            num_layers=max(1, int(configs['gcn_layer']) - 1),
            num_heads=4,
            num_seeds=1,
        ).to(device)
        self.machine_ctx_dim = 8 if self.use_full_context else 3
        mh = configs['machine_hidden_dim']
        self.use_intra = bool(configs.get('machine_intra_attn', 1))
        if self.use_intra:
            self.m1_proj = nn.Linear(
                10 if configs.get('use_agv') else 6, mh, bias=False).to(device)
            self.m2_proj = nn.Linear(8, mh, bias=False).to(device)
            self.m3_proj = nn.Linear(
                self.machine_ctx_dim, mh, bias=False).to(device)
            self.token_encoder = SetTransformerEncoder(
                input_dim=mh, hidden_dim=mh, num_layers=1, num_heads=2,
                num_seeds=1, project_input=False).to(device)
        else:
            self.input_proj = nn.Sequential(
                nn.Linear((10 if configs.get('use_agv') else 6) + 8
                          + self.machine_ctx_dim, mh),
                nn.GELU(),
                nn.Linear(mh, mh),
            ).to(device)
        self.use_machine_id = bool(configs.get('machine_id_emb', 0))
        self.no_attn = bool(configs.get('machine_no_attn', 0))
        if self.use_machine_id:
            self.machine_id_emb = nn.Embedding(
                self.n_machine, mh).to(device)
        self.machine_enc = SetTransformerEncoder(
            input_dim=mh,
            hidden_dim=mh,
            num_layers=(0 if configs.get('machine_no_sab') else 3),
            num_heads=4,
            num_seeds=1,
            project_input=False,
        ).to(device)
        self.num_heads = 4
        self.use_graph_bias = bool(configs.get('st_graph_bias', 0))
        if self.use_graph_bias:
            self.edge_bias = nn.Parameter(torch.zeros(self.num_heads))
            self.none_bias = nn.Parameter(torch.zeros(self.num_heads))
        self.critic = MLPCritic(
            configs['mlp_critic_layer'],
            configs['critic_input_dim'] * 2,
            configs['critic_hidden_dim'], 4).to(device)

    def forward(self, x_fea, graph_pool_avg, adj, candidate,
                machine_fea1, machine_fea2, machine_fea3=None,
                trans_wrk=None, idle_wrk=None, task_ctx=None,
                pref_emb=None):
        x = _as_tensor(x_fea, torch.float32)
        candidate = _as_tensor(candidate, torch.long)
        m1 = _as_tensor(machine_fea1, torch.float32)
        m2 = _as_tensor(machine_fea2, torch.float32)
        if machine_fea3 is None:
            m3 = torch.zeros(m1.size(0), m1.size(1), self.machine_ctx_dim,
                             device=m1.device, dtype=m1.dtype)
        else:
            m3 = _as_tensor(machine_fea3, torch.float32)
        batch = candidate.size(0)
        if x.dim() == 2:
            x = x.reshape(batch, self.n_total_task, -1)
        if self.use_token_context:
            x = torch.cat(
                [x, _token_context_features(
                    x, adj, self.n_machine,
                    trans_wrk, idle_wrk, task_ctx)], dim=-1)

        if self.use_graph_bias and adj is not None:
            adj_t = (_as_tensor(adj, torch.float32) > 0).float()
            b, n, _ = adj_t.shape
            bias = (adj_t.unsqueeze(1) * self.edge_bias.view(1, -1, 1, 1)
                    + (1.0 - adj_t).unsqueeze(1)
                    * self.none_bias.view(1, -1, 1, 1))
            attn_bias = bias.reshape(b * self.num_heads, n, n)
        else:
            attn_bias = None
        _, h_task = self.task_enc(x, attn_bias=attn_bias)
        if self.use_intra:
            tokens = torch.stack(
                [self.m1_proj(m1), self.m2_proj(m2), self.m3_proj(m3)],
                dim=2)                                   # (B, m, 3, H)
            b, m, t, h = tokens.shape
            _, fused = self.token_encoder(tokens.reshape(b * m, t, h))
            feat = fused.reshape(b, m, h)
        else:
            feat = self.input_proj(torch.cat([m1, m2, m3], dim=-1))
        if self.use_machine_id:
            ids = torch.arange(self.n_machine, device=feat.device)
            feat = feat + self.machine_id_emb(ids).unsqueeze(0)
        if self.no_attn:
            h_mach = feat.mean(dim=1)
        else:
            _, h_mach = self.machine_enc(feat)
        return self.critic(torch.cat([h_mach, h_task], dim=-1))


class Global_Critic_Hybrid(nn.Module):
    """GIN task encoder + Set Transformer machine encoder + 4-target critic."""

    def __init__(self, configs):
        super().__init__()
        self.n_total_task = configs['n_job'] * configs['n_machine']
        self.n_machine = configs['n_machine']
        self.encoder = Encoder(
            num_layers=configs['gcn_layer'],
            num_mlp_layers=configs['mlp_fea_extract_layer'],
            input_dim=configs['gcn_input_dim'],
            hidden_dim=configs['gcn_hidden_dim'],
            learn_eps=configs['learn_eps'],
            neighbor_pooling_type=configs['neighbor_pooling_type'],
            device=device).to(device)
        self.use_full_context = bool(configs.get('st_full_context', 0))
        self.machine_ctx_dim = 8 if self.use_full_context else 3
        mh = configs['machine_hidden_dim']
        self.use_intra = bool(configs.get('machine_intra_attn', 1))
        if self.use_intra:
            self.m1_proj = nn.Linear(
                10 if configs.get('use_agv') else 6, mh, bias=False).to(device)
            self.m2_proj = nn.Linear(8, mh, bias=False).to(device)
            self.m3_proj = nn.Linear(
                self.machine_ctx_dim, mh, bias=False).to(device)
            self.token_encoder = SetTransformerEncoder(
                input_dim=mh, hidden_dim=mh, num_layers=1, num_heads=2,
                num_seeds=1, project_input=False).to(device)
        else:
            self.input_proj = nn.Sequential(
                nn.Linear((10 if configs.get('use_agv') else 6) + 8
                          + self.machine_ctx_dim, mh),
                nn.GELU(),
                nn.Linear(mh, mh),
            ).to(device)
        self.use_machine_id = bool(configs.get('machine_id_emb', 0))
        self.no_attn = bool(configs.get('machine_no_attn', 0))
        if self.use_machine_id:
            self.machine_id_emb = nn.Embedding(
                self.n_machine, mh).to(device)
        self.machine_enc = SetTransformerEncoder(
            input_dim=mh, hidden_dim=mh,
            num_layers=(0 if configs.get('machine_no_sab') else 3),
            num_heads=4, num_seeds=1,
            project_input=False,
        ).to(device)
        self.critic = MLPCritic(
            configs['mlp_critic_layer'],
            configs['critic_input_dim'] * 2,
            configs['critic_hidden_dim'], 4).to(device)

    def forward(self, x_fea, graph_pool_avg, adj, candidate,
                machine_fea1, machine_fea2, machine_fea3=None,
                trans_wrk=None, idle_wrk=None, task_ctx=None,
                pref_emb=None):
        x = _as_tensor(x_fea, torch.float32)
        candidate = _as_tensor(candidate, torch.long)
        m1 = _as_tensor(machine_fea1, torch.float32)
        m2 = _as_tensor(machine_fea2, torch.float32)
        if machine_fea3 is None:
            m3 = torch.zeros(m1.size(0), m1.size(1), self.machine_ctx_dim,
                             device=m1.device, dtype=m1.dtype)
        else:
            m3 = _as_tensor(machine_fea3, torch.float32)

        adj_t = _as_tensor(adj, torch.float32)
        if adj_t.dim() == 3:
            adj_sp = aggr_obs(adj_t.to(device).to_sparse(), self.n_total_task)
        else:
            adj_sp = adj_t
        # The GIN encoder expects x as (batch * n_task, input_dim).
        h_pool, _ = self.encoder(x=x, graph_pool=graph_pool_avg,
                                 padded_nei=None, adj=adj_sp)

        if self.use_intra:
            tokens = torch.stack(
                [self.m1_proj(m1), self.m2_proj(m2), self.m3_proj(m3)],
                dim=2)                                   # (B, m, 3, H)
            b, m, t, h = tokens.shape
            _, fused = self.token_encoder(tokens.reshape(b * m, t, h))
            feat = fused.reshape(b, m, h)
        else:
            feat = self.input_proj(torch.cat([m1, m2, m3], dim=-1))
        if self.use_machine_id:
            ids = torch.arange(self.n_machine, device=feat.device)
            feat = feat + self.machine_id_emb(ids).unsqueeze(0)
        if self.no_attn:
            h_mach = feat.mean(dim=1)
        else:
            _, h_mach = self.machine_enc(feat)
        return self.critic(torch.cat([h_mach, h_pool], dim=-1))


class Global_Critic_JobST_MachineGAT(nn.Module):
    """ST task encoder + the original 2-node GAT machine encoder."""

    def __init__(self, configs):
        super().__init__()
        self.n_job = configs['n_job']
        self.n_machine = configs['n_machine']
        self.n_total_task = self.n_job * self.n_machine
        self.use_token_context = bool(configs.get('st_token_context', 0))
        self.use_full_context = bool(configs.get('st_full_context', 0))
        self.use_task_features = bool(configs.get('st_task_features', 0))
        self.use_edge_features = bool(configs.get('st_edge_features', 0))
        hidden = configs['gcn_hidden_dim']
        ctx_dim = 0
        if self.use_token_context:
            ctx_dim = TOKEN_CONTEXT_DIM
            if self.use_edge_features:
                ctx_dim += EDGE_CONTEXT_DIM
            if self.use_task_features:
                ctx_dim += TASK_CONTEXT_DIM
        in_dim = configs['gcn_input_dim'] + ctx_dim
        self.task_enc = SetTransformerEncoder(
            input_dim=in_dim,
            hidden_dim=hidden,
            num_layers=max(1, int(configs['gcn_layer']) - 1),
            num_heads=4,
            num_seeds=1,
        ).to(device)

        self.machine_hidden_dim = configs['machine_hidden_dim']
        self.heads = 1
        self.bn = nn.BatchNorm1d(self.machine_hidden_dim).to(device)
        self.m_fea_1_fcl = nn.Linear(
            10 if configs.get('use_agv') else 6,
            self.machine_hidden_dim, bias=False).to(device)
        self.m_fea_2_fcl = nn.Linear(
            8, self.machine_hidden_dim, bias=False).to(device)
        self.activation = nn.ELU()
        self.gat_layer = GATLayer(
            in_features=self.machine_hidden_dim,
            out_features=self.machine_hidden_dim,
            n_heads=self.heads,
            concat=False,
            dropout=0,
            leaky_relu_slope=0.2).to(device)
        self.fcl_pooling = nn.Linear(
            self.machine_hidden_dim, self.machine_hidden_dim,
            bias=False).to(device)
        self.critic = MLPCritic(
            configs['mlp_critic_layer'],
            configs['critic_input_dim'] * 2,
            configs['critic_hidden_dim'], 4).to(device)

    def forward(self, x_fea, graph_pool_avg, adj, candidate,
                machine_fea1, machine_fea2, machine_fea3=None,
                trans_wrk=None, idle_wrk=None, task_ctx=None,
                pref_emb=None):
        x = _as_tensor(x_fea, torch.float32)
        if x.dim() == 2:
            x = x.reshape(candidate.size(0), self.n_total_task, -1)
        if self.use_token_context:
            x = torch.cat(
                [x, _token_context_features(
                    x, adj, self.n_machine,
                    trans_wrk, idle_wrk, task_ctx)], dim=-1)
        _, h_task = self.task_enc(x)

        m1 = _as_tensor(machine_fea1, torch.float32)
        m2 = _as_tensor(machine_fea2, torch.float32)
        m_fea1 = self.m_fea_1_fcl(m1).view(-1, self.machine_hidden_dim)
        m_fea2 = self.m_fea_2_fcl(m2).view(-1, self.machine_hidden_dim)
        pair = torch.stack([m_fea1, m_fea2], dim=1)
        adj_m = torch.zeros(pair.size(0), self.heads, 2, 2, device=device)
        adj_m[:, :, 0, :] = 1
        adj_m[:, :, 1, 1] = 1
        h = self.activation(self.gat_layer(h=pair, adj_mat=adj_m))
        h = self.activation(self.gat_layer(h=h, adj_mat=adj_m))
        h = self.gat_layer(h=h, adj_mat=adj_m)
        h_machine = h.mean(dim=-2).reshape(
            -1, self.n_machine, self.machine_hidden_dim)
        h_machine = self.bn(h_machine.reshape(
            -1, self.machine_hidden_dim)).reshape(
            -1, self.n_machine, self.machine_hidden_dim)
        h_pool = h_machine.mean(dim=1)
        return self.critic(torch.cat([h_pool, h_task], dim=-1))
