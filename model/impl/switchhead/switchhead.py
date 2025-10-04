import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math
from .cvmm import cvmm, cvmm_prepare_sel2, CVMMSel
from dataclasses import dataclass


@dataclass
class AttentionMask:
    def __init__(self, position_mask: Optional[torch.Tensor] = None, src_length_mask: Optional[torch.Tensor] = None):
        self.position_mask = position_mask
        self.src_length_mask = src_length_mask
    
    position_mask: Optional[torch.Tensor]
    src_length_mask: Optional[torch.Tensor]


KVCache = Optional[Dict[str, torch.Tensor]]


class SwitchHeadCore(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_experts: int, dropout: float = 0.0,
                 d_head: Optional[int] = None, expert_dropout: float = 0.0, moe_k: int = 2):

        super().__init__()

        self.input_size = d_model
        self.output_size = d_model
        self.pe_size = self.input_size
        self.expert_dropout = expert_dropout
        self.moe_k = moe_k
        self.attention_to_visualize = []
        self.selections_to_visualize = {}
        self.n_experts = n_experts

        self.n_heads = n_heads
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else lambda x: x
        self.projection_size = d_head or (d_model // n_heads)

        self.q = torch.nn.Linear(self.input_size, self.projection_size * self.n_heads, bias=False)
        self.k = torch.nn.Linear(self.input_size, self.projection_size * self.n_heads, bias=False)

        if self.n_experts > 1:
            self.v = torch.nn.Parameter(torch.empty(self.n_heads * self.n_experts, self.input_size, self.projection_size))
            self.o = torch.nn.Parameter(torch.empty(self.n_heads * self.n_experts, self.projection_size, self.output_size))
            self.sel_v = torch.nn.Parameter(torch.empty(self.n_heads * self.n_experts, self.input_size))
        else:
            self.v = torch.nn.Parameter(torch.empty(self.n_heads * self.projection_size, self.input_size))
            self.o = torch.nn.Parameter(torch.empty(self.output_size, self.n_heads * self.projection_size))

        self.sel_o = torch.nn.Parameter(torch.empty(self.n_heads * self.n_experts, self.input_size))

        self.register_buffer("scale", torch.full([1], 1.0 / math.sqrt(self.projection_size)), persistent=False)

    def generate_causal_attention_mask(self, sz: int) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=self.q.weight.device), diagonal=1)

    @torch.no_grad
    def reset_parameters(self, init_scale: float):
        if self.n_experts > 1:
            torch.nn.init.normal_(self.sel_v, 0, init_scale / math.sqrt(self.input_size))
            self.renorm_rows(self.sel_v)

        torch.nn.init.normal_(self.sel_o, 0, init_scale / math.sqrt(self.input_size))
        self.renorm_rows(self.sel_o)

        torch.nn.init.normal_(self.k.weight, 0, init_scale / math.sqrt(self.input_size))
        torch.nn.init.normal_(self.q.weight, 0, init_scale / math.sqrt(self.input_size))
        torch.nn.init.normal_(self.v, 0, init_scale / math.sqrt(self.input_size))
        torch.nn.init.normal_(self.o, 0, init_scale / math.sqrt(self.n_heads * self.projection_size))

    def renorm_rows(self, x: torch.Tensor):
        with torch.no_grad():
            std_t = x.std(dim=-1, keepdim=True)
            x.div_(x.norm(dim=-1, keepdim=True))
            x.mul_(std_t / x.std())

    def project_to_torch_order(self, x: torch.Tensor):
        return x.view(*x.shape[:-1], self.n_heads, -1).transpose(-2, -3)

    def get_mask_tensor(self, src_len: int, batch_size: int, mask: Optional[AttentionMask], target_dtype: torch.dtype) -> Optional[torch.Tensor]:

        if mask is None or (mask.position_mask is None and mask.src_length_mask is None):
            return None

        final_pm = None
        final_slm = None

        if mask.position_mask is not None:
            pm = mask.position_mask
            n_pad = src_len - pm.shape[-1]
            if n_pad > 0:
                pm = F.pad(pm, (n_pad, 0), 'constant', value=0)

            if pm.dtype == torch.bool:
                pm = torch.zeros_like(pm, dtype=target_dtype).masked_fill_(pm, float("-inf"))
            else:
                pm = pm.to(target_dtype)

            final_pm = pm.view(batch_size, self.n_heads, pm.shape[-2], pm.shape[-1])

        if mask.src_length_mask is not None:
            slm = mask.src_length_mask
            slm = torch.zeros_like(slm, dtype=target_dtype).masked_fill_(slm, float("-inf"))
            final_slm = slm.unsqueeze(1).unsqueeze(2)

        if final_pm is None:
            return final_slm
        if final_slm is None:
            return final_pm

        return final_pm + final_slm

    def get_sel(self, t: torch.Tensor, w: torch.Tensor) -> Tuple[CVMMSel, torch.Tensor]:
        sel = F.linear(t, w).float()
        sel = sel_raw = sel.view(*sel.shape[:-1], self.n_heads, -1)
        sel = sel.sigmoid()

        with torch.no_grad():
            if self.expert_dropout > 0 and self.training:
                mask = torch.rand_like(sel) < self.expert_dropout
                sel2 = sel.masked_fill(mask, float('-inf'))
            else:
                sel2 = sel
            _, sel_index = sel2.topk(self.moe_k, dim=-1, sorted=False)
        sel_val = torch.gather(sel, -1, sel_index)

        sel_index_shifted = (torch.arange(self.n_heads, device=sel_index.device, dtype=sel_index.dtype) * self.n_experts).unsqueeze(-1) + sel_index
        return cvmm_prepare_sel2(sel_index_shifted.flatten(-2,-1), sel_val), sel_raw

    def attend(self, pos_offset: int, v: torch.Tensor, k: torch.Tensor, q: torch.Tensor,
               mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def forward(self, q_src: torch.Tensor, k_src: torch.Tensor, v_src: torch.Tensor, mask: Optional[AttentionMask],
                kv_cache: KVCache = None) -> Tuple[torch.Tensor, KVCache]:
        # *src: [batch_size, out_len, c]

        pos_offset = q_src.shape[1] - k_src.shape[1]
        # assert pos_offset >= 0

        #if mask is None:
        #    mask = AttentionMask(self.generate_causal_attention_mask(q_src.shape[1]))

        scale = self.scale.sqrt()

        q = self.q(q_src)
        k = self.k(k_src)
        q = q * scale.type_as(q)
        k = k * scale.type_as(k)

        if self.n_experts > 1:
            v_sel, v_sel_r = self.get_sel(k_src, self.sel_v)
            o_sel, o_sel_r = self.get_sel(q_src, self.sel_o)

            v = cvmm(v_src, v_sel, self.v).transpose(-2, -3)
        else:
            o_gate = F.sigmoid(F.linear(q_src, self.sel_o))
            v = self.project_to_torch_order(F.linear(v_src, self.v))

        q = self.project_to_torch_order(q)
        k = self.project_to_torch_order(k)

        if kv_cache is not None:
            v = torch.cat([kv_cache["v"], v], dim=-2) if "v" in kv_cache else v
            k = torch.cat([kv_cache["k"], k], dim=-2) if "k" in kv_cache else k
            kv_cache = {
                "v": v,
                "k": k
            }

        q = self.dropout(q)
        processed_mask = self.get_mask_tensor(v.shape[-2], q_src.shape[0], mask, q.dtype)
        res = self.attend(pos_offset, v, k, q, processed_mask)
        res = res.transpose(-2, -3)

        if self.n_experts > 1:
            # The output selection indices are calculated from the current state and are also used for projecting "q".
            # But that projection needs to create multiple copies for the different heads. Here we already have the
            # heads, but we have to create copies for the top-k elements. We can calculate that from the reduction
            # weight. We also want to compute not only the weighted average between the top-k elements, but also
            # of the different heads. So reshape the reduction weight accordingly.
            o_sel.sel_index = o_sel.out_index // o_sel.reduction_weight.shape[-1]
            o_sel.reduction_weight = o_sel.reduction_weight.flatten(-2)
            out = cvmm(res, o_sel, self.o)
        else:
            res = res * o_gate[..., None]
            out = F.linear(res.flatten(-2), self.o)

        return out, kv_cache

class SwitchHead(SwitchHeadCore):
    def __init__(self, d_model: int, n_heads: int, n_experts: int, dropout: float = 0.0,
                 d_head: Optional[int] = None, expert_dropout: float = 0.0, moe_k: int = 2,
                 init_scale: float = 1.0, rotate_fraction: float = 0.5, rope_base: float = 10000):

        super().__init__(
            d_model, n_heads, n_experts, dropout, d_head, expert_dropout, moe_k)

        super().reset_parameters(init_scale)

    def attend(self, pos_offset: int, v: torch.Tensor, k: torch.Tensor, q: torch.Tensor,
               mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:

        return F.scaled_dot_product_attention(q, k, v, mask, scale=1.0)