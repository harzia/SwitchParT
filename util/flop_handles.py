import typing
from typing import Any, Callable, List, Optional, Union
from numbers import Number

Handle = Callable[[List[Any], List[Any]], Union[typing.Counter[str], Number]]

def get_shape(val: Any) -> Optional[List[int]]:
    if val.isCompleteTensor():
        return val.type().sizes()
    else:
        return None

def baddbmm_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count FLOPs for torch.baddbmm.
    Performs a batched matrix multiply and an add.
    FLOPs = B * M * N * K
    """
    input_shapes = [get_shape(v) for v in inputs[1:3]]
    # batch1: [B, M, K]
    # batch2: [B, K, N]
    assert len(input_shapes[0]) == 3 and len(input_shapes[1]) == 3, input_shapes
    B, N, M = input_shapes[0]
    P = input_shapes[1][2]
    flops =  B * N * M * P
    return flops

def scaled_dot_product_attention_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count FLOPs for torch.nn.functional.scaled_dot_product_attention,
    excluding softmax and scaling.

    Operation: output = (Q @ K.T) @ V

    Shapes:
    - Q: (B, H, N, Dk)
    - K: (B, H, S, Dk)
    - V: (B, H, S, Dv)

    FLOPs = B * H * N * S * (Dk + Dv)
    """

    q_shape = get_shape(inputs[0])
    k_shape = get_shape(inputs[1])
    v_shape = get_shape(inputs[2])

    if len(q_shape) == 3:
        B, N, Dk = q_shape
        H = 1
        S = k_shape[1]
        Dv = v_shape[2]
    elif len(q_shape) == 4:
        B, H, N, Dk = q_shape
        S = k_shape[2]
        Dv = v_shape[3]
    else:
        raise ValueError(f"Unsupported query shape {q_shape}")

    flops = B * H * N * S * (Dk + Dv)
    return flops

def cvmm_flop_counter(num_heads: int, k_part: int, k_class: int, num_blocks: int) -> Handle:
    call_counter = {"count": 0}

    def cvmm_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
        call_counter["count"] += 1
        call_idx = call_counter["count"]

        x_shape = get_shape(inputs[0])
        B, T = x_shape[0], x_shape[1]
        
        w_shape = get_shape(inputs[2])
        if len(w_shape) == 3:
            _, dim1, dim2 = w_shape
        else:
            raise ValueError(f"Unexpected cvmm weight shape {w_shape}")
        
        if dim1 > dim2:
            d_model = dim1
            d_head = dim2
        else:
            d_head = dim1
            d_model = dim2
        
        if call_idx <= 2 * num_blocks:
            active_k = k_part
        else:
            active_k = k_class
        
        flops = B * num_heads * T * active_k * d_head * (d_model + 1)
        return flops
    
    return cvmm_flop_jit