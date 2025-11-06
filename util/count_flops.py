import argparse
import sys
import torch

from fvcore.nn import FlopCountAnalysis, flop_count_str

from .flop_handles import (
    baddbmm_flop_jit,
    scaled_dot_product_attention_flop_jit,
    cvmm_flop_counter
)

from model.impl.SwitchHeadParticleTransformer import SwitchHeadParticleTransformer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Count FLOPs for SwitchHeadParticleTransformer using fvcore.")
    parser.add_argument('--particle-switchhead', action=argparse.BooleanOptionalAction, default=False, help='Enable SwitchHead for particle blocks.')
    parser.add_argument('--class-switchhead', action=argparse.BooleanOptionalAction, default=False, help='Enable SwitchHead for class blocks.')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads.')
    parser.add_argument('--num-experts', type=int, default=4, help='Total experts per head in particle blocks.')
    parser.add_argument('--active-experts', type=int, default=2, help='Active experts per token in particle blocks.')
    parser.add_argument('--num-cls-experts', type=int, default=4, help='Total experts per head in class blocks.')
    parser.add_argument('--active-cls-experts', type=int, default=2, help='Active experts per token in class blocks.')
    parser.add_argument('--d-head', type=int, default=None, help='Optional head dimension (overrides default calculation).')

    args = parser.parse_args()

    model_cfg = dict(
        input_dim=17,
        num_classes=10,
        pair_input_dim=4,
        use_pre_activation_pair=False,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=args.num_heads,
        num_experts=args.num_experts,
        active_experts=args.active_experts,
        num_cls_experts=args.num_cls_experts,
        active_cls_experts=args.active_cls_experts,
        particle_switchhead=args.particle_switchhead,
        class_switchhead=args.class_switchhead,
        head_dim=args.d_head,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        activation='gelu',
        trim=True,
        for_inference=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = SwitchHeadParticleTransformer(**model_cfg).to(device)
        model.eval()
        print("--- Model Instantiated Successfully ---")
        print(f"Config: heads={args.num_heads}, particle_switchhead={args.particle_switchhead}, class_switchhead={args.class_switchhead}, E={args.num_experts}, k={args.active_experts}, d_head={args.d_head if args.d_head else 'auto'}")
    except Exception as e:
        print("--- ERROR: Failed to instantiate model ---")
        print(e)
        sys.exit(1)
    
    batch_size = 1
    num_particles = 128
    pf_features = torch.randn(batch_size, 17, num_particles, device=device)
    pf_vectors = torch.randn(batch_size, 4, num_particles, device=device)
    pf_mask = torch.ones(batch_size, 1, num_particles, device=device).bool()
    inputs_tuple = (pf_features, pf_vectors, pf_mask)

    try:
        flop_analyzer = FlopCountAnalysis(model, inputs_tuple)
        flop_analyzer.set_op_handle("aten::baddbmm", baddbmm_flop_jit,
                                    "aten::scaled_dot_product_attention", scaled_dot_product_attention_flop_jit,
                                    "prim::PythonOp.CVMM", cvmm_flop_counter(args.num_heads, args.active_experts,
                                                                             args.active_cls_experts, len(model.blocks)))
    
        
        total_flops = flop_analyzer.total()
        unsupported_ops = flop_analyzer.unsupported_ops()

        print("\n--- FLOP Count Analysis ---")
        print("Total flops: " + str(total_flops))
        print(flop_count_str(flop_analyzer))
        
        if unsupported_ops:
            print("\n--- Unsupported Operations Encountered ---")
            for op, freq in unsupported_ops.items():
                print(f"- {op}: {freq} occurrences")
            print("FLOP count may be inaccurate due to unsupported ops.")

    except Exception as e:
        print("\n--- ERROR during FLOP Count Analysis ---")
        print(e)