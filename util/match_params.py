import argparse
import torch
import copy

from model.impl.SwitchHeadParticleTransformer import SwitchHeadParticleTransformer

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def match_parameters(baseline_model, switchhead_base_cfg, start_d_head=1, step=1):
    target_params = count_parameters(baseline_model)
    print(f"--- Target Parameters (from baseline): {target_params:,} ---")
    print(f"--- Searching for optimal d_head for SwitchHead config... ---")

    d_head = start_d_head
    prev_params = 0
    prev_cfg = {}

    while True:
        current_cfg = switchhead_base_cfg.copy()
        current_cfg['head_dim'] = d_head
        
        try:
            model_to_test = SwitchHeadParticleTransformer(**current_cfg)
        except TypeError as e:
            print("\n\nERROR: Could not create SwitchHeadParticleTransformer.")
            print("Please ensure your class __init__ method accepts a 'd_head' argument.")
            print(f"Original Error: {e}")
            return None

        current_params = count_parameters(model_to_test)
        
        print(f"Trying d_head = {d_head} -> Parameters = {current_params:,}")

        if current_params >= target_params:
            diff_prev = abs(target_params - prev_params)
            diff_curr = abs(target_params - current_params)
            
            final_cfg = prev_cfg if diff_prev < diff_curr and prev_cfg else current_cfg
            final_params = count_parameters(SwitchHeadParticleTransformer(**final_cfg))
            
            print("\n--- Match Found! ---")
            print(f"Optimal d_head: {final_cfg['head_dim']}")
            print(f"Final Parameters: {final_params:,} (Difference: {abs(final_params-target_params):,})")
            return final_cfg
        
        prev_params = current_params
        prev_cfg = current_cfg
        d_head += step
        
        if d_head > 128: 
            print("ERROR: Search for d_head exceeded 128. Check your configuration.")
            return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Find the optimal d_head to parameter-match a SwitchHead model to a baseline.")
    parser.add_argument('--num-heads', type=int, required=True, help='Number of attention heads for the SwitchHead model.')
    parser.add_argument('--num-experts', type=int, required=True, help='Total number of experts per head for particle blocks (E).')
    parser.add_argument('--active-experts', type=int, required=True, help='Number of active experts per token for particle blocks (k).')
    parser.add_argument('--num-cls-experts', type=int, required=True, help='Total number of experts per head for class blocks (E).')
    parser.add_argument('--active-cls-experts', type=int, required=True, help='Number of active experts per token for class blocks (k).')
    
    args = parser.parse_args()

    baseline_cfg = dict(
        input_dim=17,
        num_classes=10,
        pair_input_dim=4,
        use_pre_activation_pair=False,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        particle_switchhead=False,
        class_switchhead=False,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        activation='gelu',
        # misc
        trim=True,
        for_inference=True,
    )
    baseline_model = SwitchHeadParticleTransformer(**baseline_cfg)
    switchhead_model_cfg = copy.deepcopy(baseline_cfg)

    
    switchhead_overrides = dict(
        particle_switchhead=True,
        class_switchhead=True,
        num_heads=args.num_heads,
        num_experts=args.num_experts,
        active_experts=args.active_experts,
        num_cls_experts=args.num_cls_experts,
        active_cls_experts=args.active_cls_experts,
    )
    switchhead_model_cfg.update(switchhead_overrides)

    matched_config = match_parameters(baseline_model, switchhead_model_cfg)

    if matched_config:
        print("\n--- Final Parameter-Matched Configuration ---")
        for key, value in matched_config.items():
            print(f"  {key}: {value}")