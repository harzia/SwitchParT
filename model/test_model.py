import torch
from impl.SwitchHeadParticleTransformer import SwitchHeadParticleTransformer

def run_test_case(test_name, model, dummy_x, dummy_v, dummy_mask=None, dummy_uu=None):
    print(f"\n--- Running Test Case: {test_name} ---")
    try:
        with torch.no_grad():
            output = model(x=dummy_x, v=dummy_v, mask=dummy_mask, uu=dummy_uu)
        
        print(f"Forward pass completed successfully!")
        
        expected_shape = (dummy_x.shape[0], model.fc[-1].out_features)
        print(f"   Output shape: {output.shape}")
        assert output.shape == expected_shape, "Output shape is incorrect!"

    except Exception as e:
        print(f"TEST CASE FAILED: {test_name}")
        raise e

def run_all_tests():
    print("Starting Sanity Check")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    batch_size = 4        
    input_dim = 16       
    num_particles = 100 
    num_classes = 5
    pair_extra_dim = 4

    model = SwitchHeadParticleTransformer(
        input_dim=input_dim,
        num_classes=num_classes,
        pair_input_dim=4,
        pair_extra_dim=pair_extra_dim,
        num_heads=8,
        particle_switchhead=True,
        class_switchhead=True
    ).to(device)
    model.eval()
    print("Model instantiated successfully!")

    dummy_x = torch.randn(batch_size, input_dim, num_particles, device=device)
    dummy_v = torch.randn(batch_size, 4, num_particles, device=device)
    

    # Test Case 1: No Masks
    # Pass only x and v to the model.
    run_test_case("No Masks", model, dummy_x, dummy_v)

    # Test Case 2: Boolean Padding Mask Only
    # Create the boolean mask and pass it to the `mask` argument.
    dummy_padding_mask = torch.ones(batch_size, 1, num_particles, device=device)
    dummy_padding_mask[:, :, -10:] = 0 # Mask out the last 10 particles
    run_test_case("Boolean Padding Mask Only", model, dummy_x, dummy_v, dummy_mask=dummy_padding_mask)

    # Test Case 3: Float Attention Bias Only (from uu)
    # Create a dummy `uu` tensor and pass it.
    dummy_uu = torch.randn(batch_size, pair_extra_dim, num_particles, num_particles, device=device)
    run_test_case("Float Attention Bias Only (from uu)", model, dummy_x, dummy_v, dummy_uu=dummy_uu)

    # Test Case 4: Combined Boolean Padding and Float Bias
    run_test_case("Combined Boolean and Float Mask", model, dummy_x, dummy_v, dummy_mask=dummy_padding_mask, dummy_uu=dummy_uu)

    print("\nAll sanity checks passed!")

if __name__ == '__main__':
    run_all_tests()