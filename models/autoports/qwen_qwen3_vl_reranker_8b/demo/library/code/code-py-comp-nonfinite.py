def _comp_nonfinite(golden, calculated):
    """
    Returns True if tensors contain the same non-finite values (nan, inf, -inf) at the same positions. Also returns True if all elements are finite.
    Returns False if non-finite values differ between both tensors.
    """

    import torch

    # torch.equal(['nan'], ['nan']] => False
    # For this reason, we check for nan and inf separately
    if torch.not_equal(torch.isnan(golden), torch.isnan(calculated)).any():
        return False

    golden_inf_mask = torch.isinf(golden)
    calculated_inf_mask = torch.isinf(calculated)

    if torch.not_equal(golden_inf_mask, calculated_inf_mask).any():
        return False

    golden_inf = golden[golden_inf_mask]
    calculated_inf = calculated[calculated_inf_mask]
    return torch.equal(golden_inf, calculated_inf)
