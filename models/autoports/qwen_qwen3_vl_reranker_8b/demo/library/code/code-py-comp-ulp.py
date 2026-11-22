def comp_ulp(golden, calculated, ulp_threshold, allow_nonfinite=False):
    """
    Compute absolute error between two tensors in Units of Least Precision (ULP)
    """

    import torch

    # If both tensors are empty, then we can return True
    if torch.numel(golden) == 0 and torch.numel(calculated) == 0:
        return True, "Both tensors are empty"

    if not allow_nonfinite and not torch.all(torch.isfinite(calculated)):
        return False, "Calculated tensor contains non-finite values"

    if not _comp_nonfinite(golden, calculated):
        return False, "Tensors are not finite at the same positions"
    # nonfinite elements can interfere with ULP error calculation
    # To avoid this, replace nan, +inf, -inf with 0
    # (we have already checked that both tensors have the same nonfinite elements)
    mask_finite = ~torch.isfinite(golden)
    golden = golden.clone()
    calculated = calculated.clone()
    golden[mask_finite] = 0
    calculated[mask_finite] = 0

    # ULP is measured according to the golden tensor
    # In most cases, data type of golden tensor should be the same as calculated tensor.
    # However, in some cases, we may want to measure < 1 ULP differences, which requires golden tensor
    # to have higher precision than calculated tensor.
    # If we passed golden tensor to ulp() as is, we would get ULP of higher precision.
    # e.g. ulp of float32 rather bfloat16 calculation, which would give us a wrong value.
    ulp_value = ulp(golden.type(calculated.dtype))

    if golden.dtype != calculated.dtype:  # Note: assumes that golden has higher precision than calculated tensor
        calculated = calculated.type(golden.dtype)
        ulp_value = ulp_value.type(golden.dtype)  # Convert ULP to higher precision (for sub-1 ULP measurements)

    ulp_tensor = torch.abs(calculated - golden) / ulp_value
    ulp_delta = torch.max(ulp_tensor)
    within_threshold = ulp_delta <= ulp_threshold
    message = f"Max ULP Delta: {ulp_delta}"
    if not within_threshold:
        ulp_index = torch.argmax(ulp_tensor)
        ulp_index_tuple = tuple(int(idx) for idx in torch.unravel_index(ulp_index, golden.shape))
        message += (
            f" @ {list(ulp_index_tuple)} = "
            f"|calculated {calculated[ulp_index_tuple]} - golden {golden[ulp_index_tuple]}| "
            f"/ ULP(golden) {ulp_value[ulp_index_tuple]}"
        )
    return (within_threshold, message)
