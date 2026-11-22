def ulp(x: _ttnn.Tensor | _torch.Tensor) -> _ttnn.Tensor | _torch.Tensor:
    "Return Unit of Least Precision for each element of a given tensor"

    import torch

    import ttnn

    received_ttnn_input = False
    if isinstance(x, ttnn.Tensor):
        x = ttnn.to_torch(x)
        received_ttnn_input = True

    # Notes:
    # - This should be identical to the definition of ULP by Goldberg
    #   "What every computer scientist should know about floating-point arithmetic"
    #   https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
    # - We use torch.abs(x) to ensure symmetry ULP(-x) == ULP(x)
    # - For x powers of 2, x + ULP(x) is not closest number but second closest (previous number is 2x closer)
    #   However, this avoids rounding-to-nearest-tie-to-even issues on addition (i.e. x + ULP(x) != x)
    abs_x = torch.abs(x)
    next = torch.nextafter(
        abs_x, torch.tensor(math.inf, dtype=x.dtype)
    )  # 1 ULP ~ Difference between two consecutive floating point numbers
    ulp_value = next - abs_x

    # Special case: if abs_x == torch.finfo(x.dtype).max, then next == math.inf, which leads to ULP(x) == inf rather than finite number
    # We fix this problem by manually calculating ULP at max value, and masking tensor when input == max
    dtype_max = torch.finfo(x.dtype).max
    max_epsilon = dtype_max - torch.nextafter(
        torch.tensor(dtype_max, dtype=x.dtype), torch.tensor(-math.inf, dtype=x.dtype)
    )
    ulp_value = torch.where(abs_x == dtype_max, max_epsilon, ulp_value)

    if received_ttnn_input:  # Ensures that type(input) == type(output)
        ulp_value = ttnn.from_torch(ulp_value)

    return ulp_value
