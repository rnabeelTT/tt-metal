def _capture_python_stack_trace() -> list[str]:
    """Capture the current Python call stack, filtering out ttnn internals.

    Returns a list of formatted frame strings (file:line in function) with
    infrastructure frames from decorators.py and graph.py removed.
    """
    frames = traceback.extract_stack()
    result = []

    for frame in frames:
        try:
            filename = str(frame.filename)
            path = pathlib.PurePath(filename)

            # Match without filesystem resolution to avoid fragile behavior during capture
            if any(path.match(p) for p in _STACK_TRACE_INTERNAL_PATTERNS):
                continue

            result.append(f'  File "{filename}", line {frame.lineno}, in {frame.name}\n    {frame.line}\n')
        except Exception:
            # Never let stack-trace capture break graph capture
            continue

    return result[::-1]
