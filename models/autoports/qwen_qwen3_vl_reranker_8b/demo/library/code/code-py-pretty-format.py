def pretty_format(captured_graph):
    output = ""
    tabs = ""
    for node in captured_graph:
        if node["node_type"] == "function_end":
            tabs = tabs[:-4]

        length = 50 - len(tabs)
        format_string = f"{{:{length}}}"

        if node["node_type"] == "capture_start":
            node_string = format_string.format("Capture Start")
        elif node["node_type"] == "capture_end":
            node_string = format_string.format("Capture End")
        elif node["node_type"] == "buffer":
            node_string = format_string.format("Add Device Buffer")
        elif node["node_type"] == "buffer_allocate":
            node_string = format_string.format("Allocate Device Buffer")
        elif node["node_type"] == "buffer_deallocate":
            node_string = format_string.format("Deallocate Device Buffer")
        elif node["node_type"] == "function_start":
            node_string = format_string.format("Function Start: " + node["params"]["name"])
        elif node["node_type"] == "function_end":
            node_string = format_string.format("Function End:   " + node["params"]["name"])
        elif node["node_type"] == "tensor":
            node_string = format_string.format("Add Tensor: " + str(node["params"]["tensor_id"]))
        elif node["node_type"] == "circular_buffer_allocate":
            node_string = format_string.format("Allocate Circular Buffer")
        elif node["node_type"] == "dataflow_buffer_allocate":
            node_string = format_string.format("Allocate Dataflow Buffer")
        elif node["node_type"] == "scratchpad_allocate":
            node_string = format_string.format("Allocate Kernel Scratchpad")
        elif node["node_type"] == "circular_buffer_deallocate_all":
            node_string = format_string.format("Deallocate All Program-Scope L1")
        elif node["node_type"] == "sub_device_manager":
            node_string = format_string.format("Sub-Device Manager: " + str(node["params"]["sub_device_manager_id"]))
        elif node["node_type"] == "program_execution":
            sub_device_id = node["params"].get("sub_device_id")
            node_string = format_string.format(
                "Program Execution: sub-device " + (str(sub_device_id) if sub_device_id is not None else "unresolved")
            )
        else:
            raise ValueError(f"Unknown node type: {node['node_type']}")

        output += f"{tabs}{node_string}\n"

        if node["node_type"] == "function_start":
            tabs += "    "

    return output
