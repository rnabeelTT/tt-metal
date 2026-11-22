size_t FreeListOpt::find_block_to_shrink(DeviceAddr shrink_size, bool bottom_up) const {
    TT_FATAL(bottom_up, "Shrinking from the top is currently not supported");
    TT_FATAL(
        shrink_size < this->max_size_bytes_,
        "Shrink size {} must be smaller than max size {}",
        shrink_size,
        max_size_bytes_);

    // loop and scan the block list to find if the shrink cut into any allocated block
    std::optional<size_t> found_block;
    const DeviceAddr shrunk_address = shrink_size_ + shrink_size;
    // TODO: There must be a way to force the beginning of all blocks be at index 0
    for (size_t i = 0; i < block_address_.size(); i++) {
        if (!meta_block_is_allocated_[i]) {
            continue;
        }
        if (block_is_allocated_[i]) {
            TT_FATAL(
                block_address_[i] >= shrunk_address,
                "Shrink size {} cuts into allocated block at address {}",
                shrunk_address,
                block_address_[i]);
        } else if (block_address_[i] <= shrunk_address && block_address_[i] + block_size_[i] >= shrunk_address) {
            found_block = i;
        }
    }

    TT_FATAL(
        found_block.has_value(), "Shrink size {} does not align with any block. This must be a bug", shrunk_address);
    return *found_block;
}
