void FreeListOpt::deallocate(DeviceAddr absolute_address) {
    // The existing FreeList implementation does not check if the address is actually allocated. Just return if it's not
    // Do we want to keep this behavior?

    DeviceAddr addr = absolute_address - offset_bytes_;
    auto block_index_opt = get_and_remove_from_alloc_table(addr);
    if (!block_index_opt.has_value()) {
        return;
    }
    size_t block_index = *block_index_opt;
    block_is_allocated_[block_index] = false;
    size_t prev_block = block_prev_block_[block_index];
    size_t next_block = block_next_block_[block_index];

    // Merge with previous block if it's free
    if (prev_block != no_block && !block_is_allocated_[prev_block]) {
        // Look into the size segregated list to remove the block
        size_t size_segregated_index = get_size_segregated_index(block_size_[prev_block]);
        std::vector<size_t>& segregated_list = free_blocks_segregated_by_size_[size_segregated_index];
        auto it = std::find(segregated_list.begin(), segregated_list.end(), prev_block);
        TT_ASSERT(
            it != segregated_list.end(),
            "Prev block {} not found in size segregated list during deallocation of block {}",
            prev_block,
            block_index);
        segregated_list.erase(it);

        block_size_[prev_block] += block_size_[block_index];
        unlink_block(block_index);
        free_meta_block(block_index);
        block_index = prev_block;
    }

    // Merge with next block if it's free
    if (next_block != no_block && !block_is_allocated_[next_block]) {
        // Look into the size segregated list to remove the block
        size_t size_segregated_index = get_size_segregated_index(block_size_[next_block]);
        std::vector<size_t>& segregated_list = free_blocks_segregated_by_size_[size_segregated_index];
        auto it = std::find(segregated_list.begin(), segregated_list.end(), next_block);
        TT_ASSERT(
            it != segregated_list.end(),
            "Next block {} not found in size segregated list during deallocation of block {}",
            next_block,
            block_index);
        segregated_list.erase(it);

        block_size_[block_index] += block_size_[next_block];
        // block_prev_block_[next_block] == block_index here, so unlinking next_block is what rewires
        // block_index's forward link onto next_block's successor
        unlink_block(next_block);
        free_meta_block(next_block);
    }

    TT_ASSERT(lowest_occupied_address_.has_value(), "Lowest occupied address should have a value");
    if (addr <= *lowest_occupied_address_) {
        lowest_occupied_address_ = std::nullopt;
        size_t curr_block_index = block_next_block_[block_index];
        while (curr_block_index != no_block) {
            if (block_is_allocated_[curr_block_index]) {
                lowest_occupied_address_ = block_address_[curr_block_index];
                break;
            }
            curr_block_index = block_next_block_[curr_block_index];
        }
    }
    // Update the segregated list
    insert_block_to_segregated_list(block_index);
}
