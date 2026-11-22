std::vector<std::pair<DeviceAddr, DeviceAddr>> BankManager::merge_allocated_ranges_from(
    const ttsl::SmallVector<AllocatorDependencies::AllocatorID>& dependent_allocators) {
    // Collect allocated address ranges per dependent allocator (single pass)
    std::vector<std::pair<DeviceAddr, DeviceAddr>> all_allocated_ranges;
    for (const auto dep_allocator_id : dependent_allocators) {
        auto* dep_alloc = this->get_allocator_from_id(dep_allocator_id);
        TT_FATAL(dep_alloc, "Allocator not initialized!");
        const auto allocated_addresses = dep_alloc->allocated_addresses();
        all_allocated_ranges.reserve(all_allocated_ranges.size() + allocated_addresses.size());
        for (const auto& [addr, end_addr] : allocated_addresses) {
            if (end_addr > addr) {
                all_allocated_ranges.emplace_back(addr, end_addr);
            }
        }
    }

    // Sort allocated ranges across all dependent allocators by start address
    std::sort(all_allocated_ranges.begin(), all_allocated_ranges.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    // Coalesce overlaps across all dependent allocators
    std::vector<std::pair<DeviceAddr, DeviceAddr>> coalesced_ranges;
    coalesced_ranges.reserve(all_allocated_ranges.size());
    for (const auto& r : all_allocated_ranges) {
        if (coalesced_ranges.empty() || r.first > coalesced_ranges.back().second) {
            coalesced_ranges.push_back(r);
        } else {
            coalesced_ranges.back().second = std::max(coalesced_ranges.back().second, r.second);
        }
    }

    return coalesced_ranges;
}
