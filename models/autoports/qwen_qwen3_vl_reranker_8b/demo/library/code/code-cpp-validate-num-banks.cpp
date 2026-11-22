void validate_num_banks(uint32_t num_banks, const BufferType& buffer_type, bool disable_interleaved) {
    bool doesnt_support_interleaved = buffer_type == BufferType::L1_SMALL or disable_interleaved;
    bool is_pow2_num_banks = num_banks && (!(num_banks & (num_banks - 1)));
    // Dataflow API does not have a working implementation of generic modulo to determine bank_id for interleaved
    // address gen For non pow2 num banks, special cases need to be added to avoid falling back to generic
    // implementation. See https://github.com/tenstorrent/tt-metal/issues/3321
    std::unordered_set<uint32_t> acceptable_num_non_pow2_mem_banks = {
        7, 12, 20, 28, 48, 56, 63, 70, 72, 80, 94, 108, 110, 117, 120, 124, 126, 130, 140};
    bool custom_mod_bank_id_calculation_exists = acceptable_num_non_pow2_mem_banks.contains(num_banks);
    bool valid_num_banks = (is_pow2_num_banks or custom_mod_bank_id_calculation_exists or doesnt_support_interleaved);
    if (not valid_num_banks) {
        TT_THROW(
            "Invalid number of memory banks {} for {}. Num banks must be power of 2 or have a dedicated modulo "
            "implementation",
            num_banks,
            enchantum::to_string(buffer_type));
    }
}
