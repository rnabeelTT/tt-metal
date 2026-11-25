// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "api/dataflow/dataflow_api.h"
#include "api/remote_circular_buffer.h"

// pages of the remote CB landed and not yet acked >= n (remote_cb_wait_front's test, without blocking)
FORCE_INLINE bool remote_avail(uint32_t cb_id, uint32_t num_pages) {
    RemoteReceiverCBInterface& rc = get_remote_receiver_cb_interface(cb_id);
    uint32_t len_bytes = num_pages * rc.fifo_page_size;
    if (rc.fifo_rd_ptr + len_bytes >= rc.fifo_limit_page_aligned) {
        uint32_t fifo_size = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rc.config_ptr)[3];
        len_bytes += rc.fifo_start_addr + fifo_size - rc.fifo_limit_page_aligned;
    }
    const uint32_t need = len_bytes / REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE;
    volatile tt_l1_ptr uint32_t* acked = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rc.aligned_pages_acked_ptr);
    volatile tt_l1_ptr uint32_t* sent =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rc.aligned_pages_acked_ptr - L1_ALIGNMENT);
    invalidate_l1_cache();
    return *sent - *acked >= need;
}
