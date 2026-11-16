// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// Debug: a native watchdog thread (no GIL) that, when memdump_tick() stops being called for `secs`, copies the head of
// every large host mapping (the command-queue sysmem regions are 1 GiB each) plus /proc/self/maps into `dir`, once.
// Build: cc -O2 -fPIC -shared -o memdump.so memdump.c -lpthread
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static atomic_ulong g_tick;
static double g_secs;
static char g_dir[512];
static uint64_t g_min, g_head;

void memdump_tick(void) { atomic_fetch_add(&g_tick, 1); }

static void dump(void) {
    char path[640], line[1024];
    FILE* m = fopen("/proc/self/maps", "r");
    if (!m) {
        return;
    }
    snprintf(path, sizeof path, "%s/maps.txt", g_dir);
    FILE* mo = fopen(path, "w");
    int n = 0;
    while (fgets(line, sizeof line, m)) {
        if (mo) {
            fputs(line, mo);
        }
        unsigned long a, b;
        char perms[8];
        if (sscanf(line, "%lx-%lx %7s", &a, &b, perms) != 3) {
            continue;
        }
        if ((b - a != g_min && !strstr(line, "huge")) || perms[0] != 'r') {
            continue;
        }
        uint64_t len = b - a < g_head ? b - a : g_head;
        snprintf(path, sizeof path, "%s/region_%02d_%lx_%lx.bin", g_dir, n++, a, b - a);
        FILE* o = fopen(path, "wb");
        if (!o) {
            continue;
        }
        fwrite((void*)a, 1, len, o);
        fclose(o);
        fprintf(stderr, "[memdump] %s (%lu bytes)\n", path, (unsigned long)len);
    }
    fclose(m);
    if (mo) {
        fclose(mo);
    }
}

static void* loop(void* arg) {
    (void)arg;
    unsigned long last = atomic_load(&g_tick);
    double idle = 0;
    for (;;) {
        usleep(500000);
        unsigned long t = atomic_load(&g_tick);
        if (t != last || t == 0) {
            last = t;
            idle = 0;
            continue;
        }
        idle += 0.5;
        if (idle >= g_secs) {
            fprintf(stderr, "[memdump] no progress for %.0f s: dumping host regions to %s\n", idle, g_dir);
            dump();
            fprintf(stderr, "[memdump] done\n");
            return NULL;
        }
    }
}

// secs: stall threshold; dir: output directory; min_bytes: smallest mapping dumped; head_bytes: bytes per mapping
int memdump_start(double secs, const char* dir, uint64_t min_bytes, uint64_t head_bytes) {
    pthread_t th;
    g_secs = secs;
    g_min = min_bytes;
    g_head = head_bytes;
    snprintf(g_dir, sizeof g_dir, "%s", dir);
    if (pthread_create(&th, NULL, loop, NULL)) {
        return -1;
    }
    pthread_detach(th);
    return 0;
}
