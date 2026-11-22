# Performance

| one document, warm, p150 | pair tokens | latency (ms) |
| --- | ---: | ---: |
| text 128 | 129 | 49 |
| text 512 | 509 | 61 |
| text 2048 | 2,019 | 209 |
| text 8192 | 8,069 | 924 |
| image 640px | 291 | 107 |
| image page 1.8K tok | 1,805 | 640 |
| video 8 frames | 615 | 246 |

| profile | request | clients | pairs/s | tokens/s | p50 (ms) | p99 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| p150 | 20 texts (~520 tokens per pair) | 1 | 16.8 | 8,711 | 1,191 | 1,504 |
| p150 | 20 texts (~520 tokens per pair) | 4 | 18.1 | 9,344 | 4,437 | 4,798 |
| p150 | 20 photos (~355 tokens per pair) | 1 | 8.1 | 2,864 | 2,481 | 2,622 |
| p150 | 20 photos (~355 tokens per pair) | 4 | 8.3 | 2,934 | 9,626 | 9,878 |
| p150 | 8 document pages (~1,860 tokens per pair) | 1 | 2.0 | 3,627 | 4,176 | 4,227 |
| p150 | 8 document pages (~1,860 tokens per pair) | 4 | 2.0 | 3,693 | 16,226 | 16,664 |
| p300 | 20 texts (~520 tokens per pair) | 1 | 22.0 | 11,311 | 877 | 1,193 |
| p300 | 20 texts (~520 tokens per pair) | 4 | 30.0 | 15,680 | 2,650 | 3,138 |
| p300 | 20 photos (~355 tokens per pair) | 1 | 13.0 | 4,638 | 1,518 | 1,629 |
| p300 | 20 photos (~355 tokens per pair) | 4 | 16.2 | 5,759 | 4,898 | 5,307 |
| p300 | 8 document pages (~1,860 tokens per pair) | 1 | 3.4 | 6,294 | 2,345 | 2,498 |
| p300 | 8 document pages (~1,860 tokens per pair) | 4 | 4.0 | 7,373 | 8,056 | 8,421 |
| p300x2 | 20 texts (~520 tokens per pair) | 1 | 34.8 | 17,925 | 565 | 731 |
| p300x2 | 20 texts (~520 tokens per pair) | 4 | 62.4 | 32,571 | 1,277 | 1,928 |
| p300x2 | 20 photos (~355 tokens per pair) | 1 | 24.6 | 8,797 | 809 | 986 |
| p300x2 | 20 photos (~355 tokens per pair) | 4 | 30.7 | 10,929 | 2,533 | 3,947 |
| p300x2 | 8 document pages (~1,860 tokens per pair) | 1 | 5.7 | 10,572 | 1,390 | 1,574 |
| p300x2 | 8 document pages (~1,860 tokens per pair) | 4 | 7.1 | 13,127 | 4,412 | 6,438 |

a real server in the packaged container; closed-loop clients sent /v1/rerank requests for 60 s per row, with 0 errors. p150 rows ran on one chip of a p300c board (11x10 grid, the p150 code path). Single-document latency is end to end: HTTP, preprocessing and the device.
