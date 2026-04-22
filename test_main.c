#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "aec_v2.h"

#define SAMPLE_RATE  16000
#define NUM_BLOCKS   600
#define PI           3.14159265358979323846

int main(void) {
    AECV2 *aec = AECV2_Create();
    if (!aec) {
        fprintf(stderr, "AECV2_Create failed\n");
        return 1;
    }

    AECV2_Init(aec, SAMPLE_RATE, 1);
    printf("AEC initialized: %d Hz, %d partitions\n", SAMPLE_RATE, aec->num_partitions);

    float far[PART_LEN], near[PART_LEN], out[PART_LEN];
    double echo_power_in = 0.0, echo_power_out = 0.0;
    int i, block;

    /* Warm-up: feed pure echo for NUM_BLOCKS/2 blocks to let filter converge */
    for (block = 0; block < NUM_BLOCKS; block++) {
        /* Far-end: 400 Hz sine */
        for (i = 0; i < PART_LEN; i++) {
            int sample_idx = block * PART_LEN + i;
            far[i] = 0.5f * (float)sin(2.0 * PI * 400.0 * sample_idx / SAMPLE_RATE);
        }
        /* Near-end = echo of far-end (ideal echo scenario) */
        memcpy(near, far, sizeof(far));

        AECV2_Process(aec, far, near, out);

        /* Accumulate power after warm-up phase */
        if (block >= NUM_BLOCKS / 2) {
            for (i = 0; i < PART_LEN; i++) {
                echo_power_in  += near[i] * near[i];
                echo_power_out += out[i]  * out[i];
            }
        }
    }

    double reduction_db = 10.0 * log10((echo_power_out + 1e-12) / (echo_power_in + 1e-12));
    printf("Echo power in:  %.6f\n", echo_power_in);
    printf("Echo power out: %.6f\n", echo_power_out);
    printf("Echo reduction: %.1f dB\n", reduction_db);

    if (reduction_db < -6.0) {
        printf("PASS: AEC is suppressing echo (>6 dB reduction)\n");
    } else {
        printf("WARN: Echo reduction less than expected\n");
    }

    AECV2_Free(aec);
    return 0;
}
