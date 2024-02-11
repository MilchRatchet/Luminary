#include "utils.cuh"

__global__ __launch_bounds__(THREADS_PER_BLOCK, 10) void balance_trace_tasks() {
  const int warp = THREAD_ID;

  if (warp >= (THREADS_PER_BLOCK * BLOCKS_PER_GRID) >> 5)
    return;

  __shared__ uint16_t counts[THREADS_PER_BLOCK][32];
  uint32_t sum = 0;

  for (int i = 0; i < 32; i += 4) {
    ushort4 c                  = __ldcs((ushort4*) (device.trace_count + 32 * warp + i));
    counts[threadIdx.x][i + 0] = c.x;
    counts[threadIdx.x][i + 1] = c.y;
    counts[threadIdx.x][i + 2] = c.z;
    counts[threadIdx.x][i + 3] = c.w;
    sum += c.x;
    sum += c.y;
    sum += c.z;
    sum += c.w;
  }

  const uint16_t average = 1 + (sum >> 5);

  for (int i = 0; i < 32; i++) {
    uint16_t count        = counts[threadIdx.x][i];
    int source_index      = -1;
    uint16_t source_count = 0;

    if (count >= average)
      continue;

    for (int j = 0; j < 32; j++) {
      uint16_t c = counts[threadIdx.x][j];
      if (c > average && c > count + 1 && c > source_count) {
        source_count = c;
        source_index = j;
      }
    }

    if (source_index != -1) {
      const int swaps = (source_count - count) >> 1;

      static_assert(THREADS_PER_BLOCK == 128, "The following code assumes that we have 4 warps per block.");
      const int thread_id_base = ((warp & 0b11) << 5);
      const int block_id       = warp >> 2;

      for (int j = 0; j < swaps; j++) {
        TraceTask* source_ptr = device.trace_tasks + get_task_address_of_thread(thread_id_base + source_index, block_id, source_count - 1);
        TraceTask* sink_ptr   = device.trace_tasks + get_task_address_of_thread(thread_id_base + i, block_id, count);

        __stwb((float4*) sink_ptr, __ldca((float4*) source_ptr));
        __stwb((float4*) (sink_ptr) + 1, __ldca((float4*) (source_ptr) + 1));

        sink_ptr++;
        count++;

        source_ptr--;
        source_count--;
      }
      counts[threadIdx.x][i]            = count;
      counts[threadIdx.x][source_index] = source_count;
    }
  }

  for (int i = 0; i < 32; i += 4) {
    ushort4 vals = make_ushort4(counts[threadIdx.x][i], counts[threadIdx.x][i + 1], counts[threadIdx.x][i + 2], counts[threadIdx.x][i + 3]);
    __stcs((ushort4*) (device.trace_count + 32 * warp + i), vals);
  }
}
