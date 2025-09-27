#ifndef CU_LUMINARY_MIN_HEAP_H
#define CU_LUMINARY_MIN_HEAP_H

#include "math.cuh"
#include "utils.cuh"

struct MinHeap {
  uint8_t max_num_elements;
  uint8_t num_elements;
  MinHeapEntry* data;
} typedef MinHeap;

__device__ MinHeapEntry min_heap_entry_get(uint16_t value, float key) {
  MinHeapEntry entry;
  entry.value = value;
  entry.key   = unsigned_bfloat_pack(key);

  return entry;
}

__device__ MinHeap min_heap_get(MinHeapEntry* data, uint8_t max_num_elements) {
  MinHeap heap;
  heap.max_num_elements = max_num_elements;
  heap.num_elements     = 0;
  heap.data             = data;

  return heap;
}

__device__ void min_heap_swap(MinHeap& heap, uint8_t index0, uint8_t index1) {
  const MinHeapEntry swap = heap.data[index0];
  heap.data[index0]       = heap.data[index1];
  heap.data[index1]       = swap;
}

__device__ void min_heap_bubble_up(MinHeap& heap, MinHeapEntry entry, uint8_t index) {
  while (index > 0) {
    const MinHeapEntry parent = heap.data[index >> 1];

    if (parent.key <= entry.key)
      break;

    heap.data[index] = parent;

    index = index >> 1;
  }

  heap.data[index] = entry;
}

__device__ void min_heap_bubble_down(MinHeap& heap, MinHeapEntry entry) {
  uint8_t index = 0;

  while (index < heap.num_elements) {
    const uint8_t left_index  = (index + 1) * 2;
    const uint8_t right_index = (index + 1) * 2 + 1;

    const MinHeapEntry left  = (left_index < heap.num_elements) ? heap.data[left_index] : min_heap_entry_get(0xFFFF, UBF16_MAX);
    const MinHeapEntry right = (right_index < heap.num_elements) ? heap.data[right_index] : min_heap_entry_get(0xFFFF, UBF16_MAX);

    const MinHeapEntry smallest  = (left.key <= right.key) ? left : right;
    const uint8_t smallest_index = (left.key <= right.key) ? left_index : right_index;

    if (smallest.key >= entry.key)
      break;

    heap.data[index] = smallest;

    index = smallest_index;
  }

  heap.data[index] = entry;
}

__device__ void min_heap_insert(MinHeap& heap, uint16_t value, float key) {
  MinHeapEntry new_entry = min_heap_entry_get(value, key);

  if (heap.num_elements < heap.max_num_elements) {
    min_heap_bubble_up(heap, new_entry, heap.num_elements);

    heap.num_elements++;
    return;
  }

  const MinHeapEntry root = heap.data[0];

  if (root.key >= new_entry.key)
    return;

  min_heap_bubble_down(heap, new_entry);
}

#endif /* CU_LUMINARY_MIN_HEAP_H */
