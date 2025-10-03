#ifndef CU_LUMINARY_MIN_HEAP_H
#define CU_LUMINARY_MIN_HEAP_H

#include "math.cuh"
#include "utils.cuh"

struct MinHeap {
  uint8_t max_num_elements;
  uint8_t num_elements;
  float sum_key;
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
  heap.sum_key          = 0.0f;
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
    heap.sum_key += key;

    min_heap_bubble_up(heap, new_entry, heap.num_elements);

    heap.num_elements++;
    return;
  }

  const MinHeapEntry root = heap.data[0];

  if (root.key >= new_entry.key)
    return;

  heap.sum_key += key;
  heap.sum_key -= unsigned_bfloat_unpack(root.key);

  min_heap_bubble_down(heap, new_entry);
}

__device__ bool min_heap_remove_min(MinHeap& heap, const float sum_key_threshold) {
  if (heap.num_elements == 0)
    return false;

  const float min_key = unsigned_bfloat_unpack(heap.data[0].key);

  if (heap.sum_key - min_key < sum_key_threshold)
    return false;

  heap.num_elements--;
  heap.sum_key -= min_key;

  if (heap.num_elements == 0)
    return true;

  MinHeapEntry last_entry = heap.data[heap.num_elements];

  min_heap_bubble_down(heap, last_entry);

  return true;
}

#endif /* CU_LUMINARY_MIN_HEAP_H */
