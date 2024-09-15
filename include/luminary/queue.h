/*
  Copyright (c) 2021-2024, MilchRatchet

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/

#ifndef LUMINARY_QUEUE_H
#define LUMINARY_QUEUE_H

#include <luminary/api_utils.h>
#include <luminary/error.h>

struct LuminaryQueue;
typedef struct LuminaryQueue LuminaryQueue;

#define queue_create(queue, size_of_element, num_elements) \
  _queue_create(queue, size_of_element, num_elements, (const char*) #queue, (const char*) __func__, __LINE__)
#define queue_destroy(queue) _queue_destroy(queue, (const char*) #queue, (const char*) __func__, __LINE__)

LUMINARY_API LuminaryResult
  _queue_create(LuminaryQueue** queue, size_t size_of_element, size_t num_elements, const char* buf_name, const char* func, uint32_t line);
LUMINARY_API LuminaryResult queue_push(LuminaryQueue* queue, void* object);
LUMINARY_API LuminaryResult queue_pop(LuminaryQueue* queue, void* object, bool* success);
LUMINARY_API LuminaryResult queue_pop_blocking(LuminaryQueue* queue, void* object, bool* success);
LUMINARY_API LuminaryResult queue_flush_blocking(LuminaryQueue* queue);
LUMINARY_API LuminaryResult _queue_destroy(LuminaryQueue** queue, const char* buf_name, const char* func, uint32_t line);

#endif /* LUMINARY_QUEUE_H */
