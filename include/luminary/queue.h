/*
  Copyright (C) 2021-2024 Max Jenke

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU Affero General Public License as published
  by the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU Affero General Public License for more details.

  You should have received a copy of the GNU Affero General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#ifndef LUMINARY_QUEUE_H
#define LUMINARY_QUEUE_H

#include <luminary/api_utils.h>
#include <luminary/error.h>

struct LuminaryQueue;
typedef struct LuminaryQueue LuminaryQueue;

typedef bool (*LuminaryEqOp)(void* lhs, void* rhs);

#define queue_create(queue, size_of_element, num_elements) \
  _queue_create(queue, size_of_element, num_elements, (const char*) #queue, (const char*) __func__, __LINE__)
#define queue_destroy(queue) _queue_destroy(queue, (const char*) #queue, (const char*) __func__, __LINE__)

LUMINARY_API LuminaryResult
  _queue_create(LuminaryQueue** queue, size_t size_of_element, size_t num_elements, const char* buf_name, const char* func, uint32_t line);
LUMINARY_API LuminaryResult queue_push(LuminaryQueue* queue, void* object);
LUMINARY_API LuminaryResult queue_push_unique(LuminaryQueue* queue, void* object, LuminaryEqOp equal_operator, bool* already_queued);
LUMINARY_API LuminaryResult queue_pop(LuminaryQueue* queue, void* object, bool* success);
LUMINARY_API LuminaryResult queue_pop_blocking(LuminaryQueue* queue, void* object, bool* success);
LUMINARY_API LuminaryResult queue_set_is_blocking(LuminaryQueue* queue, bool is_blocking);
LUMINARY_API LuminaryResult _queue_destroy(LuminaryQueue** queue, const char* buf_name, const char* func, uint32_t line);

#endif /* LUMINARY_QUEUE_H */
