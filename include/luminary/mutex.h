/*
  Copyright (C) 2021-2026 Max Jenke

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

#ifndef LUMINARY_MUTEX_H
#define LUMINARY_MUTEX_H

#include <luminary/api_utils.h>
#include <luminary/error.h>

struct LuminaryMutex;
typedef struct LuminaryMutex LuminaryMutex;

LuminaryResult mutex_create(LuminaryMutex** mutex);
LuminaryResult mutex_lock(LuminaryMutex* mutex);
LuminaryResult mutex_timed_lock(LuminaryMutex* mutex, const double timeout_time, bool* success);
LuminaryResult mutex_try_lock(LuminaryMutex* mutex, bool* success);
LuminaryResult mutex_unlock(LuminaryMutex* mutex);
LuminaryResult mutex_destroy(LuminaryMutex** mutex);

#endif /* LUMINARY_MUTEX_H */
