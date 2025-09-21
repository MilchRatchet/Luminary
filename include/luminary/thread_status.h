/*
  Copyright (C) 2021-2025 Max Jenke

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

#ifndef LUMINARY_WALL_TIME_H
#define LUMINARY_WALL_TIME_H

#include <luminary/api_utils.h>
#include <luminary/error.h>

struct LuminaryThreadStatus;
typedef struct LuminaryThreadStatus LuminaryThreadStatus;

LUMINARY_API LuminaryResult thread_status_create(LuminaryThreadStatus** thread_status);
LUMINARY_API LuminaryResult thread_status_set_worker_name(LuminaryThreadStatus* thread_status, const char* name);
LUMINARY_API LuminaryResult thread_status_get_worker_name(LuminaryThreadStatus* thread_status, const char** name);
LUMINARY_API LuminaryResult thread_status_start(LuminaryThreadStatus* thread_status, const char* string);
LUMINARY_API LuminaryResult thread_status_get_time(LuminaryThreadStatus* thread_status, double* time);
LUMINARY_API LuminaryResult thread_status_get_string(LuminaryThreadStatus* thread_status, const char** string);
LUMINARY_API LuminaryResult thread_status_stop(LuminaryThreadStatus* thread_status);
LUMINARY_API LuminaryResult thread_status_destroy(LuminaryThreadStatus** thread_status);

#endif /* LUMINARY_WALL_TIME_H */
