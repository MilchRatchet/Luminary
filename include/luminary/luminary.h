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

#ifndef LUMINARY_H
#define LUMINARY_H

#include <luminary/api_utils.h>
#include <luminary/error.h>
#include <luminary/host.h>
#include <luminary/name_strings.h>
#include <luminary/path.h>
#include <luminary/structs.h>

// Luminary provides multiple utility functions that can be used
// for a better integration of Luminary. However, they are not following
// Luminary API naming schemes.
#ifdef LUMINARY_INCLUDE_EXTRA_UTILS
#include <luminary/array.h>
#include <luminary/host_memory.h>
#include <luminary/log.h>
#include <luminary/queue.h>
#include <luminary/ringbuffer.h>
#include <luminary/thread_status.h>
#endif /* LUMINARY_INCLUDE_EXTRA_UTILS */

/*
 * Initializes all internal utilities necessary for Luminary to function correctly. This must be called exactly once before any other API
 * functions.
 */
LUMINARY_API void luminary_init(void);

/*
 * Shuts down all internal utilities necessary for Luminary to function correctly. This may only be called after luminary_init.
 */
LUMINARY_API void luminary_shutdown(void);

#endif /* LUMINARY_H */
