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

#ifndef LUMINARY_HOST_H
#define LUMINARY_HOST_H

#include <luminary/api_utils.h>
#include <luminary/error.h>

struct LuminaryHost;
typedef struct LuminaryHost LuminaryHost;

LUMINARY_API enum LuminaryDeviceSelectorStrategy {
  /* Select the device with highest estimated compute performance. */
  LUMINARY_DEVICE_SELECTOR_STRATEGY_PERFORMANCE = 0,
  /* Select the device with highest amount of memory. */
  LUMINARY_DEVICE_SELECTOR_STRATEGY_MEMORY = 1,
  /* Select the device that matches a specified name. */
  LUMINARY_DEVICE_SELECTOR_STRATEGY_NAME = 2
} typedef LuminaryDeviceSelectorStrategy;

LUMINARY_API struct LuminaryDeviceSelector {
  LuminaryDeviceSelectorStrategy strategy;
  const char* name;
} typedef LuminaryDeviceSelector;

LUMINARY_API LuminaryResult luminary_host_create(LuminaryHost** host);
LUMINARY_API LuminaryResult luminary_host_destroy(LuminaryHost** host);

LUMINARY_API LuminaryResult luminary_host_add_device(LuminaryHost* host, LuminaryDeviceSelector luminary_device_selector);
LUMINARY_API LuminaryResult luminary_host_remove_device(LuminaryHost* host, LuminaryDeviceSelector luminary_device_selector);

LUMINARY_API LuminaryResult luminary_host_load_lum_file(LuminaryHost* host);
LUMINARY_API LuminaryResult luminary_host_load_obj_file(LuminaryHost* host);

LUMINARY_API LuminaryResult luminary_host_start_render(LuminaryHost* host);
LUMINARY_API LuminaryResult luminary_host_skip_render(LuminaryHost* host);
LUMINARY_API LuminaryResult luminary_host_stop_render(LuminaryHost* host);

/*
 * Returns the string identifying the host's current work.
 * @param host Host instance.
 * @param string The destination the address of the string will be written to. If the host is idle, NULL will be written.
 */
LUMINARY_API LuminaryResult luminary_host_get_current_work_string(const LuminaryHost* host, char** string);

LUMINARY_API LuminaryResult luminary_host_set_enable_output(LuminaryHost* host, int enable_output);
LUMINARY_API LuminaryResult luminary_host_get_last_render(LuminaryHost* host);

LUMINARY_API LuminaryResult luminary_host_get_camera(LuminaryHost* host, LuminaryCamera* camera);
LUMINARY_API LuminaryResult luminary_host_set_camera(LuminaryHost* host, LuminaryCamera* camera);

#endif /* LUMINARY_HOST_H */
