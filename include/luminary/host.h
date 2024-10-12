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

#ifndef LUMINARY_HOST_H
#define LUMINARY_HOST_H

#include <luminary/api_utils.h>
#include <luminary/error.h>
#include <luminary/path.h>
#include <luminary/structs.h>

struct LuminaryHost;
typedef struct LuminaryHost LuminaryHost;

LUMINARY_API LuminaryResult luminary_host_create(LuminaryHost** host);
LUMINARY_API LuminaryResult luminary_host_destroy(LuminaryHost** host);

LUMINARY_API LuminaryResult luminary_host_get_device_list(LuminaryHost* host);
LUMINARY_API LuminaryResult luminary_host_start_device(LuminaryHost* host, uint32_t index);
LUMINARY_API LuminaryResult luminary_host_shutdown_device(LuminaryHost* host, uint32_t index);

LUMINARY_API LuminaryResult luminary_host_load_lum_file(LuminaryHost* host, LuminaryPath* path);
LUMINARY_API LuminaryResult luminary_host_load_obj_file(LuminaryHost* host, LuminaryPath* path);

LUMINARY_API LuminaryResult luminary_host_start_render(LuminaryHost* host);
LUMINARY_API LuminaryResult luminary_host_skip_render(LuminaryHost* host);
LUMINARY_API LuminaryResult luminary_host_stop_render(LuminaryHost* host);

/*
 * Returns the string identifying the host's current queue task.
 * @param host Host instance.
 * @param string The destination the address of the string will be written to. If the host is idle, NULL will be written.
 */
LUMINARY_API LuminaryResult luminary_host_get_queue_string(const LuminaryHost* host, const char** string);

/*
 * Returns the wall time that the host's current queue task has thus far taken up.
 * @param host Host instance.
 * @param time The destination the time will be written to. The time is given in seconds. If the host is idle, 0.0 will be returned.
 */
LUMINARY_API LuminaryResult luminary_host_get_queue_time(const LuminaryHost* host, double* time);

LUMINARY_API LuminaryResult luminary_host_set_enable_output(LuminaryHost* host, int enable_output);
LUMINARY_API LuminaryResult luminary_host_get_last_render(LuminaryHost* host);

LUMINARY_API LuminaryResult luminary_host_get_max_sample_count(LuminaryHost* host, uint32_t* max_sample_count);
LUMINARY_API LuminaryResult luminary_host_set_max_sample_count(LuminaryHost* host, uint32_t* max_sample_count);

LUMINARY_API LuminaryResult luminary_host_get_settings(LuminaryHost* host, LuminaryRendererSettings* settings);
LUMINARY_API LuminaryResult luminary_host_set_settings(LuminaryHost* host, LuminaryRendererSettings* settings);

LUMINARY_API LuminaryResult luminary_host_get_camera(LuminaryHost* host, LuminaryCamera* camera);
LUMINARY_API LuminaryResult luminary_host_set_camera(LuminaryHost* host, LuminaryCamera* camera);

LUMINARY_API LuminaryResult luminary_host_get_ocean(LuminaryHost* host, LuminaryOcean* ocean);
LUMINARY_API LuminaryResult luminary_host_set_ocean(LuminaryHost* host, LuminaryOcean* ocean);

LUMINARY_API LuminaryResult luminary_host_get_sky(LuminaryHost* host, LuminarySky* sky);
LUMINARY_API LuminaryResult luminary_host_set_sky(LuminaryHost* host, LuminarySky* sky);

LUMINARY_API LuminaryResult luminary_host_get_cloud(LuminaryHost* host, LuminaryCloud* cloud);
LUMINARY_API LuminaryResult luminary_host_set_cloud(LuminaryHost* host, LuminaryCloud* cloud);

LUMINARY_API LuminaryResult luminary_host_get_fog(LuminaryHost* host, LuminaryFog* fog);
LUMINARY_API LuminaryResult luminary_host_set_fog(LuminaryHost* host, LuminaryFog* fog);

LUMINARY_API LuminaryResult luminary_host_get_particles(LuminaryHost* host, LuminaryParticles* particles);
LUMINARY_API LuminaryResult luminary_host_set_particles(LuminaryHost* host, LuminaryParticles* particles);

LUMINARY_API LuminaryResult luminary_host_get_toy(LuminaryHost* host, LuminaryToy* toy);
LUMINARY_API LuminaryResult luminary_host_set_toy(LuminaryHost* host, LuminaryToy* toy);

#endif /* LUMINARY_HOST_H */