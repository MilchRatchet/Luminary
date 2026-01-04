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

#ifndef LUMINARY_HOST_H
#define LUMINARY_HOST_H

#include <luminary/api_utils.h>
#include <luminary/error.h>
#include <luminary/path.h>
#include <luminary/structs.h>

struct LuminaryHost;
typedef struct LuminaryHost LuminaryHost;

LUMINARY_API LuminaryResult luminary_host_create(LuminaryHost** host, LuminaryHostCreateInfo info);
LUMINARY_API LuminaryResult luminary_host_destroy(LuminaryHost** host);

LUMINARY_API LuminaryResult luminary_host_start_new_render(LuminaryHost* host);

LUMINARY_API LuminaryResult luminary_host_get_device_count(LuminaryHost* host, uint32_t* device_count);
LUMINARY_API LuminaryResult luminary_host_get_device_info(LuminaryHost* host, uint32_t device_id, LuminaryDeviceInfo* info);
LUMINARY_API LuminaryResult luminary_host_set_device_enable(LuminaryHost* host, uint32_t device_id, bool enable);

LUMINARY_API LuminaryResult luminary_host_start_device(LuminaryHost* host, uint32_t index);
LUMINARY_API LuminaryResult luminary_host_shutdown_device(LuminaryHost* host, uint32_t index);

LUMINARY_API LuminaryResult luminary_host_load_lum_file(LuminaryHost* host, LuminaryPath* path);
LUMINARY_API LuminaryResult luminary_host_load_obj_file(LuminaryHost* host, LuminaryPath* path);

LUMINARY_API LuminaryResult luminary_host_get_current_sample_time(LuminaryHost* host, double* time);

LUMINARY_API LuminaryResult luminary_host_get_num_queue_workers(const LuminaryHost* host, uint32_t* num_queue_workers);

/*
 * Returns the name of the selected queue worker.
 * @param host Host instance.
 * @param queue_worker_id ID of queue worker
 * @param string The destination the address of the string will be written to. If the queue worker is not online, NULL will be written.
 */
LUMINARY_API LuminaryResult luminary_host_get_queue_worker_name(const LuminaryHost* host, uint32_t queue_worker_id, const char** string);

/*
 * Returns the string identifying the selected queue worker's current queue task.
 * @param host Host instance.
 * @param queue_worker_id ID of queue worker
 * @param string The destination the address of the string will be written to. If the queue worker is idle, NULL will be written.
 */
LUMINARY_API LuminaryResult luminary_host_get_queue_worker_string(const LuminaryHost* host, uint32_t queue_worker_id, const char** string);

/*
 * Returns the wall time that the selected queue worker's current queue task has thus far taken up.
 * @param host Host instance.
 * @param queue_worker_id ID of queue worker
 * @param time The destination the time will be written to. The time is given in seconds. If the queue worker is idle, 0.0 will be returned.
 */
LUMINARY_API LuminaryResult luminary_host_get_queue_worker_time(const LuminaryHost* host, uint32_t queue_worker_id, double* time);

LUMINARY_API LuminaryResult luminary_host_set_output_properties(LuminaryHost* host, LuminaryOutputProperties properties);

LUMINARY_API LuminaryResult
  luminary_host_request_output(LuminaryHost* host, LuminaryOutputRequestProperties properties, LuminaryOutputPromiseHandle* handle);
LUMINARY_API LuminaryResult
  luminary_host_try_await_output(LuminaryHost* host, LuminaryOutputPromiseHandle handle, LuminaryOutputHandle* output_handle);

/*
 * Returns handle to an output. Every handle acquired must be released by calling luminary_host_release_output. Consecutive calls to
 * luminary_host_acquire_output are not guaranteed to return unique handles.
 * @param host Host instance.
 * @param output_handle The destination the handle will be written to.
 */
LUMINARY_API LuminaryResult luminary_host_acquire_output(LuminaryHost* host, LuminaryOutputHandle* output_handle);
LUMINARY_API LuminaryResult luminary_host_get_image(LuminaryHost* host, LuminaryOutputHandle output_handle, LuminaryImage* image);
LUMINARY_API LuminaryResult luminary_host_release_output(LuminaryHost* host, LuminaryOutputHandle output_handle);

LUMINARY_API LuminaryResult luminary_host_get_pixel_info(LuminaryHost* host, uint16_t x, uint16_t y, LuminaryPixelQueryResult* result);

LUMINARY_API LuminaryResult luminary_host_get_settings(LuminaryHost* host, LuminaryRendererSettings* settings);
LUMINARY_API LuminaryResult luminary_host_set_settings(LuminaryHost* host, const LuminaryRendererSettings* settings);

LUMINARY_API LuminaryResult luminary_host_get_camera(LuminaryHost* host, LuminaryCamera* camera);
LUMINARY_API LuminaryResult luminary_host_set_camera(LuminaryHost* host, const LuminaryCamera* camera);

LUMINARY_API LuminaryResult luminary_host_get_ocean(LuminaryHost* host, LuminaryOcean* ocean);
LUMINARY_API LuminaryResult luminary_host_set_ocean(LuminaryHost* host, const LuminaryOcean* ocean);

LUMINARY_API LuminaryResult luminary_host_get_sky(LuminaryHost* host, LuminarySky* sky);
LUMINARY_API LuminaryResult luminary_host_set_sky(LuminaryHost* host, const LuminarySky* sky);

LUMINARY_API LuminaryResult luminary_host_get_cloud(LuminaryHost* host, LuminaryCloud* cloud);
LUMINARY_API LuminaryResult luminary_host_set_cloud(LuminaryHost* host, const LuminaryCloud* cloud);

LUMINARY_API LuminaryResult luminary_host_get_fog(LuminaryHost* host, LuminaryFog* fog);
LUMINARY_API LuminaryResult luminary_host_set_fog(LuminaryHost* host, const LuminaryFog* fog);

LUMINARY_API LuminaryResult luminary_host_get_particles(LuminaryHost* host, LuminaryParticles* particles);
LUMINARY_API LuminaryResult luminary_host_set_particles(LuminaryHost* host, const LuminaryParticles* particles);

LUMINARY_API LuminaryResult luminary_host_get_material(LuminaryHost* host, uint16_t id, LuminaryMaterial* material);
LUMINARY_API LuminaryResult luminary_host_set_material(LuminaryHost* host, uint16_t id, const LuminaryMaterial* material);

LUMINARY_API LuminaryResult luminary_host_get_instance(LuminaryHost* host, uint32_t id, LuminaryInstance* instance);
LUMINARY_API LuminaryResult luminary_host_set_instance(LuminaryHost* host, uint32_t id, const LuminaryInstance* instance);
LUMINARY_API LuminaryResult luminary_host_new_instance(LuminaryHost* host, uint32_t* id);

LUMINARY_API LuminaryResult luminary_host_get_num_meshes(LuminaryHost* host, uint32_t* num_meshes);
LUMINARY_API LuminaryResult luminary_host_get_num_materials(LuminaryHost* host, uint32_t* num_materials);
LUMINARY_API LuminaryResult luminary_host_get_num_instances(LuminaryHost* host, uint32_t* num_instances);

LUMINARY_API LuminaryResult luminary_host_save_png(LuminaryHost* host, LuminaryOutputHandle handle, LuminaryPath* path);

/*
 * Calling this function will cause a rebuild of the sky HDRI and a restart of integration.
 * @param host Host instance.
 */
LUMINARY_API LuminaryResult luminary_host_request_sky_hdri_build(LuminaryHost* host);

#endif /* LUMINARY_HOST_H */
