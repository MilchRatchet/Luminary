#ifndef LUMINARY_H
#define LUMINARY_H

#include <luminary/api_utils.h>
#include <luminary/error.h>
#include <luminary/host.h>
#include <luminary/structs.h>

// Luminary provides multiple utility functions that can be used
// for a better integration of Luminary. However, they are not following
// Luminary API naming schemes.
#ifndef LUMINARY_NO_EXTRA_UTILS
#include <luminary/array.h>
#include <luminary/host_memory.h>
#include <luminary/log.h>
#include <luminary/queue.h>
#endif /* LUMINARY_NO_EXTRA_UTILS */

#endif /* LUMINARY_H */
