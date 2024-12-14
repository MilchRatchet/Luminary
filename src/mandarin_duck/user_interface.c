#include "user_interface.h"

////////////////////////////////////////////////////////////////////
// Internal functions
////////////////////////////////////////////////////////////////////

static void _user_interface_setup(UserInterface* ui) {
  MD_CHECK_NULL_ARGUMENT(ui);
}

////////////////////////////////////////////////////////////////////
// API functions
////////////////////////////////////////////////////////////////////

void user_interface_create(UserInterface** ui) {
  MD_CHECK_NULL_ARGUMENT(ui);

  LUM_FAILURE_HANDLE(host_malloc(ui, sizeof(UserInterface)));

  LUM_FAILURE_HANDLE(array_create(&(*ui)->windows, sizeof(Window), 16));

  _user_interface_setup(*ui);
}

void user_interface_destroy(UserInterface** ui) {
  MD_CHECK_NULL_ARGUMENT(ui);
  MD_CHECK_NULL_ARGUMENT(*ui);

  LUM_FAILURE_HANDLE(array_destroy(&(*ui)->windows));

  LUM_FAILURE_HANDLE(host_free(ui));
}
