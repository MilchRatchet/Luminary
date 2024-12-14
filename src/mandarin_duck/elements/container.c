#include "container.h"

void element_container_create(Element** element) {
  MD_CHECK_NULL_ARGUMENT(element);

  LUM_FAILURE_HANDLE(host_malloc(element, sizeof(Element)));

  (*element)->type = ELEMENT_TYPE_CONTAINER;

  LUM_FAILURE_HANDLE(host_malloc(&(*element)->data, sizeof(ElementContainerData)));

  ElementContainerData* data = (ElementContainerData*) (*element)->data;

  LUM_FAILURE_HANDLE(array_create(&data->elements, sizeof(Element*), 16));
}

void element_container_set(Element* element, ElementContainerArgs* args) {
  ElementContainerData* data = (ElementContainerData*) element->data;

  data->target_width  = args->target_width;
  data->target_height = args->target_height;
  data->padding       = args->padding;
  data->is_horizontal = args->is_horizontal;
}

void element_container_add_element(Element* element, Element* added_element) {
  ElementContainerData* data = (ElementContainerData*) element->data;

  LUM_FAILURE_HANDLE(array_push(&data->elements, &added_element));
}
