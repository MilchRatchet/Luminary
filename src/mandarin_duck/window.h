#ifndef MANDARIN_DUCK_WINDOW_H
#define MANDARIN_DUCK_WINDOW_H

#include "element.h"
#include "utils.h"

enum WindowType {
  WINDOW_TYPE_CAPTION_CONTROLS = 0,
  WINDOW_TYPE_RENDERER_STATUS  = 1,
  WINDOW_TYPE_ABOUT            = 2,
  WINDOW_TYPE_COUNT
} typedef WindowType;

/*
 * The Window is passed to each element so an element can resize itself based on window orientation (vertical/horizontal)
 * and its dimensions.
 */
struct Window {
  uint32_t width;
  uint32_t height;
  uint32_t padding;
  bool is_horizontal;
  bool is_visible;
  bool background;
  Element* element_container;
} typedef Window;

struct WindowRenderContext {
  Window* window;
  Element* container;
  uint32_t fill;  // Elements add to the fill based on their width/height, the window will give a warning if the fill exceeds limits,
                  // elements must make sure to not render past the fill limit but they must update the fill as if they were not limited so
                  // an accurate final fill is retrieved
} typedef WindowRenderContext;

void window_create(Window** window);
void window_destroy(Window** window);

#endif /* MANDARIN_DUCK_WINDOW_H */
