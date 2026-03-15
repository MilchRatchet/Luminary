#include "file_dialog.h"

#include "display.h"

static void _element_file_dialog_render_func(Element* file_dialog, Display* display) {
  MD_CHECK_NULL_ARGUMENT(file_dialog);
  MD_CHECK_NULL_ARGUMENT(display);

  ElementFileDialogData* data = (ElementFileDialogData*) &file_dialog->data;

  LuminaryPath* path;
  file_dialog_handler_try_acquire_path(data->handler, &path);

  uint32_t background_color = (path != (LuminaryPath*) 0) ? MD_COLOR_BLACK : MD_COLOR_ACCENT_2;

  ui_renderer_render_rounded_box(
    display->ui_renderer, display, file_dialog->width, file_dialog->height, file_dialog->x, file_dialog->y, 0, MD_COLOR_BORDER,
    background_color, UI_RENDERER_BACKGROUND_MODE_SEMITRANSPARENT);

  const uint32_t text_color = (data->is_hovered) ? MD_COLOR_ACCENT_LIGHT_2 : MD_COLOR_WHITE;

  if (path != (LuminaryPath*) 0) {
    const char* text;
    LUM_FAILURE_HANDLE(luminary_path_apply(path, (const char*) 0, &text));

    uint32_t text_width;
    text_renderer_render(
      display->text_renderer, display, text, TEXT_RENDERER_FONT_REGULAR, text_color, file_dialog->x + (file_dialog->width >> 1),
      file_dialog->y + (file_dialog->height >> 1), true, true, false, &text_width);

    file_dialog_handler_release_path(data->handler);
  }
}

bool element_file_dialog(Window* window, Display* display, const MouseState* mouse_state, ElementFileDialogArgs args) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(mouse_state);

  WindowContext* context = window->context_stack + window->context_stack_ptr;

  Element file_dialog;

  file_dialog.type        = ELEMENT_TYPE_FILE_DIALOG;
  file_dialog.render_func = _element_file_dialog_render_func;
  file_dialog.hash        = 0;

  ElementMouseResult mouse_result;
  element_apply_context(&file_dialog, context, &args.size, mouse_state, &mouse_result);

  ElementFileDialogData* data = (ElementFileDialogData*) &file_dialog.data;

  data->handler    = args.file_dialog_handler;
  data->is_hovered = mouse_result.is_hovered;

  if (mouse_result.is_hovered) {
    window->element_has_hover = true;

    display_set_cursor(display, SDL_SYSTEM_CURSOR_POINTER);

    window->status.received_hover |= true;
  }

  bool dialog_opened = false;
  if (mouse_result.is_clicked) {
    FileDialogHandlerOpenArgs open_args;
    open_args.sdl_window   = display->sdl_window;
    open_args.dialog_title = args.window_title;

    dialog_opened = file_dialog_handler_open_dialog(args.file_dialog_handler, &open_args);

    window->status.received_mouse_action |= mouse_result.is_clicked;
  }

  window_push_element(window, &file_dialog);

  return dialog_opened;
}
