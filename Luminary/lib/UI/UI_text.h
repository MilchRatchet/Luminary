#ifndef UI_TEXT_H
#define UI_TEXT_H

#include "UI.h"
#include "SDL_ttf.h"

void init_text(UI* ui);
SDL_Surface* render_text(UI* ui);

#endif /* UI_TEXT_H */
