#ifndef MANDARIN_DUCK_ARGUMENT_PARSER_H
#define MANDARIN_DUCK_ARGUMENT_PARSER_H

#include "utils.h"

enum ArgumentCategory { ARGUMENT_CATEGORY_DEFAULT = 0 } typedef ArgumentCategory;

struct ArgumentParser typedef ArgumentParser;

typedef void (*ArgumentHandlerFunc)(ArgumentParser* parser, LuminaryHost* host, const uint32_t num_arguments, const char** arguments);

struct ArgumentDescriptor {
  ArgumentCategory category;
  char* long_name;
  char* short_name;
  char* description;
  uint32_t subargument_count;
  ArgumentHandlerFunc handler_func;
} typedef ArgumentDescriptor;

struct ArgumentParser {
  ArgumentDescriptor* descriptors;
  bool dry_run_requested;
};

void argument_parser_create(ArgumentParser** parser);
void argument_parser_parse(ArgumentParser* parser, uint32_t argc, const char** argv, LuminaryHost* host);
void argument_parser_destroy(ArgumentParser** parser);

#endif /* MANDARIN_DUCK_ARGUMENT_PARSER_H */
