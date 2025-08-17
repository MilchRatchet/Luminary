#ifndef MANDARIN_DUCK_ARGUMENT_PARSER_H
#define MANDARIN_DUCK_ARGUMENT_PARSER_H

#include "utils.h"

enum ArgumentCategory { ARGUMENT_CATEGORY_DEFAULT = 0, ARGUMENT_CATEGORY_LUMINARY = 1 } typedef ArgumentCategory;

struct ArgumentParser typedef ArgumentParser;

typedef void (*ArgumentHandlerFunc)(ArgumentParser* parser, LuminaryHost* host, const uint32_t num_arguments, const char** arguments);

#define MAX_NUM_SUBARGUMENTS 4

struct ArgumentDescriptor {
  ArgumentCategory category;
  const char* long_name;
  const char* short_name;
  const char* description;
  uint32_t subargument_count;
  ArgumentHandlerFunc handler_func;
  bool pre_execute;
} typedef ArgumentDescriptor;

struct ParsedArgument {
  ArgumentCategory category;
  ArgumentHandlerFunc handler_func;
  uint32_t subargument_count;
  char* subarguments[MAX_NUM_SUBARGUMENTS];
} typedef ParsedArgument;

struct ArgumentParserResults {
  bool dry_run_requested;
  const char* output_directory;
  uint32_t num_benchmark_outputs;
  const char* benchmark_name;
  uint32_t device_mask;
} typedef ArgumentParserResults;

struct ArgumentParser {
  ArgumentDescriptor* descriptors;
  char** inputs;
  ParsedArgument* parsed_arguments;
  ArgumentParserResults results;
};

void argument_parser_create(ArgumentParser** parser);
void argument_parser_parse(ArgumentParser* parser, uint32_t argc, const char** argv);
void argument_parser_execute(ArgumentParser* parser, LuminaryHost* host, ArgumentCategory category);
void argument_parser_destroy(ArgumentParser** parser);

#endif /* MANDARIN_DUCK_ARGUMENT_PARSER_H */
