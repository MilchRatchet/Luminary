#include "argument_parser.h"

#include <stdio.h>
#include <string.h>

#include "config.h"

////////////////////////////////////////////////////////////////////
// Argument handler functions
////////////////////////////////////////////////////////////////////

#define UNUSED_ARG(X) (void) X

static void _argument_parser_arg_func_benchmark(
  ArgumentParser* parser, LuminaryHost* host, uint32_t num_arguments, const char** arguments) {
  UNUSED_ARG(parser);
  UNUSED_ARG(host);
  UNUSED_ARG(num_arguments);
  UNUSED_ARG(arguments);

  if (num_arguments > 0) {
    parser->results.num_benchmark_outputs = atoll(arguments[0]);
  }
}

static void _argument_parser_arg_func_help(ArgumentParser* parser, LuminaryHost* host, uint32_t num_arguments, const char** arguments) {
  UNUSED_ARG(host);
  UNUSED_ARG(num_arguments);
  UNUSED_ARG(arguments);

  uint32_t num_available_arguments;
  LUM_FAILURE_HANDLE(array_get_num_elements(parser->descriptors, &num_available_arguments));

  printf("OVERVIEW: Mandarin Duck Frontend for Luminary\n\n");

#ifdef _WIN32
  printf("USAGE: MandarinDuck.exe [options] file...\n\n");
#else
  printf("USAGE: MandarinDuck [options] file...\n\n");
#endif

  printf("OPTIONS:\n");

  for (uint32_t argument_id = 0; argument_id < num_available_arguments; argument_id++) {
    const ArgumentDescriptor* argument = parser->descriptors + argument_id;

    printf("\t--%s", argument->long_name);
    if (argument->short_name) {
      printf(", -%s", argument->short_name);
    }

    printf("\n\t\t%s\n", argument->description);
  }

  parser->results.dry_run_requested = true;
}

static void _argument_parser_arg_func_output(ArgumentParser* parser, LuminaryHost* host, uint32_t num_arguments, const char** arguments) {
  UNUSED_ARG(parser);
  UNUSED_ARG(host);
  UNUSED_ARG(num_arguments);
  UNUSED_ARG(arguments);

  if (num_arguments > 0) {
    parser->results.output_directory = arguments[0];
  }
}

static void _argument_parser_arg_func_version(ArgumentParser* parser, LuminaryHost* host, uint32_t num_arguments, const char** arguments) {
  UNUSED_ARG(parser);
  UNUSED_ARG(host);
  UNUSED_ARG(num_arguments);
  UNUSED_ARG(arguments);

  printf("Mandarin Duck Frontend\n");
  printf("Luminary %s (Branch: %s)\n", LUMINARY_VERSION_DATE, LUMINARY_BRANCH_NAME);
  printf("(%s, %s, CUDA %s, OptiX %s)\n", LUMINARY_COMPILER, LUMINARY_OS, LUMINARY_CUDA_VERSION, LUMINARY_OPTIX_VERSION);

  parser->results.dry_run_requested = true;
}

////////////////////////////////////////////////////////////////////
// Internal functions
////////////////////////////////////////////////////////////////////

struct ParseResult {
  bool is_empty;
  bool is_argument;
  const ArgumentDescriptor* matched_argument;
} typedef ParseResult;

static void _argument_parser_parse_argument(ArgumentParser* parser, const char* argument, ParseResult* result) {
  if (argument[0] == '\0') {
    // This is an empty argument.
    *result = (ParseResult) {.is_empty = true, .is_argument = false, .matched_argument = (const ArgumentDescriptor*) 0};
    return;
  }

  if (argument[0] != '-') {
    // This is an input file
    *result = (ParseResult) {.is_empty = false, .is_argument = false, .matched_argument = (const ArgumentDescriptor*) 0};
    return;
  }

  const bool use_long_name = (argument[1] == '-');

  uint32_t num_available_arguments;
  LUM_FAILURE_HANDLE(array_get_num_elements(parser->descriptors, &num_available_arguments));

  const char* arg_string = (use_long_name) ? argument + 2 : argument + 1;

  for (uint32_t argument_id = 0; argument_id < num_available_arguments; argument_id++) {
    const ArgumentDescriptor* argument = parser->descriptors + argument_id;

    // If this argument cannot match, skip it.
    if ((use_long_name && argument->long_name == (char*) 0) || (!use_long_name && argument->short_name == (char*) 0)) {
      continue;
    }

    const int cmp_val = strcmp(arg_string, use_long_name ? argument->long_name : argument->short_name);

    if (cmp_val != 0) {
      continue;
    }

    *result = (ParseResult) {.is_empty = false, .is_argument = true, .matched_argument = argument};

    return;
  }

  *result = (ParseResult) {.is_empty = false, .is_argument = true, .matched_argument = (const ArgumentDescriptor*) 0};
}

////////////////////////////////////////////////////////////////////
// API functions
////////////////////////////////////////////////////////////////////

void argument_parser_create(ArgumentParser** parser) {
  MD_CHECK_NULL_ARGUMENT(parser);

  host_malloc(parser, sizeof(ArgumentParser));
  memset(*parser, 0, sizeof(ArgumentParser));

  LUM_FAILURE_HANDLE(array_create(&(*parser)->descriptors, sizeof(ArgumentDescriptor), 16));
  LUM_FAILURE_HANDLE(array_create(&(*parser)->inputs, sizeof(char*), 16));
  LUM_FAILURE_HANDLE(array_create(&(*parser)->parsed_arguments, sizeof(ParsedArgument), 16));

  // Add descriptors sorted by category and alphabetically.

  ArgumentDescriptor descriptor;

  descriptor.category          = ARGUMENT_CATEGORY_DEFAULT;
  descriptor.long_name         = "benchmark";
  descriptor.short_name        = "b";
  descriptor.description       = "Run the benchmark";
  descriptor.subargument_count = 1;
  descriptor.handler_func      = (ArgumentHandlerFunc) _argument_parser_arg_func_benchmark;
  descriptor.pre_execute       = false;

  LUM_FAILURE_HANDLE(array_push(&(*parser)->descriptors, &descriptor));

  descriptor.category          = ARGUMENT_CATEGORY_DEFAULT;
  descriptor.long_name         = "help";
  descriptor.short_name        = "h";
  descriptor.description       = "Print available commandline arguments";
  descriptor.subargument_count = 0;
  descriptor.handler_func      = (ArgumentHandlerFunc) _argument_parser_arg_func_help;
  descriptor.pre_execute       = true;

  LUM_FAILURE_HANDLE(array_push(&(*parser)->descriptors, &descriptor));

  descriptor.category          = ARGUMENT_CATEGORY_DEFAULT;
  descriptor.long_name         = "output";
  descriptor.short_name        = "o";
  descriptor.description       = "Set output directory";
  descriptor.subargument_count = 1;
  descriptor.handler_func      = (ArgumentHandlerFunc) _argument_parser_arg_func_output;
  descriptor.pre_execute       = false;

  LUM_FAILURE_HANDLE(array_push(&(*parser)->descriptors, &descriptor));

  descriptor.category          = ARGUMENT_CATEGORY_DEFAULT;
  descriptor.long_name         = "version";
  descriptor.short_name        = "v";
  descriptor.description       = "Print version information";
  descriptor.subargument_count = 0;
  descriptor.handler_func      = (ArgumentHandlerFunc) _argument_parser_arg_func_version;
  descriptor.pre_execute       = true;

  LUM_FAILURE_HANDLE(array_push(&(*parser)->descriptors, &descriptor));
}

void argument_parser_parse(ArgumentParser* parser, uint32_t argc, const char** argv) {
  MD_CHECK_NULL_ARGUMENT(parser);

  uint32_t subargs_left = 0;
  ParseResult subarg_target;
  const char** subargs;
  LUM_FAILURE_HANDLE(array_create(&subargs, sizeof(const char*), 8));

  for (uint32_t argument_id = 1; argument_id < argc; argument_id++) {
    const char* argument = argv[argument_id];

    ParseResult result;
    _argument_parser_parse_argument(parser, argument, &result);

    if (result.is_empty) {
      continue;
    }

    if (result.is_argument) {
      if (subargs_left > 0) {
        warn_message("Encountered argument but expected sub-argument.");
        subargs_left = 0;
        LUM_FAILURE_HANDLE(array_clear(subargs));
      }
    }

    if (result.is_argument && result.matched_argument == (ArgumentDescriptor*) 0) {
      warn_message("Argument: %s is unknown.", argument);
      continue;
    }

    if (result.is_argument && result.matched_argument->subargument_count == 0) {
      if (result.matched_argument->pre_execute) {
        result.matched_argument->handler_func(parser, (LuminaryHost*) 0, 0, (const char**) 0);

        // Encountered argument that prevents Mandarin Duck from actually executing.
        parser->results.dry_run_requested = true;
        break;
      }

      ParsedArgument parsed_argument;

      parsed_argument.handler_func      = result.matched_argument->handler_func;
      parsed_argument.subargument_count = 0;

      LUM_FAILURE_HANDLE(array_push(&parser->parsed_arguments, &parsed_argument));

      continue;
    }

    if (result.is_argument && result.matched_argument->subargument_count > 0) {
      subarg_target = result;
      subargs_left  = result.matched_argument->subargument_count;

      continue;
    }

    if (result.is_argument == false && subargs_left > 0) {
      LUM_FAILURE_HANDLE(array_push(&subargs, &argument));
      subargs_left--;

      if (subargs_left == 0) {
        ParsedArgument parsed_argument;

        parsed_argument.handler_func      = subarg_target.matched_argument->handler_func;
        parsed_argument.subargument_count = subarg_target.matched_argument->subargument_count;

        memcpy(parsed_argument.subarguments, subargs, subarg_target.matched_argument->subargument_count * sizeof(const char*));

        LUM_FAILURE_HANDLE(array_push(&parser->parsed_arguments, &parsed_argument));

        LUM_FAILURE_HANDLE(array_clear(subargs));
      }

      continue;
    }

    if (!result.is_argument) {
      LUM_FAILURE_HANDLE(array_push(&parser->inputs, &argument));

      continue;
    }
  }

  LUM_FAILURE_HANDLE(array_destroy(&subargs));
}

void argument_parser_execute(ArgumentParser* parser, LuminaryHost* host) {
  MD_CHECK_NULL_ARGUMENT(parser);
  MD_CHECK_NULL_ARGUMENT(host);

  uint32_t num_inputs;
  LUM_FAILURE_HANDLE(array_get_num_elements(parser->inputs, &num_inputs));

  for (uint32_t input_id = 0; input_id < num_inputs; input_id++) {
    const char* input = parser->inputs[input_id];

    // TODO: Handle obj file inputs
    LuminaryPath* lum_path;
    LUM_FAILURE_HANDLE(luminary_path_create(&lum_path));
    LUM_FAILURE_HANDLE(luminary_path_set_from_string(lum_path, input));

    LUM_FAILURE_HANDLE(luminary_host_load_lum_file(host, lum_path));

    LUM_FAILURE_HANDLE(luminary_path_destroy(&lum_path));
  }

  uint32_t num_arguments;
  LUM_FAILURE_HANDLE(array_get_num_elements(parser->parsed_arguments, &num_arguments));

  for (uint32_t argument_id = 0; argument_id < num_arguments; argument_id++) {
    const ParsedArgument* argument = parser->parsed_arguments + argument_id;

    argument->handler_func(parser, host, argument->subargument_count, (const char**) argument->subarguments);
  }
}

void argument_parser_destroy(ArgumentParser** parser) {
  MD_CHECK_NULL_ARGUMENT(parser);
  MD_CHECK_NULL_ARGUMENT(*parser);

  LUM_FAILURE_HANDLE(array_destroy(&(*parser)->descriptors));
  LUM_FAILURE_HANDLE(array_destroy(&(*parser)->inputs));
  LUM_FAILURE_HANDLE(array_destroy(&(*parser)->parsed_arguments));

  LUM_FAILURE_HANDLE(host_free(parser));
}
