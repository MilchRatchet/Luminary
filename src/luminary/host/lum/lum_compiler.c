#include "lum_compiler.h"

#include <stdio.h>
#include <string.h>

#include "internal_error.h"
#include "lum_function_tables.h"

LuminaryResult lum_compiler_create(LumCompiler** compiler) {
  __CHECK_NULL_ARGUMENT(compiler);

  __FAILURE_HANDLE(host_malloc(compiler, sizeof(LumCompiler)));
  memset(*compiler, 0, sizeof(LumCompiler));

  return LUMINARY_SUCCESS;
}

#define LUM_VARIABLE_REGISTER_NOT_ASSIGNED 0xFFFFFFFF
#define LUM_VARIABLE_NOT_FOUND 0xFFFFFFFF
#define LUM_COMPILER_MAX_MESSAGE_LENGTH 1024
#define LUM_COMPILER_CONTEXT_STACK_SIZE 16
#define LUM_COMPILER_CONTEXT_STACK_EMPTY 0xFFFFFFFF
#define LUM_FUNCTION_MAX_ARGUMENTS 4

////////////////////////////////////////////////////////////////////
// Compiler Message
////////////////////////////////////////////////////////////////////

enum LumCompilerMessageType {
  LUM_COMPILER_MESSAGE_TYPE_INFO,
  LUM_COMPILER_MESSAGE_TYPE_WARNING,
  LUM_COMPILER_MESSAGE_TYPE_ERROR
} typedef LumCompilerMessageType;

struct LumCompilerMessage {
  LumCompilerMessageType type;
  char message[LUM_COMPILER_MAX_MESSAGE_LENGTH];
  uint32_t line;
  uint32_t col;
} typedef LumCompilerMessage;

static LuminaryResult _lum_compiler_message_init(LumCompilerMessage* message, const LumCompilerMessageType type, const LumToken* token) {
  __CHECK_NULL_ARGUMENT(message);
  __CHECK_NULL_ARGUMENT(token);

  memset(message, 0, sizeof(LumCompilerMessage));

  message->type = type;
  message->line = token->line;
  message->col  = token->col;

  return LUMINARY_SUCCESS;
}

struct LumVariable {
  size_t offset;
  uint32_t current_register_map;
  LumBuiltinType type;
  bool is_named;
  char* name;
} typedef LumVariable;

enum LumCompilerContextType {
  LUM_COMPILER_CONTEXT_TYPE_NULL,
  LUM_COMPILER_CONTEXT_TYPE_STATEMENT,
  LUM_COMPILER_CONTEXT_TYPE_FUNCTION,
  LUM_COMPILER_CONTEXT_TYPE_DECLARATION,
  LUM_COMPILER_CONTEXT_TYPE_VECTOR,
  LUM_COMPILER_CONTEXT_TYPE_OPERATOR
} typedef LumCompilerContextType;

struct LumFunctionContext {
  uint32_t parsed_identifiers;
  uint32_t caller_variable_id;
  LumBuiltinType class;
  uint32_t function_id;
  uint32_t expected_arguments;
  uint32_t argument_variable_ids[LUM_FUNCTION_MAX_ARGUMENTS];
} typedef LumFunctionContext;

struct LumDeclarationContext {
  LumBuiltinType type;
  const char* name;
} typedef LumDeclarationContext;

struct LumCompilerContext {
  LumCompilerContextType type;
  union {
    LumFunctionContext function;
    LumDeclarationContext declaration;
  };
} typedef LumCompilerContext;

////////////////////////////////////////////////////////////////////
// Data Section
////////////////////////////////////////////////////////////////////

struct LumCompilerDataEntry {
  uint32_t size;
  void* data;
} typedef LumCompilerDataEntry;

struct LumCompilerDataSection {
  uint32_t total_bytes;
  ARRAY LumCompilerDataEntry* entries;
} typedef LumCompilerDataSection;

static LuminaryResult _lum_compiler_data_section_add(LumCompilerDataSection* section, uint32_t size, const void* data, uint32_t* offset) {
  __CHECK_NULL_ARGUMENT(section);
  __CHECK_NULL_ARGUMENT(data);
  __CHECK_NULL_ARGUMENT(offset);

  LumCompilerDataEntry entry;
  entry.size = size;

  __FAILURE_HANDLE(host_malloc(&entry.data, size));

  memcpy(entry.data, data, size);

  __FAILURE_HANDLE(array_push(&section->entries, &entry));

  *offset = section->total_bytes;

  section->total_bytes += size;

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Compiler State
////////////////////////////////////////////////////////////////////

struct LumCompilerState {
  LumBinary* binary;
  bool error_occurred;
  ARRAY LumCompilerMessage* messages;
  ARRAY LumVariable* variables;
  uint32_t returned_variable_id;
  uint32_t stack_ptr;
  LumCompilerContext context_stack[LUM_COMPILER_CONTEXT_STACK_SIZE];
  size_t used_stack_size;
  LumCompilerDataSection data;
} typedef LumCompilerState;

static LuminaryResult _lum_compiler_state_create(LumCompilerState** state) {
  __CHECK_NULL_ARGUMENT(state);

  __FAILURE_HANDLE(host_malloc(state, sizeof(LumCompilerState)));
  memset(*state, 0, sizeof(LumCompilerState));

  (*state)->stack_ptr = LUM_COMPILER_CONTEXT_STACK_EMPTY;

  __FAILURE_HANDLE(array_create(&(*state)->messages, sizeof(LumCompilerMessage), 16));
  __FAILURE_HANDLE(array_create(&(*state)->variables, sizeof(LumVariable), 16));

  (*state)->data.total_bytes = 0;
  __FAILURE_HANDLE(array_create(&(*state)->data.entries, sizeof(LumCompilerDataEntry), 16));

  return LUMINARY_SUCCESS;
}

static LumCompilerContextType _lum_compiler_state_get_current_context_type(const LumCompilerState* state) {
  return (state->stack_ptr == LUM_COMPILER_CONTEXT_STACK_EMPTY) ? LUM_COMPILER_CONTEXT_TYPE_NULL
                                                                : state->context_stack[state->stack_ptr].type;
}

static LuminaryResult _lum_compiler_state_destroy(LumCompilerState** state) {
  __CHECK_NULL_ARGUMENT(state);
  __CHECK_NULL_ARGUMENT(*state);

  __FAILURE_HANDLE(array_destroy(&(*state)->data.entries));

  __FAILURE_HANDLE(array_destroy(&(*state)->messages));
  __FAILURE_HANDLE(array_destroy(&(*state)->variables));

  __FAILURE_HANDLE(host_free(state));

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Instructions
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_compiler_emit_return(LumBinary* binary) {
  __CHECK_NULL_ARGUMENT(binary);

  LumInstruction instruction;
  __FAILURE_HANDLE(lum_instruction_encode_ret(&instruction));

  __FAILURE_HANDLE(array_push(&binary->instructions, &instruction));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_emit_call(LumBinary* binary) {
  __CHECK_NULL_ARGUMENT(binary);

  LumInstruction instruction;
  __FAILURE_HANDLE(lum_instruction_encode_call(&instruction));

  __FAILURE_HANDLE(array_push(&binary->instructions, &instruction));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_emit_move(LumBinary* binary) {
  __CHECK_NULL_ARGUMENT(binary);

  LumInstruction instruction;
  __FAILURE_HANDLE(lum_instruction_encode_mov(&instruction));

  __FAILURE_HANDLE(array_push(&binary->instructions, &instruction));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_emit_regmap(LumBinary* binary, uint8_t reg, bool is_data_section, uint32_t offset) {
  __CHECK_NULL_ARGUMENT(binary);

  LumInstruction instruction;
  __FAILURE_HANDLE(lum_instruction_encode_regmap(&instruction, reg, is_data_section, offset));

  __FAILURE_HANDLE(array_push(&binary->instructions, &instruction));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_null_binary(LumBinary* binary) {
  __CHECK_NULL_ARGUMENT(binary);

  __FAILURE_HANDLE(array_clear(binary->instructions));

  for (uint32_t entrypoint_id = 0; entrypoint_id < LUM_BINARY_ENTRY_POINT_COUNT; entrypoint_id++) {
    binary->entry_points[entrypoint_id] = 0;
  }

  binary->stack_frame_size = 0;

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Variables
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_compiler_allocate_variable(LumCompilerState* state, LumBuiltinType type, const char* name, uint32_t* id) {
  __CHECK_NULL_ARGUMENT(state);

  LumVariable new_variable;
  new_variable.offset               = state->used_stack_size;
  new_variable.current_register_map = LUM_VARIABLE_REGISTER_NOT_ASSIGNED;
  new_variable.type                 = type;

  new_variable.is_named = (name != (const char*) 0);

  if (new_variable.is_named) {
    const size_t name_length = strlen(name) + 1;

    __FAILURE_HANDLE(host_malloc(&new_variable.name, name_length));
    memcpy(new_variable.name, name, name_length - 1);
    new_variable.name[name_length - 1] = '\0';
  }

  if (id != (uint32_t*) 0) {
    __FAILURE_HANDLE(array_get_num_elements(state->variables, id));
  }

  __FAILURE_HANDLE(array_push(&state->variables, &new_variable));

  state->used_stack_size += lum_builtin_types_sizes[type];

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_allocate_literal(LumCompilerState* state, const LumToken* token, uint32_t* id) {
  __CHECK_NULL_ARGUMENT(state);
  __CHECK_NULL_ARGUMENT(token);
  __CHECK_NULL_ARGUMENT(id);

  if (token->type != LUM_TOKEN_TYPE_LITERAL) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Token is not a literal.");
  }

  LumBuiltinType type = LUM_BUILTIN_TYPE_UINT32;
  const void* data;

  switch (token->literal.type) {
    case LUM_LITERAL_TYPE_FLOAT:
      type = LUM_BUILTIN_TYPE_FLOAT;
      data = (const void*) &token->literal.val_float;
      break;
    case LUM_LITERAL_TYPE_STRING:
      type = LUM_BUILTIN_TYPE_STRING;
      data = (const void*) token->literal.val_string;
      break;
    default:
      break;
  }

  uint32_t data_section_offset;
  __FAILURE_HANDLE(_lum_compiler_data_section_add(&state->data, lum_builtin_types_sizes[type], data, &data_section_offset));

  __FAILURE_HANDLE(_lum_compiler_allocate_variable(state, type, (const char*) 0, id));

  // TODO: Compute register to use for this variable
  __FAILURE_HANDLE(_lum_compiler_emit_regmap(state->binary, 0, true, data_section_offset));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_find_variable(LumCompilerState* state, const char* name, uint32_t* id) {
  __CHECK_NULL_ARGUMENT(state);
  __CHECK_NULL_ARGUMENT(name);

  uint32_t variable_count;
  __FAILURE_HANDLE(array_get_num_elements(state->variables, &variable_count));

  uint32_t found_id = LUM_VARIABLE_NOT_FOUND;

  for (uint32_t variable_id = 0; variable_id < variable_count; variable_id++) {
    const LumVariable* variable = state->variables + variable_id;

    if (variable->is_named == false)
      continue;

    if (strcmp(name, variable->name) == 0) {
      found_id = variable_id;
      break;
    }
  }

  *id = found_id;

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Context resolution
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_compiler_context_resolve(LumCompilerState* state) {
  __CHECK_NULL_ARGUMENT(state);

  if (state->stack_ptr == LUM_COMPILER_CONTEXT_STACK_EMPTY) {
    LumCompilerMessage message;
    message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
    sprintf(message.message, "error: internal compiler error, resolved empty stack context");
    __FAILURE_HANDLE(array_push(&state->messages, &message));

    state->error_occurred = true;
    return LUMINARY_SUCCESS;
  }

  LumCompilerContext* context = state->context_stack + state->stack_ptr;

  switch (context->type) {
    case LUM_COMPILER_CONTEXT_TYPE_FUNCTION: {
      if (context->function.expected_arguments + 2 > context->function.parsed_identifiers) {
        LumCompilerMessage message;
        message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
        sprintf(message.message, "error: too few arguments to function call");
        __FAILURE_HANDLE(array_push(&state->messages, &message));

        state->error_occurred = true;
        return LUMINARY_SUCCESS;
      }

      __FAILURE_HANDLE(_lum_compiler_emit_call(state->binary));

      // TODO:
      // state->returned_variable_id = instruction.dst_operand;
    } break;
    case LUM_COMPILER_CONTEXT_TYPE_DECLARATION:
      __FAILURE_HANDLE(
        _lum_compiler_allocate_variable(state, context->declaration.type, context->declaration.name, &state->returned_variable_id));
      break;
    case LUM_COMPILER_CONTEXT_TYPE_OPERATOR: {
      __FAILURE_HANDLE(_lum_compiler_emit_move(state->binary));

      state->returned_variable_id = LUM_VARIABLE_NOT_FOUND;
    } break;
    default:
      break;
  }

  state->stack_ptr--;

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Identifier
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_compiler_handle_identifier_null_context(LumCompilerState* state, const LumToken* token) {
  __CHECK_NULL_ARGUMENT(state);
  __CHECK_NULL_ARGUMENT(token);

  if (token->identifier.is_builtin_type) {
    LumCompilerContext context;
    context.type             = LUM_COMPILER_CONTEXT_TYPE_DECLARATION;
    context.declaration.type = token->identifier.builtin_type;

    state->context_stack[++state->stack_ptr] = context;
  }
  else {
    uint32_t corresponding_variable;
    __FAILURE_HANDLE(_lum_compiler_find_variable(state, token->identifier.name, &corresponding_variable));

    if (corresponding_variable == LUM_VARIABLE_NOT_FOUND) {
      LumCompilerMessage message;
      message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
      sprintf(message.message, "<source>:%u:%u: error: use of undeclared identifier `%s`", token->line, token->col, token->identifier.name);
      __FAILURE_HANDLE(array_push(&state->messages, &message));

      state->error_occurred = true;
      return LUMINARY_SUCCESS;
    }

    LumCompilerContext context;
    context.type = LUM_COMPILER_CONTEXT_TYPE_STATEMENT;

    state->context_stack[++state->stack_ptr] = context;
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_handle_identifier_function_context(LumCompilerState* state, const LumToken* token) {
  __CHECK_NULL_ARGUMENT(state);
  __CHECK_NULL_ARGUMENT(token);

  const uint32_t identifier_id = state->context_stack[state->stack_ptr].function.parsed_identifiers;

  switch (identifier_id) {
    case 0:
      if (token->identifier.is_builtin_type) {
        state->context_stack[state->stack_ptr].function.class              = token->identifier.builtin_type;
        state->context_stack[state->stack_ptr].function.caller_variable_id = LUM_VARIABLE_NOT_FOUND;
        state->context_stack[state->stack_ptr].function.parsed_identifiers++;
      }
      else {
        uint32_t corresponding_variable;
        __FAILURE_HANDLE(_lum_compiler_find_variable(state, token->identifier.name, &corresponding_variable));

        if (corresponding_variable == LUM_VARIABLE_NOT_FOUND) {
          LumCompilerMessage message;
          message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
          sprintf(message.message, "<source>:%u:%u: error: `%s` was not declared", token->line, token->col, token->identifier.name);
          __FAILURE_HANDLE(array_push(&state->messages, &message));

          state->error_occurred = true;
          return LUMINARY_SUCCESS;
        }

        const LumVariable* variable = state->variables + corresponding_variable;

        state->context_stack[state->stack_ptr].function.class              = variable->type;
        state->context_stack[state->stack_ptr].function.caller_variable_id = corresponding_variable;
        state->context_stack[state->stack_ptr].function.parsed_identifiers++;
      }
      break;
    case 1:
      if (token->identifier.is_builtin_type) {
        LumCompilerMessage message;
        message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
        sprintf(message.message, "<source>:%u:%u: error: `%s` does not name a function", token->line, token->col, token->identifier.name);
        __FAILURE_HANDLE(array_push(&state->messages, &message));

        state->error_occurred = true;
        return LUMINARY_SUCCESS;
      }
      else {
        const LumBuiltinType type = state->context_stack[state->stack_ptr].function.class;

        const LumFunctionEntry* functions = lum_function_tables[type];
        const uint32_t num_functions      = lum_function_tables_count[type];

        uint32_t corresponding_function = 0xFFFFFFFF;

        for (uint32_t function_id = 0; function_id < num_functions; function_id++) {
          const LumFunctionEntry* function = functions + function_id;

          if (strcmp(function->name, token->identifier.name) == 0) {
            corresponding_function = function_id;
            break;
          }
        }

        if (corresponding_function == 0xFFFFFFFF) {
          LumCompilerMessage message;
          message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
          sprintf(message.message, "<source>:%u:%u: error: `%s` does not name a function", token->line, token->col, token->identifier.name);
          __FAILURE_HANDLE(array_push(&state->messages, &message));

          state->error_occurred = true;
          return LUMINARY_SUCCESS;
        }

        const LumFunctionEntry* function = functions + corresponding_function;

        const bool caller_is_variable = state->context_stack[state->stack_ptr].function.caller_variable_id != LUM_VARIABLE_NOT_FOUND;

        if (caller_is_variable == function->is_static) {
          LumCompilerMessage message;
          message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;

          if (function->is_static) {
            sprintf(
              message.message, "<source>:%u:%u: error: cannot call static function `%s` with object", token->line, token->col,
              token->identifier.name);
          }
          else {
            sprintf(
              message.message, "<source>:%u:%u: error: cannot call member function `%s` without object", token->line, token->col,
              token->identifier.name);
          }

          __FAILURE_HANDLE(array_push(&state->messages, &message));

          state->error_occurred = true;
          return LUMINARY_SUCCESS;
        }

        uint32_t expected_arguments = 0;
        expected_arguments += (function->signature.src_a != LUM_DATA_TYPE_NULL) ? 1 : 0;
        expected_arguments += (function->signature.src_b != LUM_DATA_TYPE_NULL) ? 1 : 0;
        expected_arguments += (function->signature.src_c != LUM_DATA_TYPE_NULL) ? 1 : 0;

        state->context_stack[state->stack_ptr].function.function_id        = corresponding_function;
        state->context_stack[state->stack_ptr].function.expected_arguments = expected_arguments;
        state->context_stack[state->stack_ptr].function.parsed_identifiers++;
      }
      break;
    default: {
      const uint32_t expected_arguments = state->context_stack[state->stack_ptr].function.expected_arguments;
      if (identifier_id >= 2 + expected_arguments) {
        LumCompilerMessage message;
        message.type = LUM_COMPILER_MESSAGE_TYPE_WARNING;
        sprintf(message.message, "<source>:%u:%u: warning: too many arguments in function call", token->line, token->col);
        __FAILURE_HANDLE(array_push(&state->messages, &message));

        state->error_occurred = true;
        return LUMINARY_SUCCESS;
      }

      uint32_t corresponding_variable;
      __FAILURE_HANDLE(_lum_compiler_find_variable(state, token->identifier.name, &corresponding_variable));

      if (corresponding_variable == LUM_VARIABLE_NOT_FOUND) {
        LumCompilerMessage message;
        message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
        sprintf(
          message.message, "<source>:%u:%u: error: use of undeclared identifier `%s`", token->line, token->col, token->identifier.name);
        __FAILURE_HANDLE(array_push(&state->messages, &message));

        state->error_occurred = true;
        return LUMINARY_SUCCESS;
      }

      state->context_stack[state->stack_ptr].function.argument_variable_ids[identifier_id - 2] = corresponding_variable;
      state->context_stack[state->stack_ptr].function.parsed_identifiers++;
    } break;
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_handle_identifier_declaration_context(LumCompilerState* state, const LumToken* token) {
  __CHECK_NULL_ARGUMENT(state);
  __CHECK_NULL_ARGUMENT(token);

  if (token->identifier.is_builtin_type) {
    LumCompilerMessage message;
    message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
    sprintf(message.message, "<source>:%u:%u: error: `%s` names a type", token->line, token->col, token->identifier.name);
    __FAILURE_HANDLE(array_push(&state->messages, &message));

    state->error_occurred = true;
  }
  else {
    uint32_t corresponding_variable;
    __FAILURE_HANDLE(_lum_compiler_find_variable(state, token->identifier.name, &corresponding_variable));

    if (corresponding_variable != LUM_VARIABLE_NOT_FOUND) {
      LumCompilerMessage message;
      message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
      sprintf(message.message, "<source>:%u:%u: error: redefinition of `%s`", token->line, token->col, token->identifier.name);
      __FAILURE_HANDLE(array_push(&state->messages, &message));

      state->error_occurred = true;
      return LUMINARY_SUCCESS;
    }

    state->context_stack[state->stack_ptr].declaration.name = token->identifier.name;

    __FAILURE_HANDLE(_lum_compiler_context_resolve(state));
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_handle_identifier(LumCompilerState* state, const LumToken* token) {
  __CHECK_NULL_ARGUMENT(state);
  __CHECK_NULL_ARGUMENT(token);

  LumCompilerContextType type = _lum_compiler_state_get_current_context_type(state);

  switch (type) {
    case LUM_COMPILER_CONTEXT_TYPE_NULL:
      __FAILURE_HANDLE(_lum_compiler_handle_identifier_null_context(state, token));
      break;
    case LUM_COMPILER_CONTEXT_TYPE_FUNCTION:
      __FAILURE_HANDLE(_lum_compiler_handle_identifier_function_context(state, token));
      break;
    case LUM_COMPILER_CONTEXT_TYPE_DECLARATION:
      __FAILURE_HANDLE(_lum_compiler_handle_identifier_declaration_context(state, token));
      break;
    default:
      break;
  }

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Literal
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_compiler_handle_literal_function_context(LumCompilerState* state, const LumToken* token) {
  __CHECK_NULL_ARGUMENT(state);
  __CHECK_NULL_ARGUMENT(token);

  const uint32_t identifier_id = state->context_stack[state->stack_ptr].function.parsed_identifiers;

  if (identifier_id < 2) {
    LumCompilerMessage message;
    __FAILURE_HANDLE(_lum_compiler_message_init(&message, LUM_COMPILER_MESSAGE_TYPE_ERROR, token));
    sprintf(message.message, "<source>:%u:%u: error: unexpected literal", token->line, token->col);
    __FAILURE_HANDLE(array_push(&state->messages, &message));
    return LUMINARY_SUCCESS;
  }

  uint32_t literal_variable;
  __FAILURE_HANDLE(_lum_compiler_allocate_literal(state, token, &literal_variable));

  state->context_stack[state->stack_ptr].function.argument_variable_ids[identifier_id - 2] = literal_variable;
  state->context_stack[state->stack_ptr].function.parsed_identifiers++;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_handle_literal(LumCompilerState* state, const LumToken* token) {
  __CHECK_NULL_ARGUMENT(state);

  LumCompilerContextType type = _lum_compiler_state_get_current_context_type(state);

  switch (type) {
    case LUM_COMPILER_CONTEXT_TYPE_NULL: {
      LumCompilerMessage message;
      __FAILURE_HANDLE(_lum_compiler_message_init(&message, LUM_COMPILER_MESSAGE_TYPE_WARNING, token));
      sprintf(message.message, "<source>:%u:%u: error: unexpected literal", token->line, token->col);
      __FAILURE_HANDLE(array_push(&state->messages, &message));
      return LUMINARY_SUCCESS;
    } break;
    case LUM_COMPILER_CONTEXT_TYPE_FUNCTION:
      __FAILURE_HANDLE(_lum_compiler_handle_literal_function_context(state, token));
      break;
    default:
      break;
  }

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Operator
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_compiler_handle_operator(LumCompilerState* state, const LumToken* token) {
  __CHECK_NULL_ARGUMENT(state);

  LumCompilerContextType type = _lum_compiler_state_get_current_context_type(state);

  if (type != LUM_COMPILER_CONTEXT_TYPE_NULL) {
    LumCompilerMessage message;
    message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
    sprintf(message.message, "<source>:%u:%u: error: unexpected operator", token->line, token->col);
    __FAILURE_HANDLE(array_push(&state->messages, &message));

    state->error_occurred = true;
    return LUMINARY_SUCCESS;
  }

  LumCompilerContext context;
  context.type = LUM_COMPILER_CONTEXT_TYPE_OPERATOR;

  state->context_stack[++state->stack_ptr] = context;

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Separator
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_compiler_handle_separator_null_context(LumCompilerState* state, const LumToken* token) {
  __CHECK_NULL_ARGUMENT(state);
  __CHECK_NULL_ARGUMENT(token);

  switch (token->separator.type) {
    case LUM_SEPARATOR_TYPE_EOL:
      break;
    case LUM_SEPARATOR_TYPE_FUNC_BEGIN: {
      LumCompilerContext context;
      context.type                        = LUM_COMPILER_CONTEXT_TYPE_FUNCTION;
      context.function.parsed_identifiers = 0;

      state->context_stack[++state->stack_ptr] = context;
    } break;
    case LUM_SEPARATOR_TYPE_FUNC_END: {
      LumCompilerMessage message;
      message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
      sprintf(message.message, "<source>:%u:%u: error: unexpected end of function", token->line, token->col);
      __FAILURE_HANDLE(array_push(&state->messages, &message));

      state->error_occurred = true;
    } break;
    case LUM_SEPARATOR_TYPE_MEMBER: {
      LumCompilerMessage message;
      message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
      sprintf(message.message, "<source>:%u:%u: error: unexpected member separator", token->line, token->col);
      __FAILURE_HANDLE(array_push(&state->messages, &message));

      state->error_occurred = true;
    } break;
    case LUM_SEPARATOR_TYPE_VECTOR_BEGIN: {
      LumCompilerMessage message;
      message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
      sprintf(message.message, "<source>:%u:%u: error: unexpected begin of vector", token->line, token->col);
      __FAILURE_HANDLE(array_push(&state->messages, &message));

      state->error_occurred = true;
    } break;
    case LUM_SEPARATOR_TYPE_VECTOR_END: {
      LumCompilerMessage message;
      message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
      sprintf(message.message, "<source>:%u:%u: error: unexpected end of vector", token->line, token->col);
      __FAILURE_HANDLE(array_push(&state->messages, &message));

      state->error_occurred = true;
    } break;
    case LUM_SEPARATOR_TYPE_LIST: {
      LumCompilerMessage message;
      message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
      sprintf(message.message, "<source>:%u:%u: error: unexpected list separator", token->line, token->col);
      __FAILURE_HANDLE(array_push(&state->messages, &message));

      state->error_occurred = true;
    } break;
    default:
      break;
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_handle_separator_function_context(LumCompilerState* state, const LumToken* token) {
  __CHECK_NULL_ARGUMENT(state);

  switch (token->separator.type) {
    case LUM_SEPARATOR_TYPE_EOL:
      LumCompilerMessage message;
      message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
      sprintf(message.message, "<source>:%u:%u: error: unexpected end of line", token->line, token->col);
      __FAILURE_HANDLE(array_push(&state->messages, &message));

      state->error_occurred = true;
      break;
    case LUM_SEPARATOR_TYPE_FUNC_BEGIN: {
      LumCompilerContext context;
      context.type                        = LUM_COMPILER_CONTEXT_TYPE_FUNCTION;
      context.function.parsed_identifiers = 0;

      state->context_stack[++state->stack_ptr] = context;
    } break;
    case LUM_SEPARATOR_TYPE_FUNC_END: {
      __FAILURE_HANDLE(_lum_compiler_context_resolve(state));
    } break;
    case LUM_SEPARATOR_TYPE_MEMBER: {
      LumCompilerMessage message;
      message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
      sprintf(message.message, "<source>:%u:%u: error: unexpected member separator", token->line, token->col);
      __FAILURE_HANDLE(array_push(&state->messages, &message));

      state->error_occurred = true;
    } break;
    case LUM_SEPARATOR_TYPE_VECTOR_BEGIN: {
      LumCompilerMessage message;
      message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
      sprintf(message.message, "<source>:%u:%u: error: unexpected begin of vector", token->line, token->col);
      __FAILURE_HANDLE(array_push(&state->messages, &message));

      state->error_occurred = true;
    } break;
    case LUM_SEPARATOR_TYPE_VECTOR_END: {
      LumCompilerMessage message;
      message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
      sprintf(message.message, "<source>:%u:%u: error: unexpected end of vector", token->line, token->col);
      __FAILURE_HANDLE(array_push(&state->messages, &message));

      state->error_occurred = true;
    } break;
    case LUM_SEPARATOR_TYPE_LIST: {
      LumCompilerMessage message;
      message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
      sprintf(message.message, "<source>:%u:%u: error: unexpected list separator", token->line, token->col);
      __FAILURE_HANDLE(array_push(&state->messages, &message));

      state->error_occurred = true;
    } break;
    default:
      break;
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_handle_separator_operator_context(LumCompilerState* state, const LumToken* token) {
  __CHECK_NULL_ARGUMENT(state);
  __CHECK_NULL_ARGUMENT(token);

  switch (token->separator.type) {
    case LUM_SEPARATOR_TYPE_EOL:
      __FAILURE_HANDLE(_lum_compiler_context_resolve(state));
      break;
    case LUM_SEPARATOR_TYPE_FUNC_BEGIN: {
      LumCompilerContext context;
      context.type                        = LUM_COMPILER_CONTEXT_TYPE_FUNCTION;
      context.function.parsed_identifiers = 0;

      state->context_stack[++state->stack_ptr] = context;
    } break;
    case LUM_SEPARATOR_TYPE_FUNC_END: {
      LumCompilerMessage message;
      message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
      sprintf(message.message, "<source>:%u:%u: error: unexpected end of function", token->line, token->col);
      __FAILURE_HANDLE(array_push(&state->messages, &message));

      state->error_occurred = true;
    } break;
    case LUM_SEPARATOR_TYPE_MEMBER: {
      LumCompilerMessage message;
      message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
      sprintf(message.message, "<source>:%u:%u: error: unexpected member separator", token->line, token->col);
      __FAILURE_HANDLE(array_push(&state->messages, &message));

      state->error_occurred = true;
    } break;
    case LUM_SEPARATOR_TYPE_VECTOR_BEGIN: {
      LumCompilerMessage message;
      message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
      sprintf(message.message, "<source>:%u:%u: error: unexpected begin of vector", token->line, token->col);
      __FAILURE_HANDLE(array_push(&state->messages, &message));

      state->error_occurred = true;
    } break;
    case LUM_SEPARATOR_TYPE_VECTOR_END: {
      LumCompilerMessage message;
      message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
      sprintf(message.message, "<source>:%u:%u: error: unexpected end of vector", token->line, token->col);
      __FAILURE_HANDLE(array_push(&state->messages, &message));

      state->error_occurred = true;
    } break;
    case LUM_SEPARATOR_TYPE_LIST: {
      LumCompilerMessage message;
      message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
      sprintf(message.message, "<source>:%u:%u: error: unexpected list separator", token->line, token->col);
      __FAILURE_HANDLE(array_push(&state->messages, &message));

      state->error_occurred = true;
    } break;
    default:
      break;
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_handle_separator(LumCompilerState* state, const LumToken* token) {
  __CHECK_NULL_ARGUMENT(state);

  LumCompilerContextType type = _lum_compiler_state_get_current_context_type(state);

  switch (type) {
    case LUM_COMPILER_CONTEXT_TYPE_NULL:
      __FAILURE_HANDLE(_lum_compiler_handle_separator_null_context(state, token));
      break;
    case LUM_COMPILER_CONTEXT_TYPE_OPERATOR:
      __FAILURE_HANDLE(_lum_compiler_handle_separator_operator_context(state, token));
      break;
    case LUM_COMPILER_CONTEXT_TYPE_FUNCTION:
      __FAILURE_HANDLE(_lum_compiler_handle_separator_function_context(state, token));
    default:
      break;
  }

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////////////

LuminaryResult lum_compiler_compile(LumCompiler* compiler, ARRAY const LumToken* tokens, LumBinary* binary) {
  __CHECK_NULL_ARGUMENT(compiler);
  __CHECK_NULL_ARGUMENT(tokens);
  __CHECK_NULL_ARGUMENT(binary);

  LumCompilerState* state;
  __FAILURE_HANDLE(_lum_compiler_state_create(&state));

  state->binary = binary;

  uint32_t token_count;
  __FAILURE_HANDLE(array_get_num_elements(tokens, &token_count));

  uint32_t token_id = 0;
  while (token_id < token_count) {
    const LumToken* token = tokens + token_id;

    switch (token->type) {
      case LUM_TOKEN_TYPE_IDENTIFIER:
        __FAILURE_HANDLE(_lum_compiler_handle_identifier(state, token));
        break;
      case LUM_TOKEN_TYPE_KEYWORD:
        // No keywords
        break;
      case LUM_TOKEN_TYPE_LITERAL:
        __FAILURE_HANDLE(_lum_compiler_handle_literal(state, token));
        break;
      case LUM_TOKEN_TYPE_OPERATOR:
        __FAILURE_HANDLE(_lum_compiler_handle_operator(state, token));
        break;
      case LUM_TOKEN_TYPE_SEPARATOR:
        __FAILURE_HANDLE(_lum_compiler_handle_separator(state, token));
        break;
      default:
        break;
    }

    if (state->error_occurred) {
      __FAILURE_HANDLE(_lum_compiler_null_binary(binary));
      break;
    }

    token_id++;
  }

  __FAILURE_HANDLE(_lum_compiler_emit_return(binary));

  uint32_t message_count;
  __FAILURE_HANDLE(array_get_num_elements(state->messages, &message_count));

  uint32_t num_errors = 0;

  for (uint32_t message_id = 0; message_id < message_count; message_id++) {
    const LumCompilerMessage* message = state->messages + message_id;

    switch (message->type) {
      case LUM_COMPILER_MESSAGE_TYPE_INFO:
        if (compiler->log_level >= 2) {
          info_message("%s", message->message);
        }
        break;
      case LUM_COMPILER_MESSAGE_TYPE_WARNING:
        if (compiler->log_level >= 1) {
          warn_message("%s", message->message);
        }
        break;
      case LUM_COMPILER_MESSAGE_TYPE_ERROR:
        error_message("%s", message->message);
        num_errors++;
        break;
      default:
        break;
    }
  }

  if (num_errors) {
    info_message("%u error%s generated.", num_errors, (num_errors > 1) ? "s" : "");
  }

  __FAILURE_HANDLE(_lum_compiler_state_destroy(&state));

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_compiler_destroy(LumCompiler** compiler) {
  __CHECK_NULL_ARGUMENT(compiler);
  __CHECK_NULL_ARGUMENT(*compiler);

  __FAILURE_HANDLE(host_free(compiler));

  return LUMINARY_SUCCESS;
}
