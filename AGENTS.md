# Agent Instructions

- Never build the project.
- Never run tests.
- Do not invoke scripts or tools that build the project or run tests.

# Style guide

- To negate bool types, never use `!`. Always use `== false`.

# Host Code Hints

- If errors occur as part of the LuminaryResult system, cleanup is not required. Any leaks you identify through this system are intentional and can be ignored.

# GPU Performance hints

- The code is compiled with -use-fast-math, this means fast approximate instructions are used automatically, there is no need to use intrinsics for that purpose.
- Using fma instead of writing a * b + c is usually slower because the compiler contracts this operation already if it is faster. However, in rare cases it can be faster not to contract this expression but using fma would force the compiler to contract, resulting in slower code.
- Target platforms are Nvidia SM75 and later.
- The code is extremely performance oriented, so even very small things are relevant.
