execute_process(COMMAND git log --pretty=format:'%h' -n 1
                OUTPUT_VARIABLE GIT_REV
                ERROR_QUIET)

# Check whether we got any revision (which isn't
# always the case, e.g. when someone downloaded a zip
# file from Github instead of a checkout)
if("${GIT_REV}" STREQUAL "")
    set(GIT_COMMIT_DATE "Unknown")
    set(GIT_BRANCH_NAME "Unknown")
    set(GIT_COMMIT_HASH "Unknown")
else()
    execute_process(
        COMMAND git log -1 --format=%cd
        RESULT_VARIABLE RESULT
        OUTPUT_VARIABLE GIT_COMMIT_DATE)

    string (REGEX REPLACE "\n" ""  GIT_COMMIT_DATE ${GIT_COMMIT_DATE})

    execute_process(
        COMMAND git rev-parse --abbrev-ref HEAD
        RESULT_VARIABLE RESULT
        OUTPUT_VARIABLE GIT_BRANCH_NAME)

    string (REGEX REPLACE "\n" ""  GIT_BRANCH_NAME ${GIT_BRANCH_NAME})

    execute_process(
        COMMAND git rev-parse --short HEAD
        RESULT_VARIABLE RESULT
        OUTPUT_VARIABLE GIT_COMMIT_HASH)

    string (REGEX REPLACE "\n" ""  GIT_COMMIT_HASH ${GIT_COMMIT_HASH})
endif()

set(GIT_STATE_HEADER_CODE
    "#define GIT_COMMIT_DATE \"${GIT_COMMIT_DATE}\"
     #define GIT_BRANCH_NAME \"${GIT_BRANCH_NAME}\"
     #define GIT_COMMIT_HASH \"${GIT_COMMIT_HASH}\"
     ")

if(EXISTS ${GIT_STATE_TARGET_DIR}/git_state.h)
    file(READ ${GIT_STATE_TARGET_DIR}/git_state.h CURRENT_GIT_STATE_HEADER_CODE)
else()
    set(CURRENT_GIT_STATE_HEADER_CODE "")
endif()

if (NOT "${GIT_STATE_HEADER_CODE}" STREQUAL "${CURRENT_GIT_STATE_HEADER_CODE}")
    file(WRITE ${GIT_STATE_TARGET_DIR}/git_state.h "${GIT_STATE_HEADER_CODE}")
endif()
