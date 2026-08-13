# script-mode helper behind the fetch targets in assets.cmake. downloads one
# asset and verifies its sha256:
#
#   cmake -DURL=<url> -DDEST=<path> -DSHA256=<hash> -P downloadAsset.cmake
#
# the download lands on a .part file and is renamed only after the hash checks
# out, so an interrupted fetch never leaves a half-written model where an
# example would happily load it as weights.

foreach(argument URL DEST SHA256)
    if(NOT DEFINED ${argument})
        message(FATAL_ERROR "downloadAsset.cmake: ${argument} was not set")
    endif()
endforeach()

get_filename_component(assetName "${DEST}" NAME)
string(TOLOWER "${SHA256}" expectedHash)

# re-running a fetch target after a failed one should only pull what is missing
if(EXISTS "${DEST}")
    file(SHA256 "${DEST}" existingHash)
    if(existingHash STREQUAL "${expectedHash}")
        message(STATUS "${assetName}: already present, skipping")
        return()
    endif()
    message(STATUS "${assetName}: present but its hash differs, fetching again")
endif()

get_filename_component(destinationDirectory "${DEST}" DIRECTORY)
file(MAKE_DIRECTORY "${destinationDirectory}")

message(STATUS "${assetName}: downloading from ${URL}")
file(DOWNLOAD "${URL}" "${DEST}.part"
    STATUS downloadStatus
    TLS_VERIFY ON
    SHOW_PROGRESS
)

list(GET downloadStatus 0 downloadCode)
if(NOT downloadCode EQUAL 0)
    list(GET downloadStatus 1 downloadMessage)
    file(REMOVE "${DEST}.part")
    message(FATAL_ERROR "${assetName}: download failed (${downloadMessage})")
endif()

file(SHA256 "${DEST}.part" downloadedHash)
if(NOT downloadedHash STREQUAL "${expectedHash}")
    file(REMOVE "${DEST}.part")
    message(FATAL_ERROR
        "${assetName}: sha256 mismatch\n"
        "  expected ${expectedHash}\n"
        "  got      ${downloadedHash}")
endif()

file(REMOVE "${DEST}")
file(RENAME "${DEST}.part" "${DEST}")
message(STATUS "${assetName}: done")
