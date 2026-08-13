# fetch targets for the pretrained weights and the mnist dataset. a fresh clone
# carries neither — both live on the v0.1.0 release, which is what keeps the
# clone around a megabyte instead of sixty.
#
#   cmake --build build --target fetch-cats-decoder   # 14 MB, enough for `cats generate`
#   cmake --build build --target fetch-mnist          # weights and dataset
#   cmake --build build --target fetch-assets         # everything
#
# the targets are deliberately granular: `cats generate` is the headline demo
# and must not cost a 60 MB download.
#
# these are defined here rather than in the example projects because each
# example pulls this directory in with add_subdirectory, so defining them once
# makes them reachable from an example build tree and from a library-only one.
#
# mnist ships decompressed under the exact names the example opens. the mirrors
# serve gzipped idx, and cmake cannot inflate a raw .gz — its libarchive is
# built without the raw reader, so file(ARCHIVE_EXTRACT) and `cmake -E tar xzf`
# both fail with "Unrecognized archive format". re-hosting sidesteps that and
# drops the dependency on someone else's bucket staying up.

set(MYGRAD_ASSET_BASE_URL
    "https://github.com/matraass11/mygrad/releases/download/v0.1.0"
    CACHE STRING "base url the fetch-* targets download release assets from")

set(mygradDownloadScript "${CMAKE_CURRENT_LIST_DIR}/downloadAsset.cmake")
set(mygradExamplesDirectory "${CMAKE_CURRENT_LIST_DIR}/../examples")

# appends one `COMMAND cmake -P downloadAsset.cmake ...` to a command list,
# so a single target can fetch several assets
function(appendFetchCommand commandListVariable assetName destination sha256)
    set(${commandListVariable}
        ${${commandListVariable}}
        COMMAND ${CMAKE_COMMAND}
            -DURL=${MYGRAD_ASSET_BASE_URL}/${assetName}
            -DDEST=${destination}
            -DSHA256=${sha256}
            -P ${mygradDownloadScript}
        PARENT_SCOPE)
endfunction()

function(addFetchTarget targetName assetName destination sha256)
    set(fetchCommand "")
    appendFetchCommand(fetchCommand "${assetName}" "${destination}" "${sha256}")
    add_custom_target(${targetName}
        ${fetchCommand}
        COMMENT "fetching ${assetName}"
        VERBATIM
        USES_TERMINAL)
endfunction()

addFetchTarget(fetch-cats-decoder
    pretrained_decoder.model
    "${mygradExamplesDirectory}/cats/pretrained_decoder.model"
    c3df94dc43ded5f1b1b38523638302215a0ecf31aad5cf4c9ad8d8572c1e71d1)

addFetchTarget(fetch-cats-encoder
    pretrained_encoder.model
    "${mygradExamplesDirectory}/cats/pretrained_encoder.model"
    fbcef90a81c74fbb3e7e98f731a306b8292f94de673f13d661f44210757121a4)

# the release asset is named mnist_classifier.model; the example loads ../model
addFetchTarget(fetch-mnist-model
    mnist_classifier.model
    "${mygradExamplesDirectory}/mnist/model"
    35ef9390f2520f6b6503b9c0e6b3aeff815a3f2134729b4564d020b86181ff14)

set(mnistDatasetDirectory "${mygradExamplesDirectory}/mnist/dataset")
set(mnistDatasetCommands "")
appendFetchCommand(mnistDatasetCommands train-images-ubyte
    "${mnistDatasetDirectory}/train-images-ubyte"
    ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db)
appendFetchCommand(mnistDatasetCommands train-labels-ubyte
    "${mnistDatasetDirectory}/train-labels-ubyte"
    65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5)
appendFetchCommand(mnistDatasetCommands test-images-ubyte
    "${mnistDatasetDirectory}/test-images-ubyte"
    0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7)
appendFetchCommand(mnistDatasetCommands test-labels-ubyte
    "${mnistDatasetDirectory}/test-labels-ubyte"
    ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2)

add_custom_target(fetch-mnist-dataset
    ${mnistDatasetCommands}
    COMMENT "fetching the mnist dataset (55 MB)"
    VERBATIM
    USES_TERMINAL)

add_custom_target(fetch-mnist)
add_dependencies(fetch-mnist fetch-mnist-model fetch-mnist-dataset)

add_custom_target(fetch-cats)
add_dependencies(fetch-cats fetch-cats-decoder fetch-cats-encoder)

add_custom_target(fetch-assets)
add_dependencies(fetch-assets fetch-mnist fetch-cats)
