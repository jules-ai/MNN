include(ExternalProject)
include(ProcessorCount)

set(DOWNLOAD_URL https://github.com/oneapi-src/oneDNN/archive/v3.10.2.zip)
set(DOWNLOAD_HASH "SHA256=76913c8ef97e9ce58118368b5b46d90459c0b4552452f34ea592b8709a7d6dfe")
set(ROOT ${CMAKE_CURRENT_LIST_DIR}/../3rd_party/)
set(ONEDNN_DIR ${ROOT}/oneDNN/)
set(MNN_BUILD_DIR ${CMAKE_CURRENT_LIST_DIR}/../build/)
ProcessorCount(PROCESSOR_COUNT)

set(CONFIGURE_CMD cd ${ONEDNN_DIR} && cmake -DCMAKE_INSTALL_PREFIX=${MNN_BUILD_DIR} -DONEDNN_BUILD_GRAPH=OFF -DDNNL_BUILD_EXAMPLES=OFF -DDNNL_BUILD_TESTS=OFF -DDNNL_CPU_RUNTIME=SEQ)
set(BUILD_CMD cd ${ONEDNN_DIR} && make -j${PROCESSOR_COUNT})
set(INSTALL_CMD cd ${ONEDNN_DIR} && make install)

ExternalProject_Add(oneDNN
    PREFIX              oneDNN
    URL                 ${DOWNLOAD_URL}
    URL_HASH            ${DOWNLOAD_HASH}
    DOWNLOAD_DIR        ${ROOT}
    SOURCE_DIR          ${ONEDNN_DIR}
    CONFIGURE_COMMAND   ${CONFIGURE_CMD}
    BUILD_COMMAND       ${BUILD_CMD}
    INSTALL_COMMAND     ${INSTALL_CMD}
)

