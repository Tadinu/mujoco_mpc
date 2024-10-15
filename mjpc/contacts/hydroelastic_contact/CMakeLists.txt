# drake
find_package(drake CONFIG REQUIRED)
# /opt/drake/lib/cmake/drake/drake-config.cmake
message("DRAKE: ${drake_VERSION}")

set(HYDROELASTIC_CONTACT_HEADERS
        common_types.h
        hydroelastic_contact.h
        sensors/float3.h
        sensors/bvh.h
        sensors/tactile_sensor_base.h
        sensors/flat_tactile_sensor.h
        sensors/curved_sensor.h
        sensors/taxel_sensor.h
)

target_sources(
        libmjpc PRIVATE
        hydroelastic_contact.cpp
        sensors/bvh.cpp
        sensors/tactile_sensor_base.cpp
        sensors/flat_tactile_sensor.cpp
        sensors/curved_sensor.cpp
        sensors/taxel_sensor.cpp
        ${HYDROELASTIC_CONTACT_HEADERS}
)
target_link_libraries(
        libmjpc
        drake::drake
)
target_include_directories(libmjpc
        PUBLIC
        ${drake_INCLUDE_DIR}
)
