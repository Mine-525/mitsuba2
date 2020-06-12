#pragma once

#ifdef __CUDACC__
// List all shape's CUDA header files to be included in the PTX code generation.
// Those header files are located in '/mitsuba/src/shapes/optix/'
#include "cylinder.cuh"
#include "disk.cuh"
#include "mesh.cuh"
#include "rectangle.cuh"
#include "sphere.cuh"
#else

#include <mitsuba/render/optix/common.h>
#include <mitsuba/render/optix_api.h>
#include <mitsuba/render/shape.h>

NAMESPACE_BEGIN(mitsuba)
/// List of the custom shapes supported by OptiX
static std::string custom_optix_shapes[] = {
    "Disk", "Rectangle", "Sphere", "Cylinder",
};
static constexpr size_t custom_optix_shapes_count = std::size(custom_optix_shapes);

/// Retrieve index of custom shape descriptor in the list above for a given shape
template <typename Shape>
size_t get_shape_descr_idx(Shape *shape) {
    std::string name = shape->class_()->name();
    for (size_t i = 0; i < custom_optix_shapes_count; i++) {
        if (custom_optix_shapes[i] == name)
            return i;
    }
    Throw("Unexpected shape: %s. Couldn't be found in the "
          "'custom_optix_shapes' table.", name);
}

struct OptixAccelData {
    OptixTraversableHandle handle = 0ull;
    void* buffer_meshes = nullptr;
    void* buffer_others = nullptr;
    void* buffer_ias = nullptr;

    ~OptixAccelData() {
        if (handle) {
            cuda_free(buffer_meshes);
            cuda_free(buffer_others);
            cuda_free(buffer_ias);
        }
    }
};

template <typename Shape>
void fill_hitgroup_records(std::vector<HitGroupSbtRecord> &hitgroup_records,
                           std::vector<ref<Shape>> &shapes,
                           OptixProgramGroup *program_groups) {
    for (size_t i = 0; i < 2; i++) {
        for (Shape* shape: shapes) {
            // This trick allows meshes to be processed first
            if (i == !shape->is_mesh())
                shape->optix_fill_hitgroup_records(hitgroup_records, program_groups);
        }
    }
}

template <typename Shape>
void build_gas(std::vector<ref<Shape>> &shapes,
               const OptixDeviceContext &context,
               uint32_t base_sbt_offset,
               OptixAccelData &accel) {

    if (shapes.empty())
        return;

    // ----------------------------------------
    //  Build GAS for meshes and custom shapes
    // ----------------------------------------

    std::vector<Shape*> shape_meshes, shape_instances, shape_others;
    for (Shape* shape: shapes) {
        if (shape->is_mesh())
            shape_meshes.push_back(shape);
        else if (shape->is_instance())
            shape_instances.push_back(shape);
        else
            shape_others.push_back(shape);
    }

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;
    accel_options.motionOptions.numKeys = 0;

    // Lambda function to build a GAS given a subset of shape pointers
    auto build_gas = [&context, &accel_options](std::vector<Shape*> &shape_subset, void* &output_buffer) {
        if (output_buffer) {
            cuda_free((void*)output_buffer);
            output_buffer = nullptr;
        }

        size_t shapes_count = shape_subset.size();

        if (shapes_count == 0)
            return OptixTraversableHandle(0);

        std::vector<OptixBuildInput> build_inputs(shapes_count);
        for (size_t i = 0; i < shapes_count; i++)
            shape_subset[i]->optix_build_input(build_inputs[i]);

        OptixAccelBufferSizes buffer_sizes;
        rt_check(optixAccelComputeMemoryUsage(
            context,
            &accel_options,
            build_inputs.data(),
            (unsigned int) shapes_count,
            &buffer_sizes
        ));

        void* d_temp_buffer = cuda_malloc(buffer_sizes.tempSizeInBytes);
        output_buffer = cuda_malloc(buffer_sizes.outputSizeInBytes + 8);

        OptixAccelEmitDesc emit_property = {};
        emit_property.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emit_property.result = (CUdeviceptr)((char*)output_buffer + buffer_sizes.outputSizeInBytes);

        OptixTraversableHandle accel;
        rt_check(optixAccelBuild(
            context,
            0,              // CUDA stream
            &accel_options,
            build_inputs.data(),
            (unsigned int) shapes_count, // num build inputs
            (CUdeviceptr)d_temp_buffer,
            buffer_sizes.tempSizeInBytes,
            (CUdeviceptr)output_buffer,
            buffer_sizes.outputSizeInBytes,
            &accel,
            &emit_property,  // emitted property list
            1                // num emitted properties
        ));

        cuda_free((void*)d_temp_buffer);

        size_t compact_size;
        cuda_memcpy_from_device(&compact_size, (void*)emit_property.result, sizeof(size_t));
        if (compact_size < buffer_sizes.outputSizeInBytes) {
            void* compact_buffer = cuda_malloc(compact_size);
            // Use handle as input and output
            rt_check(optixAccelCompact(
                context,
                0, // CUDA stream
                accel,
                (CUdeviceptr)compact_buffer,
                compact_size,
                &accel
            ));
            cuda_free((void*)output_buffer);
            output_buffer = compact_buffer;
        }

        return accel;
    };

    OptixTraversableHandle meshes_accel = build_gas(shape_meshes, accel.buffer_meshes);
    OptixTraversableHandle others_accel = build_gas(shape_others, accel.buffer_others);

    // ----------------------------------------------------------
    //  Build IAS to support mixture of meshes, instances and custom shapes
    // ----------------------------------------------------------

    if (!shape_others.empty() && shape_meshes.empty() && shape_instances.empty())
        accel.handle = others_accel;
    else if (shape_others.empty() && !shape_meshes.empty() && shape_instances.empty())
        accel.handle = meshes_accel;
    else {
        uint32_t instances_count = !shape_meshes.empty() + !shape_others.empty() + shape_instances.size();
        std::vector<OptixInstance> instances(instances_count);

        uint32_t instance_id = 0;
        unsigned int sbt_offset = base_sbt_offset;
        if (!shape_meshes.empty()) {
            instances[instance_id] = {
                { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 }, // transform
                instance_id,                            // instanceId
                sbt_offset,                             // sbtOffset
                255,                                    // visibilityMask
                OPTIX_INSTANCE_FLAG_DISABLE_TRANSFORM,  // flags
                meshes_accel,                           // handle
                { 0, 0 }                                // pad
            };
            ++instance_id;
            sbt_offset += (unsigned int) shape_meshes.size();
        }
        if (!shape_others.empty()) {
            instances[instance_id] = {
                { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 }, // transform
                instance_id,                            // instanceId
                sbt_offset,                             // sbtOffset
                255,                                    // visibilityMask
                OPTIX_INSTANCE_FLAG_DISABLE_TRANSFORM,  // flags
                others_accel,                           // handle
                { 0, 0 }                                // pad
            };
            ++instance_id;
        }
        // handle instances
        for(Shape* instance: shape_instances) {
            instance->optix_prepare_instance(context, instances[instance_id], instance_id);
            ++instance_id;
        }

        void* d_instances = cuda_malloc(instances.size() * sizeof(OptixInstance));
        cuda_memcpy_to_device(d_instances, instances.data(), instances.size() * sizeof(OptixInstance));

        OptixBuildInput build_input;
        build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        build_input.instanceArray.instances = (CUdeviceptr) d_instances;
        build_input.instanceArray.numInstances = instances.size();
        build_input.instanceArray.aabbs = 0;
        build_input.instanceArray.numAabbs = 0;

        OptixAccelBufferSizes buffer_sizes;
        rt_check(optixAccelComputeMemoryUsage(context, &accel_options, &build_input, 1, &buffer_sizes));
        void* d_temp_buffer = cuda_malloc(buffer_sizes.tempSizeInBytes);
        accel.buffer_ias    = cuda_malloc(buffer_sizes.outputSizeInBytes);

        rt_check(optixAccelBuild(
            context,
            0,              // CUDA stream
            &accel_options,
            &build_input,
            1,              // num build inputs
            (CUdeviceptr)d_temp_buffer,
            buffer_sizes.tempSizeInBytes,
            (CUdeviceptr)accel.buffer_ias,
            buffer_sizes.outputSizeInBytes,
            &accel.handle,
            0,  // emitted property list
            0   // num emitted properties
        ));

        cuda_free((void*)d_temp_buffer);
        cuda_free((void*)d_instances); // TODO: check if we can free this now...
    }
}

NAMESPACE_END(mitsuba)
#endif
