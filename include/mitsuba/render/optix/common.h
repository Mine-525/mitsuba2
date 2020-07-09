#pragma once

#ifdef __CUDACC__
# include <optix.h>
#else
# include <mitsuba/render/optix_api.h>
#endif

#include <mitsuba/render/optix/bbox.cuh>
#include <mitsuba/render/optix/matrix.cuh>

struct OptixHitGroupData {
    unsigned long long shape_ptr;
    void* data;
};

template <typename T>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) EmptySbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

using RayGenSbtRecord   = EmptySbtRecord;
using MissSbtRecord     = EmptySbtRecord;
using HitGroupSbtRecord = SbtRecord<OptixHitGroupData>;

/// Launch-varying parameters
struct OptixParams {
    bool    *in_mask;
    float   *in_ox, *in_oy, *in_oz,
            *in_dx, *in_dy, *in_dz,
            *in_mint, *in_maxt;
    float   *out_t, *out_u, *out_v,
            *out_ng_x, *out_ng_y, *out_ng_z,
            *out_ns_x, *out_ns_y, *out_ns_z,
            *out_p_x, *out_p_y, *out_p_z,
            *out_dp_du_x, *out_dp_du_y, *out_dp_du_z,
            *out_dp_dv_x, *out_dp_dv_y, *out_dp_dv_z;

    unsigned long long *out_shape_ptr;
    unsigned int *out_primitive_id;
    unsigned int *out_instance_id;

    bool *out_hit;

    OptixTraversableHandle handle;
    unsigned int max_instance_id;
};

#ifdef __CUDACC__
extern "C" {
    __constant__ OptixParams params;
}

using namespace optix;

__device__ void write_output_params(OptixParams &params,
                                    unsigned int launch_index,
                                    unsigned long long shape_ptr,
                                    unsigned int prim_id,
                                    Vector3f p,
                                    Vector2f uv,
                                    Vector3f ns,
                                    Vector3f ng,
                                    Vector3f dp_du,
                                    Vector3f dp_dv,
                                    float t) {
    unsigned int instance_index = optixGetInstanceId();
    // Transform inputs if the object is inside an instance
    if (instance_index < params.max_instance_id) {
        float m[12], inv[12];
        optixGetObjectToWorldTransformMatrix(m);
        optixGetWorldToObjectTransformMatrix(inv);
        Transform4f to_world(m, inv);

        p = to_world.transform_point(p);
        ns = normalize(to_world.transform_normal(ns));
        ng = normalize(to_world.transform_normal(ng));
        dp_du = to_world.transform_vector(dp_du);
        dp_dv = to_world.transform_vector(dp_dv);
    }

    params.out_shape_ptr[launch_index]    = shape_ptr;
    params.out_primitive_id[launch_index] = prim_id;
    params.out_instance_id[launch_index]  = instance_index;

    params.out_u[launch_index] = uv.x();
    params.out_v[launch_index] = uv.y();

    params.out_ng_x[launch_index] = ng.x();
    params.out_ng_y[launch_index] = ng.y();
    params.out_ng_z[launch_index] = ng.z();

    params.out_ns_x[launch_index] = ns.x();
    params.out_ns_y[launch_index] = ns.y();
    params.out_ns_z[launch_index] = ns.z();

    params.out_p_x[launch_index] = p.x();
    params.out_p_y[launch_index] = p.y();
    params.out_p_z[launch_index] = p.z();

    params.out_dp_du_x[launch_index] = dp_du.x();
    params.out_dp_du_y[launch_index] = dp_du.y();
    params.out_dp_du_z[launch_index] = dp_du.z();

    params.out_dp_dv_x[launch_index] = dp_dv.x();
    params.out_dp_dv_y[launch_index] = dp_dv.y();
    params.out_dp_dv_z[launch_index] = dp_dv.z();

    params.out_t[launch_index] = t;
}

/// Returns a unique launch index per ray
__device__ unsigned int calculate_launch_index() {
    uint3 launch_dims = optixGetLaunchDimensions();
    uint3 launch_index3 = optixGetLaunchIndex();
    return launch_index3.x + (launch_index3.y + launch_index3.z * launch_dims.y) * launch_dims.x;
}
#endif