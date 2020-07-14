#include "librender_ptx.h"
#include <iomanip>

#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/mmap.h>
#include <mitsuba/render/optix/common.h>
#include <mitsuba/render/optix/shapes.h>
#include <mitsuba/render/optix_api.h>

NAMESPACE_BEGIN(mitsuba)

#if !defined(NDEBUG)
# define MTS_OPTIX_DEBUG 1
#endif

#ifdef __WINDOWS__
# define strdup _strdup
#endif

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */) {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag
              << "]: " << message << "\n";
}

#if defined(MTS_OPTIX_DEBUG)
    static constexpr size_t ProgramGroupCount = 4 + custom_optix_shapes_count;
#else
    static constexpr size_t ProgramGroupCount = 3 + custom_optix_shapes_count;
#endif

static constexpr const char* ptx_file_path = "/home/benoit/Desktop/cpp/mitsuba2_instancing/resources/ptx/optix_rt.ptx";

struct OptixState {
    OptixDeviceContext context;
    OptixPipeline pipeline = nullptr;
    OptixModule module = nullptr;
    OptixProgramGroup program_groups[ProgramGroupCount];
    OptixShaderBindingTable sbt = {};
    OptixAccelData accel;
    OptixTraversableHandle ias_handle = 0ull;
    void* ias_buffer = nullptr;

    void* params;
    char *custom_optix_shapes_program_names[2 * custom_optix_shapes_count];

    enoki::CUDAArray<const void*> shapes_ptr;
};

MTS_VARIANT void Scene<Float, Spectrum>::accel_init_gpu(const Properties &/*props*/) {
    if constexpr (is_cuda_array_v<Float>) {
        Log(Info, "Building scene in OptiX ..");
        m_accel = new OptixState();
        OptixState &s = *(OptixState *) m_accel;

        // Copy shapes pointers to the GPU
        s.shapes_ptr = ShapePtr::copy((void**)m_shapes.data(), m_shapes.size());

        // ------------------------
        //  OptiX context creation
        // ------------------------

        CUcontext cuCtx = 0;  // zero means take the current context
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction       = &context_log_cb;
    #if !defined(MTS_OPTIX_DEBUG)
        options.logCallbackLevel          = 1;
    #else
        options.logCallbackLevel          = 3;
    #endif
        rt_check(optixDeviceContextCreate(cuCtx, &options, &s.context));

        // ----------------------------------------------
        //  Pipeline generation - Create Module from PTX
        // ----------------------------------------------

        OptixPipelineCompileOptions pipeline_compile_options = {};
        OptixModuleCompileOptions module_compile_options = {};

        module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    #if !defined(MTS_OPTIX_DEBUG)
        module_compile_options.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        module_compile_options.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    #else
        module_compile_options.optLevel         = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        module_compile_options.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    #endif

        pipeline_compile_options.usesMotionBlur        = false;
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
        pipeline_compile_options.numPayloadValues      = 3;
        pipeline_compile_options.numAttributeValues    = 3;
        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    #if !defined(MTS_OPTIX_DEBUG)
        pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    #else
        pipeline_compile_options.exceptionFlags =
                OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW
                | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH
                | OPTIX_EXCEPTION_FLAG_USER
                | OPTIX_EXCEPTION_FLAG_DEBUG;
    #endif

        auto fs = Thread::thread()->file_resolver();
        fs::path file_path = fs->resolve(ptx_file_path);

        if (!fs::exists(file_path))
            Throw("ptx file not found \"%s\" ..", file_path);

        ref<MemoryMappedFile> mmap = new MemoryMappedFile(file_path);

        rt_check_log(optixModuleCreateFromPTX(
            s.context,
            &module_compile_options,
            &pipeline_compile_options,
            (const char *)mmap->data(),
            mmap->size(),
            optix_log_buffer,
            &optix_log_buffer_size,
            &s.module
        ));

        // ---------------------------------------------
        //  Pipeline generation - Create program groups
        // ---------------------------------------------

        OptixProgramGroupOptions program_group_options = {};

        OptixProgramGroupDesc prog_group_descs[ProgramGroupCount];
        memset(prog_group_descs, 0, sizeof(prog_group_descs));

        prog_group_descs[0].kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        prog_group_descs[0].raygen.module            = s.module;
        prog_group_descs[0].raygen.entryFunctionName = "__raygen__rg";

        prog_group_descs[1].kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        prog_group_descs[1].miss.module            = s.module;
        prog_group_descs[1].miss.entryFunctionName = "__miss__ms";

        prog_group_descs[2].kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        prog_group_descs[2].hitgroup.moduleCH            = s.module;
        prog_group_descs[2].hitgroup.entryFunctionNameCH = "__closesthit__mesh";

        for (size_t i = 0; i < custom_optix_shapes_count; i++) {
            prog_group_descs[3+i].kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

            std::string name = string::to_lower(custom_optix_shapes[i]);
            s.custom_optix_shapes_program_names[2*i] = strdup(("__closesthit__" + name).c_str());
            s.custom_optix_shapes_program_names[2*i+1] = strdup(("__intersection__" + name).c_str());

            prog_group_descs[3+i].hitgroup.moduleCH            = s.module;
            prog_group_descs[3+i].hitgroup.entryFunctionNameCH = s.custom_optix_shapes_program_names[2*i];
            prog_group_descs[3+i].hitgroup.moduleIS            = s.module;
            prog_group_descs[3+i].hitgroup.entryFunctionNameIS = s.custom_optix_shapes_program_names[2*i+1];
        }

    #if defined(MTS_OPTIX_DEBUG)
        OptixProgramGroupDesc &exception_prog_group_desc = prog_group_descs[ProgramGroupCount-1];
        exception_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
        exception_prog_group_desc.hitgroup.moduleCH            = s.module;
        exception_prog_group_desc.hitgroup.entryFunctionNameCH = "__exception__err";
    #endif

        rt_check_log(optixProgramGroupCreate(
            s.context,
            prog_group_descs,
            ProgramGroupCount,
            &program_group_options,
            optix_log_buffer,
            &optix_log_buffer_size,
            s.program_groups
        ));

        // ---------------------------------------
        //  Pipeline generation - Create pipeline
        // ---------------------------------------

        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth          = 1;
    #if defined(MTS_OPTIX_DEBUG)
        pipeline_link_options.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    #else
        pipeline_link_options.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    #endif
        pipeline_link_options.overrideUsesMotionBlur = false;
        rt_check_log(optixPipelineCreate(
            s.context,
            &pipeline_compile_options,
            &pipeline_link_options,
            s.program_groups,
            ProgramGroupCount,
            optix_log_buffer,
            &optix_log_buffer_size,
            &s.pipeline
        ));

        // ---------------------------------
        //  Shader Binding Table generation
        // ---------------------------------
        std::vector<HitGroupSbtRecord> hg_sbts;
        fill_hitgroup_records(m_shapes, m_shapegroups, hg_sbts, s.program_groups);

        size_t shapes_count = hg_sbts.size();
        void* records = cuda_malloc(sizeof(RayGenSbtRecord) + sizeof(MissSbtRecord) + sizeof(HitGroupSbtRecord) * shapes_count);

        RayGenSbtRecord raygen_sbt;
        rt_check(optixSbtRecordPackHeader(s.program_groups[0], &raygen_sbt));
        void* raygen_record = records;
        cuda_memcpy_to_device(raygen_record, &raygen_sbt, sizeof(RayGenSbtRecord));

        MissSbtRecord miss_sbt;
        rt_check(optixSbtRecordPackHeader(s.program_groups[1], &miss_sbt));
        void* miss_record = (char*)records + sizeof(RayGenSbtRecord);
        cuda_memcpy_to_device(miss_record, &miss_sbt, sizeof(MissSbtRecord));

        // Allocate hitgroup records array
        void* hitgroup_records = (char*)records + sizeof(RayGenSbtRecord) + sizeof(MissSbtRecord);

        // Copy HitGroupRecords to the GPU
        cuda_memcpy_to_device(hitgroup_records, hg_sbts.data(), shapes_count * sizeof(HitGroupSbtRecord));

        s.sbt.raygenRecord                = (CUdeviceptr)raygen_record;
        s.sbt.missRecordBase              = (CUdeviceptr)miss_record;
        s.sbt.missRecordStrideInBytes     = sizeof(MissSbtRecord);
        s.sbt.missRecordCount             = 1;
        s.sbt.hitgroupRecordBase          = (CUdeviceptr)hitgroup_records;
        s.sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
        s.sbt.hitgroupRecordCount         = (unsigned int) shapes_count;

        // --------------------------------------
        //  Acceleration data structure building
        // --------------------------------------

        accel_parameters_changed_gpu();

        // Allocate params pointer
        s.params = cuda_malloc(sizeof(OptixParams));

        // This will trigger the scatter calls to upload geometry to the device
        cuda_eval();

        // TODO: check if we still want to do run a dummy launch
    }
}

MTS_VARIANT void Scene<Float, Spectrum>::accel_parameters_changed_gpu() {
    if constexpr (is_cuda_array_v<Float>) {
        if (m_shapes.empty())
            return;
        OptixState &s = *(OptixState *) m_accel;
        build_gas(s.context, m_shapes, m_shapegroups, s.accel);

        std::vector<OptixInstance> instances;
        create_instances(s.context, m_shapes, 0, s.accel, m_shapes.size(), ScalarTransform4f(), instances);

        if (instances.size() == 1) {
            s.ias_buffer = nullptr;
            s.ias_handle = instances[0].traversableHandle;
            return;
        }
        
        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;
        accel_options.motionOptions.numKeys = 0;
        
        void* d_instances = cuda_malloc(instances.size() * sizeof(OptixInstance));
        cuda_memcpy_to_device(d_instances, instances.data(), instances.size() * sizeof(OptixInstance));

        OptixBuildInput build_input;
        build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        build_input.instanceArray.instances = (CUdeviceptr) d_instances;
        build_input.instanceArray.numInstances = instances.size();
        build_input.instanceArray.aabbs = 0;
        build_input.instanceArray.numAabbs = 0;

        OptixAccelBufferSizes buffer_sizes;
        rt_check(optixAccelComputeMemoryUsage(s.context, &accel_options, &build_input, 1, &buffer_sizes));
        void* d_temp_buffer = cuda_malloc(buffer_sizes.tempSizeInBytes);
        s.ias_buffer    = cuda_malloc(buffer_sizes.outputSizeInBytes);

        rt_check(optixAccelBuild(
            s.context,
            0,              // CUDA stream
            &accel_options,
            &build_input,
            1,              // num build inputs
            (CUdeviceptr)d_temp_buffer,
            buffer_sizes.tempSizeInBytes,
            (CUdeviceptr)s.ias_buffer,
            buffer_sizes.outputSizeInBytes,
            &s.ias_handle,
            0,  // emitted property list
            0   // num emitted properties
        ));

        cuda_free(d_temp_buffer);
        cuda_free(d_instances);
    }
}

MTS_VARIANT void Scene<Float, Spectrum>::accel_release_gpu() {
    if constexpr (is_cuda_array_v<Float>) {
        OptixState &s = *(OptixState *) m_accel;
        cuda_free((void*)s.sbt.raygenRecord);
        cuda_free((void*)s.params);
        cuda_free(s.ias_buffer);
        rt_check(optixPipelineDestroy(s.pipeline));
        for (size_t i = 0; i < ProgramGroupCount; i++)
            rt_check(optixProgramGroupDestroy(s.program_groups[i]));
        for (size_t i = 0; i < 2 * custom_optix_shapes_count; i++)
            free(s.custom_optix_shapes_program_names[i]);
        rt_check(optixModuleDestroy(s.module));
        rt_check(optixDeviceContextDestroy(s.context));

        delete (OptixState *) m_accel;
        m_accel = nullptr;
    }
}

MTS_VARIANT typename Scene<Float, Spectrum>::SurfaceInteraction3f
Scene<Float, Spectrum>::ray_intersect_gpu(const Ray3f &ray_, Mask active) const {
    if constexpr (is_cuda_array_v<Float>) {
        Assert(!m_shapes.empty());
        OptixState &s = *(OptixState *) m_accel;
        Ray3f ray(ray_);
        size_t ray_count = std::max(slices(ray.o), slices(ray.d));
        set_slices(ray, ray_count);
        set_slices(active, ray_count);

        SurfaceInteraction3f si = empty<SurfaceInteraction3f>(ray_count);

        // DEBUG mode: Explicitly instantiate `si` with NaN values.
        // As the integrator should only deal with the lanes of `si` for which
        // `si.is_valid()==true`, this makes it easier to catch bugs in the
        // masking logic implemented in the integrator.
#if !defined(NDEBUG)
            #define SET_NAN(name) name = full<decltype(name)>(std::numeric_limits<scalar_t<Float>>::quiet_NaN(), ray_count);
            SET_NAN(si.t); SET_NAN(si.time); SET_NAN(si.p); SET_NAN(si.uv); SET_NAN(si.n);
            SET_NAN(si.sh_frame.n); SET_NAN(si.dp_du); SET_NAN(si.dp_dv);
            #undef SET_NAN
#endif  // !defined(NDEBUG)

        cuda_eval();

        UInt32 instance_index = full<UInt32>((unsigned int)m_shapes.size(), ray_count);

        const OptixParams params = {
            // Active mask
            active.data(),
            // In: ray origin
            ray.o.x().data(), ray.o.y().data(), ray.o.z().data(),
            // In: ray direction
            ray.d.x().data(), ray.d.y().data(), ray.d.z().data(),
            // In: ray extents
            ray.mint.data(), ray.maxt.data(),
            // Out: Distance along ray
            si.t.data(),
            // Out: UV coordinates
            si.uv.x().data(), si.uv.y().data(),
            // Out: Geometric normal
            si.n.x().data(), si.n.y().data(), si.n.z().data(),
            // Out: Shading normal
            si.sh_frame.n.x().data(), si.sh_frame.n.y().data(), si.sh_frame.n.z().data(),
            // Out: Intersection position
            si.p.x().data(), si.p.y().data(), si.p.z().data(),
            // Out: Texture space derivative (U)
            si.dp_du.x().data(), si.dp_du.y().data(), si.dp_du.z().data(),
            // Out: Texture space derivative (V)
            si.dp_dv.x().data(), si.dp_dv.y().data(), si.dp_dv.z().data(),
            // Out: Shape pointer (on host)
            (unsigned long long*)si.shape.data(),
            // Out: Primitive index
            si.prim_index.data(),
            // Out: Instance index
            instance_index.data(),
            // Out: Hit flag
            nullptr,
            // top_object
            s.ias_handle,
            // max instance id
            (unsigned int)m_shapes.size()
        };

        cuda_memcpy_to_device(s.params, &params, sizeof(OptixParams));

        // Try to make width and height close to sqrt(ray_count) (optixLaunch doesn't
        // seem to like very big dimensions)
        unsigned int width = 1, height = (unsigned int) ray_count;
        while (!(height & 1) && width < height) {
            width <<= 1;
            height >>= 1;
        }

        OptixResult rt = optixLaunch(
            s.pipeline,
            0, // default cuda stream
            (CUdeviceptr)s.params, sizeof(OptixParams),
            &s.sbt,
            width, height, /* depth = */ 1u
        );
        if (rt == OPTIX_ERROR_HOST_OUT_OF_MEMORY) {
            cuda_malloc_trim();
            rt = optixLaunch(
                s.pipeline,
                0, // default cuda stream
                (CUdeviceptr)s.params, sizeof(OptixParams),
                &s.sbt,
                width, height, /* depth = */ 1u
            );
        }
        rt_check(rt);

        si.time = ray.time;
        si.wavelengths = ray.wavelengths;
        si.duv_dx = si.duv_dy = 0.f;

        Mask valid_instances = instance_index < m_shapes.size();
        si.instance = gather<ShapePtr>(reinterpret_array<ShapePtr>(s.shapes_ptr), instance_index, active & valid_instances);

        // Gram-schmidt orthogonalization to compute local shading frame
        si.sh_frame.s = normalize(
            fnmadd(si.sh_frame.n, dot(si.sh_frame.n, si.dp_du), si.dp_du));
        si.sh_frame.t = cross(si.sh_frame.n, si.sh_frame.s);

        // Incident direction in local coordinates
        si.wi = select(si.is_valid(), si.to_local(-ray.d), -ray.d);

        return si;
    } else {
        ENOKI_MARK_USED(ray_);
        ENOKI_MARK_USED(active);
        Throw("ray_intersect_gpu() should only be called in GPU mode.");
    }
}

MTS_VARIANT typename Scene<Float, Spectrum>::Mask
Scene<Float, Spectrum>::ray_test_gpu(const Ray3f &ray_, Mask active) const {
    if constexpr (is_cuda_array_v<Float>) {
        OptixState &s = *(OptixState *) m_accel;
        Ray3f ray(ray_);
        size_t ray_count = std::max(slices(ray.o), slices(ray.d));
        Mask hit = empty<Mask>(ray_count);

        set_slices(ray, ray_count);
        set_slices(active, ray_count);

        cuda_eval();

        const OptixParams params = {
            // Active mask
            active.data(),
            // In: ray origin
            ray.o.x().data(), ray.o.y().data(), ray.o.z().data(),
            // In: ray direction
            ray.d.x().data(), ray.d.y().data(), ray.d.z().data(),
            // In: ray extents
            ray.mint.data(), ray.maxt.data(),
            // Out: Distance along ray
            nullptr,
            // Out: UV coordinates
            nullptr, nullptr,
            // Out: Geometric normal
            nullptr, nullptr, nullptr,
            // Out: Shading normal
            nullptr, nullptr, nullptr,
            // Out: Intersection position
            nullptr, nullptr, nullptr,
            // Out: Texture space derivative (U)
            nullptr, nullptr, nullptr,
            // Out: Texture space derivative (V)
            nullptr, nullptr, nullptr,
            // Out: Shape pointer (on host)
            nullptr,
            // Out: Primitive index
            nullptr,
            // Out: Instance index
            nullptr,
            // Out: Hit flag
            hit.data(),
            // top_object
            s.ias_handle,
            // max instance id
            (unsigned int)m_shapes.size()
        };

        cuda_memcpy_to_device(s.params, &params, sizeof(OptixParams));

        unsigned int width = 1, height = (unsigned int) ray_count;
        while (!(height & 1) && width < height) {
            width <<= 1;
            height >>= 1;
        }

        OptixResult rt = optixLaunch(
            s.pipeline,
            0, // default cuda stream
            (CUdeviceptr)s.params,
            sizeof(OptixParams),
            &s.sbt,
            width,
            height,
            1u // depth
        );
        if (rt == OPTIX_ERROR_HOST_OUT_OF_MEMORY) {
            cuda_malloc_trim();
            rt = optixLaunch(
                s.pipeline,
                0, // default cuda stream
                (CUdeviceptr)s.params,
                sizeof(OptixParams),
                &s.sbt,
                width,
                height,
                1u // depth
            );
        }
        rt_check(rt);

        return hit;
    } else {
        ENOKI_MARK_USED(ray_);
        ENOKI_MARK_USED(active);
        Throw("ray_test_gpu() should only be called in GPU mode.");
    }
}

NAMESPACE_END(msiuba)
