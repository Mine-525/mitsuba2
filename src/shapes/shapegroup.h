#include <mitsuba/core/fwd.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/kdtree.h>

#if defined(MTS_ENABLE_EMBREE)
    #include <embree3/rtcore.h>
#endif
#if defined(MTS_ENABLE_OPTIX)
    #include <mitsuba/render/optix/shapes.h>
#endif

NAMESPACE_BEGIN(mitsuba)


/**!

.. _shape-shapegroup:

Shape group (:monosp:`shapegroup`)
----------------------------------

.. pluginparameters::

 * - (Nested plugin)
   - :paramtype:`shape`
   - One or more shapes that should be made available for geometry instancing

This plugin implements a container for shapes that should be made available for geometry instancing.
Any shapes placed in a shapegroup will not be visible on their own—instead, the renderer will
precompute ray intersection acceleration data structures so that they can efficiently be referenced
many times using the :ref:`shape-instance` plugin. This is useful for rendering things like forests,
where only a few distinct types of trees have to be kept in memory. An example is given below:

.. code-block:: xml

    <!-- Declare a named shape group containing two objects -->
    <shape type="shapegroup" id="my_shape_group">
        <shape type="ply">
            <string name="filename" value="data.ply"/>
            <bsdf type="roughconductor"/>
        </shape>
        <shape type="sphere">
            <transform name="to_world">
                <scale value="5"/>
                <translate y="20"/>
            </transform>
            <bsdf type="diffuse"/>
        </shape>
    </shape>

    <!-- Instantiate the shape group without any kind of transformation -->
    <shape type="instance">
        <ref id="my_shape_group"/>
    </shape>

    <!-- Create instance of the shape group, but rotated, scaled, and translated -->
    <shape type="instance">
        <ref id="my_shape_group"/>
        <transform name="to_world">
            <rotate x="1" angle="45"/>
            <scale value="1.5"/>
            <translate z="10"/>
        </transform>
    </shape>

 */

template <typename Float, typename Spectrum>
class ShapeGroup final: public Shape<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Shape, is_emitter, is_sensor, m_id)
    MTS_IMPORT_TYPES(ShapeKDTree)

    using typename Base::ScalarSize;

    ShapeGroup(const Properties &props);

    ~ShapeGroup();

#if defined(MTS_ENABLE_EMBREE)
    RTCGeometry embree_geometry(RTCDevice device) override {
        if constexpr (!is_cuda_array_v<Float>) {
            // Construct the BVH only once
            if (m_embree_scene == nullptr) {
                m_embree_scene = rtcNewScene(device);
                for (auto shape : m_shapes)
                    rtcAttachGeometry(m_embree_scene, shape->embree_geometry(device));
                rtcCommitScene(m_embree_scene);
            }

            RTCGeometry instance = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_INSTANCE);
            rtcSetGeometryInstancedScene(instance, m_embree_scene);
            return instance;
        } else {
            Throw("embree_geometry() should only be called in CPU mode.");
        }
    }
#else
    std::pair<Mask, Float> ray_intersect(const Ray3f &ray, Float *cache,
                                         Mask active) const override {
        MTS_MASK_ARGUMENT(active);
        return m_kdtree->template ray_intersect<false>(ray, cache, active);
    }
    Mask ray_test(const Ray3f &ray, Mask active) const override {
        MTS_MASK_ARGUMENT(active);
        return m_kdtree->template ray_intersect<true>(ray, (Float* ) nullptr, active).first;
    }
#endif

    void fill_surface_interaction(const Ray3f &ray, const Float *cache,
                                  SurfaceInteraction3f &si_out, Mask active) const override {
        MTS_MASK_ARGUMENT(active);
    #if defined(MTS_ENABLE_EMBREE)
        SurfaceInteraction3f si(si_out);

        // Extract intersected shape from cache
        if constexpr (!is_array_v<Float>) {
            size_t shape_index = cache[2];
            Assert(shape_index < m_shapes.size());
            si.shape = m_shapes[shape_index];
        } else {
            using ShapePtr = replace_scalar_t<Float, const Base *>;
            UInt32 shape_index = cache[2];
            Assert(shape_index < m_shapes.size());
            si.shape = gather<ShapePtr>(m_shapes.data(), shape_index, active);
        }

        Float extracted_cache[2] = { cache[0], cache[1] };
        si.shape->fill_surface_interaction(ray, extracted_cache, si, active);
        masked(si_out, active) = si;
    #else
        masked(si_out, active) = m_kdtree->create_surface_interaction(ray, si_out.t, cache, active);
    #endif
    }

    ScalarSize primitive_count() const override;

    ScalarBoundingBox3f bbox() const override{ return m_bbox; }

    ScalarFloat surface_area() const override { return 0.f; }

    MTS_INLINE ScalarSize effective_primitive_count() const override { return 0; }

    std::string to_string() const override;

#if defined(MTS_ENABLE_OPTIX)
    void optix_accel_handle(const OptixDeviceContext& context,
                            OptixInstance& instance) {
        if (m_accel.handle == 0ull)
            build_gas(m_shapes, context, m_sbt_offset, m_accel, instance.instanceId);
        instance.traversableHandle = m_accel.handle;
        instance.sbtOffset = m_sbt_offset;
    }

    virtual void optix_fill_hitgroup_records(std::vector<HitGroupSbtRecord> &hitgroup_records,
                                             OptixProgramGroup *program_groups) override;
#endif

    MTS_DECLARE_CLASS()
private:
    ScalarBoundingBox3f m_bbox;

#if defined(MTS_ENABLE_EMBREE)
    std::vector<ref<Base>> m_shapes;
    RTCScene m_embree_scene = nullptr;
#else
    ref<ShapeKDTree> m_kdtree;
#endif

#if defined(MTS_ENABLE_OPTIX)
    #if !defined(MTS_ENABLE_EMBREE)
        std::vector<ref<Base>> m_shapes;
    #endif
    OptixAccelData m_accel;
    /// OptiX hitgroup sbt offset
    uint32_t m_sbt_offset;
#endif
};

MTS_IMPLEMENT_CLASS_VARIANT(ShapeGroup, Shape)
NAMESPACE_END(mitsuba)
