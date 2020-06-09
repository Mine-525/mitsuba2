#include <mitsuba/core/fwd.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/render/kdtree.h>

#if defined(MTS_ENABLE_EMBREE)
    #include <embree3/rtcore.h>
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
    MTS_IMPORT_BASE(Shape, is_emitter, is_sensor, m_id, m_shapegroup)
    MTS_IMPORT_TYPES(ShapeKDTree)

    using typename Base::ScalarSize;

    ShapeGroup(const Properties &props) {
        m_id = props.id();

#if !defined(MTS_ENABLE_EMBREE)
        m_kdtree = new ShapeKDTree(props);
#endif

        // Add children to the underlying datastructure
        for (auto &kv : props.objects()) {
            const Class *c_class = kv.second->class_();
            if (c_class->name() == "Instance") {
                Throw("Nested instancing is not permitted");
            } else if (c_class->derives_from(MTS_CLASS(Base))) {
                Base *shape = static_cast<Base *>(kv.second.get());
                if (shape->is_shapegroup())
                    Throw("Nested ShapeGroup is not permitted");
                if (shape->is_emitter())
                    Throw("Instancing of emitters is not supported");
                if (shape->is_sensor())
                    Throw("Instancing of sensors is not supported");
                else {
#if defined(MTS_ENABLE_EMBREE)
                    m_shapes.push_back(shape);
                    m_bbox.expand(shape->bbox());
#else
                    m_kdtree->add_shape(shape);
#endif
                }
            } else {
                Throw("Tried to add an unsupported object of type \"%s\"", kv.second);
            }
        }

#if !defined(MTS_ENABLE_EMBREE)
        if (!m_kdtree->ready())
            m_kdtree->build();

        m_bbox = m_kdtree->bbox();
#endif

        m_shapegroup = true;
    }

#if defined(MTS_ENABLE_EMBREE)
    void init_embree_scene(RTCDevice device) override {
        if constexpr (!is_cuda_array_v<Float>) {
            // Construct the BVH only once
            if(m_scene == nullptr) {
                m_scene = rtcNewScene(device);
                for (auto shape : m_shapes)
                    rtcAttachGeometry(m_scene, shape->embree_geometry(device));
                rtcCommitScene(m_scene);
            }
        }
    }

    void release_embree_scene() override {
        if constexpr (!is_cuda_array_v<Float>) {
            rtcReleaseScene(m_scene);
        }
    }

    RTCGeometry embree_geometry(RTCDevice device) const override {
        if constexpr (!is_cuda_array_v<Float>) {
            RTCGeometry instance = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_INSTANCE);
            if(m_scene == nullptr)
                Throw("Embree scene not initialized, call init_embree_scene() first");
            rtcSetGeometryInstancedScene(instance, m_scene);
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

    ScalarSize primitive_count() const override {
#if defined(MTS_ENABLE_EMBREE)
        ScalarSize count = 0;
        for (auto shape : m_shapes)
            count += shape->primitive_count();

        return count;
#else
        return m_kdtree->primitive_count();
#endif
    }

    ScalarBoundingBox3f bbox() const override{ return m_bbox; }

    ScalarFloat surface_area() const override { return 0.f; }

    MTS_INLINE ScalarSize effective_primitive_count() const override { return 0; }

    std::string to_string() const override {
        std::ostringstream oss;
            oss << "ShapeGroup[" << std::endl
                << "  name = \"" << m_id << "\"," << std::endl
                << "  prim_count = " << primitive_count() << std::endl
                << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    ScalarBoundingBox3f m_bbox;

    #if defined(MTS_ENABLE_EMBREE)
        RTCScene m_scene = nullptr;
        std::vector<ref<Base>> m_shapes;
    #else
        ref<ShapeKDTree> m_kdtree;
    #endif
};

MTS_IMPLEMENT_CLASS_VARIANT(ShapeGroup, Shape)
MTS_EXPORT_PLUGIN(ShapeGroup, "Grouped geometry for instancing")
NAMESPACE_END(mitsuba)
