#include "shapegroup.h"

#include <mitsuba/core/properties.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/render/optix_api.h>

NAMESPACE_BEGIN(mitsuba)

MTS_VARIANT ShapeGroup<Float, Spectrum>::ShapeGroup(const Properties &props) {
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
#if defined(MTS_ENABLE_EMBREE) || defined(MTS_ENABLE_OPTIX)
                m_shapes.push_back(shape);
                m_bbox.expand(shape->bbox());
#endif
#if !defined(MTS_ENABLE_EMBREE)
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
}

MTS_VARIANT ShapeGroup<Float, Spectrum>::~ShapeGroup() {
#if defined(MTS_ENABLE_EMBREE)
    if constexpr (!is_cuda_array_v<Float>)
        rtcReleaseScene(m_embree_scene);
#endif
}

MTS_VARIANT typename ShapeGroup<Float, Spectrum>::ScalarSize
ShapeGroup<Float, Spectrum>::primitive_count() const {
#if defined(MTS_ENABLE_EMBREE)
    ScalarSize count = 0;
    for (auto shape : m_shapes)
        count += shape->primitive_count();

    return count;
#else
    return m_kdtree->primitive_count();
#endif
}

MTS_VARIANT std::string
ShapeGroup<Float, Spectrum>::to_string() const {
    std::ostringstream oss;
        oss << "ShapeGroup[" << std::endl
            << "  name = \"" << m_id << "\"," << std::endl
            << "  prim_count = " << primitive_count() << std::endl
            << "]";
    return oss.str();
}

#if defined(MTS_ENABLE_OPTIX)
MTS_VARIANT void
ShapeGroup<Float, Spectrum>::optix_fill_hitgroup_records(std::vector<HitGroupSbtRecord> &hitgroup_records,
                                                         OptixProgramGroup *program_groups) {
    m_sbt_offset = hitgroup_records.size();
    std::vector<ref<Base>> dummy_shapegroups;
    fill_hitgroup_records(hitgroup_records, m_shapes, dummy_shapegroups, program_groups);
}
#endif

MTS_EXPORT_PLUGIN(ShapeGroup, "Grouped geometry for instancing")
NAMESPACE_END(mitsuba)
