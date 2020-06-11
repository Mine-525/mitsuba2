#include <mitsuba/core/fwd.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/bsdf.h>

#if defined(MTS_ENABLE_EMBREE)
    #include <embree3/rtcore.h>
    #include <enoki/transform.h>
#endif

NAMESPACE_BEGIN(mitsuba)

/**!

.. _shape-instance:

Instance (:monosp:`instance`)
-------------------------------------------------

.. pluginparameters::

 * - (Nested plugin)
   - :paramtype:`shapegroup`
   - A reference to a shape group that should be instantiated.
 * - to_world
   - |transform|
   - Specifies a linear object-to-world transformation. (Default: none (i.e. object space = world space))

This plugin implements a geometry instance used to efficiently replicate geometry many times. For
details on how to create instances, refer to the :ref:`shape-shapegroup` plugin.

.. warning::

    - Note that it is not possible to assign a different material to each instance — the material
      assignment specified within the shape group is the one that matters.
    - Shape groups cannot be used to replicate shapes with attached emitters, sensors, or
      subsurface scattering models.

 */

template <typename Float, typename Spectrum>
class Instance final: public Shape<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Shape, is_shapegroup, m_id, m_to_world, m_to_object, m_instance)
    MTS_IMPORT_TYPES(BSDF)

    using typename Base::ScalarSize;

    Instance(const Properties &props) {
        m_id = props.id();

        m_to_world = props.transform("to_world", ScalarTransform4f());
        m_to_object = m_to_world.inverse();

        for (auto &kv : props.objects()) {
            Base *shape = dynamic_cast<Base *>(kv.second.get());
            if (shape && shape->is_shapegroup()) {
                if (m_shapegroup)
                    Throw("Only a single shapegroup can be specified per instance.");
                m_shapegroup = shape;
            } else {
                    Throw("Only a shapegroup can be specified in an instance.");
            }
        }

        if (!m_shapegroup)
            Throw("A reference to a 'shapegroup' must be specified!");

        m_instance = true;
    }

    ScalarBoundingBox3f bbox() const override {
        const ScalarBoundingBox3f &bbox = m_shapegroup->bbox();

        // If the shape group is empty, return the invalid bbox
        if (!bbox.valid())
            return bbox;

        ScalarBoundingBox3f result;
        for (int i = 0; i < 8; ++i)
            result.expand(m_to_world * bbox.corner(i));
        return result;
    }

    ScalarSize primitive_count() const override { return 1; }

    ScalarSize effective_primitive_count() const override {
        return m_shapegroup->primitive_count();
    }

    //! @}
    // =============================================================

    // =============================================================
    //! @{ \name Ray tracing routines
    // =============================================================

   std::pair<Mask, Float> ray_intersect(const Ray3f &ray, Float *cache,
                                         Mask active) const override {
        MTS_MASK_ARGUMENT(active);
        return m_shapegroup->ray_intersect(m_to_object.transform_affine(ray), cache, active);
    }

    Mask ray_test(const Ray3f &ray, Mask active) const override {
        MTS_MASK_ARGUMENT(active);
        return m_shapegroup->ray_test(m_to_object.transform_affine(ray), active);
    }

    void fill_surface_interaction(const Ray3f &ray, const Float *cache,
                                  SurfaceInteraction3f &si_out, Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        SurfaceInteraction3f si(si_out);
        m_shapegroup->fill_surface_interaction(m_to_object.transform_affine(ray), cache, si, active);

        si.p = m_to_world.transform_affine(si.p);
        si.sh_frame.n = normalize(m_to_world.transform_affine(si.sh_frame.n));
        si.n = normalize(m_to_world.transform_affine(si.n));
        si.dp_du = m_to_world.transform_affine(si.dp_du);
        si.dp_dv = m_to_world.transform_affine(si.dp_dv);
        si.instance = this;

        masked(si_out, active) = si;
    }

    std::pair<Vector3f, Vector3f> normal_derivative(const SurfaceInteraction3f &si,
                                                    bool shading_frame,
                                                    Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        SurfaceInteraction3f temp(si);
        temp.p     = m_to_object.transform_affine(si.p);
        temp.dp_du = m_to_object.transform_affine(si.dp_du);
        temp.dp_dv = m_to_object.transform_affine(si.dp_dv);

        // Determine the length of the transformed normal before it was re-normalized
        Normal3f tn = m_to_world.transform_affine(
            normalize(m_to_object.transform_affine(si.sh_frame.n)));
        Float inv_len = rcp(norm(tn));
        tn *= inv_len;

        auto [dn_du, dn_dv] = si.shape->normal_derivative(temp, shading_frame, active);

        // Apply transform to dn_du and dn_dv
        dn_du = m_to_world.transform_affine(Normal3f(dn_du)) * inv_len;
        dn_dv = m_to_world.transform_affine(Normal3f(dn_dv)) * inv_len;

        dn_du -= tn * dot(tn, dn_du);
        dn_dv -= tn * dot(tn, dn_dv);

        return { dn_du, dn_dv };
    }

    //! @}
    // =============================================================

#if defined(MTS_ENABLE_EMBREE)
    RTCGeometry embree_geometry(RTCDevice device) const override {
        if constexpr (!is_cuda_array_v<Float>) {
            RTCGeometry instance = m_shapegroup->embree_geometry(device);
            rtcSetGeometryTimeStepCount(instance, 1);
            rtcSetGeometryTransform(instance, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, &m_to_world.matrix);
            rtcCommitGeometry(instance);
            return instance;
        } else {
            Throw("embree_geometry() should only be called in CPU mode.");
        }
    }
#endif

#if defined(MTS_ENABLE_OPTIX)
    virtual void optix_prepare_instance(const OptixDeviceContext& context, OptixInstance& instance, uint32_t instance_id) override {
        // TODO: check coeffs
        float transform[12] = {
            m_to_world.matrix(0, 0), m_to_world.matrix(0, 1), m_to_world.matrix(0, 2), m_to_world.matrix(0, 3),
            m_to_world.matrix(1, 0), m_to_world.matrix(1, 1), m_to_world.matrix(1, 2), m_to_world.matrix(1, 3),
            m_to_world.matrix(2, 0), m_to_world.matrix(2, 1), m_to_world.matrix(2, 2), m_to_world.matrix(2, 3)
        };
        std::memcpy(instance.transform, transform, sizeof(float) * 12);
        instance.instanceId = instance_id;
        instance.visibilityMask = 255;
        instance.flags = OPTIX_INSTANCE_FLAG_NONE;
        m_shapegroup->optix_accel_handle(context, instance.traversableHandle, instance.sbtOffset);
    }

    virtual void optix_fill_hitgroup_records(std::vector<HitGroupSbtRecord>&, OptixProgramGroup*) override {
        /* no op */
    }
#endif

    MTS_DECLARE_CLASS()
private:
   ref<Base> m_shapegroup;
};

MTS_IMPLEMENT_CLASS_VARIANT(Instance, Shape)
MTS_EXPORT_PLUGIN(Instance, "Instanced geometry")
NAMESPACE_END(mitsuba)
