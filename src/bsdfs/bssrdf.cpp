#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/fresnel.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/render/ior.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _bsdf-bssrdf:

BSSRDF with python
-------------------------------------------

.. pluginparameters::

 * - int_ior
   - |float| or |string|
   - Interior index of refraction specified numerically or using a known material name. (Default: bk7 / 1.5046)
 * - ext_ior
   - |float| or |string|
   - Exterior index of refraction specified numerically or using a known material name.  (Default: air / 1.000277)



This plugin is designed for simple calculation (a ray go inside or not) in c++ 
and for complex (actual BSSRDF calculation) in python.
Implementation of this plugin almost same as dielectric BSDF.

*/

template <typename Float, typename Spectrum>
class BSSRDF final : public BSDF<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(BSDF, m_flags, m_components)
    MTS_IMPORT_TYPES(Texture, Volume)

    BSSRDF(const Properties &props) : Base(props) {

        // Specifies the internal index of refraction at the interface
        ScalarFloat int_ior = lookup_ior(props, "int_ior", "bk7");

        // Specifies the external index of refraction at the interface
        ScalarFloat ext_ior = lookup_ior(props, "ext_ior", "air");

        if (int_ior < 0 || ext_ior < 0)
            Throw("The interior and exterior indices of refraction must"
                  " be positive!");

        m_eta = int_ior / ext_ior;

        m_albedo = props.volume<Volume>("albedo", 0.75f);
        m_sigmat = props.volume<Volume>("sigma_t", 1.f);
        m_scale = props.float_("scale", 1.0f);

        m_mesh_id = props.int_("mesh_id", 0);
        if(m_mesh_id <= 0){
            Log(Error, "The mesh ID should be set as larger than 0");
        }

        m_g = props.float_("g", 0.8f);
        if (m_g >= 1 || m_g <= -1)
            Log(Error, "The asymmetry parameter must lie in the interval (-1, 1)!");

        if (props.has_property("specular_reflectance"))
            m_specular_reflectance   = props.texture<Texture>("specular_reflectance", 1.f);
        if (props.has_property("specular_transmittance"))
            m_specular_transmittance = props.texture<Texture>("specular_transmittance", 1.f);

        m_components.push_back(BSDFFlags::DeltaReflection | BSDFFlags::FrontSide |
                               BSDFFlags::BackSide);
        m_components.push_back(BSDFFlags::DeltaTransmission | BSDFFlags::FrontSide |
                               BSDFFlags::BackSide | BSDFFlags::NonSymmetric);
        m_components.push_back((uint32_t)BSDFFlags::BSSRDF);

        m_flags = m_components[0] | m_components[1] | m_components[2];
    }

    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext &ctx,
                                             const SurfaceInteraction3f &si,
                                             Float sample1,
                                             const Point2f & /* sample2 */,
                                             Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        bool has_reflection   = ctx.is_enabled(BSDFFlags::DeltaReflection, 0),
             has_transmission = ctx.is_enabled(BSDFFlags::DeltaTransmission, 1);

        // Evaluate the Fresnel equations for unpolarized illumination
        Float cos_theta_i = Frame3f::cos_theta(si.wi);

        auto [r_i, cos_theta_t, eta_it, eta_ti] = fresnel(cos_theta_i, Float(m_eta));
        Float t_i = 1.f - r_i;

        // Lobe selection
        BSDFSample3f bs = zero<BSDFSample3f>();
        Mask selected_r;
        if (likely(has_reflection && has_transmission)) {
            selected_r = sample1 <= r_i && active;
            bs.pdf = select(selected_r, r_i, t_i);
        } else {
            if (has_reflection || has_transmission) {
                selected_r = Mask(has_reflection) && active;
                bs.pdf = 1.f;
            } else {
                return { bs, 0.f };
            }
        }
        Mask selected_t = !selected_r && active;

        bs.sampled_component = select(selected_r, UInt32(0), UInt32(1));
        bs.sampled_type      = select(selected_r, UInt32(+BSDFFlags::DeltaReflection),
                                                  UInt32(+BSDFFlags::DeltaTransmission));

        bs.wo = select(selected_r,
                       reflect(si.wi),
                       refract(si.wi, cos_theta_t, eta_ti));

        bs.eta = select(selected_r, Float(1.f), eta_it);

        bs.albedo = m_albedo->eval(si, selected_t);
        bs.sigma_t = m_sigmat->eval(si, selected_t);

        bs.g = select(selected_t, m_g, bs.g);

        UnpolarizedSpectrum reflectance = 1.f, transmittance = 1.f;
        if (m_specular_reflectance)
            reflectance = m_specular_reflectance->eval(si, selected_r);
        if (m_specular_transmittance)
            transmittance = m_specular_transmittance->eval(si, selected_t);

        Spectrum weight;
        if constexpr (is_polarized_v<Spectrum>) {
            /* Due to lack of reciprocity in polarization-aware pBRDFs, they are
               always evaluated w.r.t. the actual light propagation direction, no
               matter the transport mode. In the following, 'wi_hat' is toward the
               light source. */
            Vector3f wi_hat = ctx.mode == TransportMode::Radiance ? bs.wo : si.wi,
                     wo_hat = ctx.mode == TransportMode::Radiance ? si.wi : bs.wo;

            /* BSDF weights are Mueller matrices now. */
            Float cos_theta_i_hat = Frame3f::cos_theta(wi_hat);
            Spectrum R = mueller::specular_reflection(UnpolarizedSpectrum(cos_theta_i_hat), UnpolarizedSpectrum(m_eta)),
                     T = mueller::specular_transmission(UnpolarizedSpectrum(cos_theta_i_hat), UnpolarizedSpectrum(m_eta));

            if (likely(has_reflection && has_transmission)) {
                weight = select(selected_r, R, T) / bs.pdf;
            } else if (has_reflection || has_transmission) {
                weight = has_reflection ? R : T;
                bs.pdf = 1.f;
            }

            /* Apply frame reflection, according to "Stellar Polarimetry" by
               David Clarke, Appendix A.2 (A26) */
            weight = mueller::reverse(weight);

            /* The Stokes reference frame vector of this matrix lies in the plane
               of reflection / refraction. */
            Vector3f n(0, 0, 1);
            Vector3f s_axis_in = normalize(cross(n, -wi_hat)),
                     p_axis_in = normalize(cross(-wi_hat, s_axis_in)),
                     s_axis_out = normalize(cross(n, wo_hat)),
                     p_axis_out = normalize(cross(wo_hat, s_axis_out));

            /* Rotate in/out reference vector of weight s.t. it aligns with the
               implicit Stokes bases of -wi_hat & wo_hat. */
            weight = mueller::rotate_mueller_basis(weight,
                                                   -wi_hat, p_axis_in, mueller::stokes_basis(-wi_hat),
                                                    wo_hat, p_axis_out, mueller::stokes_basis(wo_hat));

            if (any_or<true>(selected_r))
                weight[selected_r] *= mueller::absorber(reflectance);

            if (any_or<true>(selected_t))
                weight[selected_t] *= mueller::absorber(transmittance);

        } else {
            if (likely(has_reflection && has_transmission)) {
                weight = 1.f;
            } else if (has_reflection || has_transmission) {
                weight = has_reflection ? r_i : t_i;
            }

            if (any_or<true>(selected_r))
                weight[selected_r] *= reflectance;

            if (any_or<true>(selected_t))
                weight[selected_t] *= transmittance;
        }

        if (any_or<true>(selected_t)) {
            /* For transmission, radiance must be scaled to account for the solid
               angle compression that occurs when crossing the interface. */
            Float factor = (ctx.mode == TransportMode::Radiance) ? eta_ti : Float(1.f);
            weight[selected_t] *= sqr(factor);
        }

        return { bs, select(active, weight, 0.f) };
    }

    Spectrum eval(const BSDFContext & /* ctx */, const SurfaceInteraction3f & /* si */,
                  const Vector3f & /* wo */, Mask /* active */) const override {
        return 0.f;
    }

    Float pdf(const BSDFContext & /* ctx */, const SurfaceInteraction3f & /* si */,
              const Vector3f & /* wo */, Mask /* active */) const override {
        return 0.f;
    }

    Int32 mesh_id(Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFID, active);
        Int32 result = m_mesh_id;
        return select(active, result, 0);
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_parameter("eta", m_eta);
        if (m_specular_reflectance)
            callback->put_object("specular_reflectance", m_specular_reflectance.get());
        if (m_specular_transmittance)
            callback->put_object("specular_transmittance", m_specular_transmittance.get());
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "SmoothDielectric[" << std::endl;
        if (m_specular_reflectance)
            oss << "  specular_reflectance = " << string::indent(m_specular_reflectance) << "," << std::endl;
        if (m_specular_transmittance)
            oss << "  specular_transmittance = " << string::indent(m_specular_transmittance) << ", " << std::endl;
        oss << "  eta = " << m_eta << "," << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    ScalarFloat m_eta;
    ref<Texture> m_specular_reflectance;
    ref<Texture> m_specular_transmittance;
    ref<Volume> m_sigmat, m_albedo;
    Float m_scale;
    ScalarFloat m_g;
    ScalarInt32 m_mesh_id;
};

MTS_IMPLEMENT_CLASS_VARIANT(BSSRDF, BSDF)
MTS_EXPORT_PLUGIN(BSSRDF, "BSSRDF")
NAMESPACE_END(mitsuba)
