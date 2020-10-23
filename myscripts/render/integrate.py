import sys
import numpy as np
import enoki as ek
import render_config as config
import mitsuba
from bssrdf import BSSRDF

mitsuba.set_variant(config.variant)

from mitsuba.core import (Spectrum, Float, UInt32, UInt64, Vector2f, Vector3f,
                          Frame3f, Color3f, RayDifferential3f, srgb_to_xyz)
from mitsuba.core import Bitmap, Struct, Thread
from mitsuba.core.xml import load_file
from mitsuba.render import ImageBlock
from mitsuba.render import (Emitter, BSDF, BSDFContext, BSDFFlags, has_flag,
                            DirectionSample3f)


def mis_weight(pdf_a, pdf_b):
    pdf_a *= pdf_a
    pdf_b *= pdf_b
    return ek.select(pdf_a > 0.0, pdf_a / (pdf_a + pdf_b), Float(0.0))


def render(scene, spp, sample_per_pass):
    """
    Generate basic objects and rays.
    Throw render job

    Args:
        scene: Target scene object
        spp: Sample per pixel
        sample_per_pass: a number of sample per pixel in one vectored
                         calculation with GPU
    """

    if spp % sample_per_pass != 0:
        sys.exit("please set spp as multilple of sample_per_pass")

    # Generate basic objects for rendering
    sensor = scene.sensors()[0]
    film = sensor.film()
    sampler = sensor.sampler()
    film_size = film.crop_size()
    block = ImageBlock(
            film.crop_size(),
            channel_count=5,
            filter=film.reconstruction_filter(),
            border=False
    )
    block.clear()

    result = 0

    # The number of iteration for gpu rendering
    n_ite = int(spp / sample_per_pass)

    # The number of samples in one gpu rendering
    total_sample_count = ek.hprod(film_size) * sample_per_pass

    # Seed the sampler
    if sampler.wavefront_size() != total_sample_count:
        sampler.seed(0, total_sample_count)

    for ite in range(n_ite):
        # Get sampling position in film uv space
        pos = ek.arange(UInt32, total_sample_count)

        pos //= sample_per_pass
        scale = Vector2f(1.0 / film_size[0], 1.0 / film_size[1])
        pos = Vector2f(Float(pos % int(film_size[0])),
                       Float(pos // int(film_size[0])))

        pos += sampler.next_2d()

        # Get rays for path tracing
        rays, weights = sensor.sample_ray_differential(
            time=0,
            sample1=sampler.next_1d(),
            sample2=pos * scale,
            sample3=0
        )

        result, valid_rays = render_sample(scene, sampler, rays)
        result = weights * result
        xyz = Color3f(srgb_to_xyz(result))
        aovs = [xyz[0], xyz[1], xyz[2],
                ek.select(valid_rays, Float(1.0), Float(0.0)),
                1.0]

        block.put(pos, aovs)
        sampler.advance()

    xyzaw_np = np.array(block.data()).reshape([film_size[1], film_size[0], 5])

    bmp = Bitmap(xyzaw_np, Bitmap.PixelFormat.XYZAW)
    bmp = bmp.convert(Bitmap.PixelFormat.RGB, Struct.Type.Float32, srgb_gamma=False)
    bmp.write('result.exr')


def render_sample(scene, sampler, rays):
    """
    Sample RTE

    Args:
        scene: Target scene object
        sampler: Sampler object for random number
        rays: Given rays for sampling

    Returns:
        result: Sampling RTE result
    """
    eta = Float(1.0)
    emission_weight = Float(1.0)
    throughput = Spectrum(1.0)
    result = Spectrum(0.0)
    active = True

    ##### First interaction #####
    si = scene.ray_intersect(rays, active)
    active = si.is_valid() & active
    valid_rays = si.is_valid()

    emitter = si.emitter(scene, active)

    depth = 0

    # bssrdf = BSSRDF(config.model_name)

    while(True):
        depth += 1

        ##### Interaction with emitters #####
        result += ek.select(active,
                            emission_weight * throughput * Emitter.eval_vec(emitter, si, active),
                            Vector3f(0.0))

        active = active & si.is_valid()

        # Process russian roulette
        if depth > config.rr_depth:
            q = ek.min(ek.hmax(throughput) * ek.sqr(eta), 0.95)
            active = active & (sampler.next_1d(active) < q)
            throughput *= ek.rcp(q)

        # Stop if the number of bouces exceeds the given limit bounce, or
        # all rays are invalid. latter check is done only when the limit
        # bounce is infinite
        if depth >= config.max_depth:
            break

        ##### Emitter sampling #####
        bsdf = si.bsdf(rays)
        ctx = BSDFContext()

        active_e = active & has_flag(BSDF.flags_vec(bsdf), BSDFFlags.Smooth)
        ds, emitter_val = scene.sample_emitter_direction(si, sampler.next_2d(active_e),
                                                         True, active_e)
        active_e &= ek.neq(ds.pdf, 0.0)

        # Query the BSDF for that emitter-sampled direction
        wo = si.to_local(ds.d)
        bsdf_val = BSDF.eval_vec(bsdf, ctx, si, wo, active_e)
        # Determine density of sampling that same direction using BSDF sampling
        bsdf_pdf = BSDF.pdf_vec(bsdf, ctx, si, wo, active_e)

        mis = ek.select(ds.delta, Float(1), mis_weight(ds.pdf, bsdf_pdf))
        result += ek.select(active_e,
                            mis * throughput * bsdf_val * emitter_val,
                            Vector3f(0.0))

        ##### BSDF sampling #####
        bs, bsdf_val = BSDF.sample_vec(bsdf, ctx, si, sampler.next_1d(active),
                                       sampler.next_2d(active), active)
        throughput = throughput * bsdf_val
        active &= ek.any(ek.neq(throughput, 0))

        eta *= bs.eta

        # Whether the BSDF is BSSRDF or not?
        is_bssrdf = (active & has_flag(BSDF.flags_vec(bsdf), BSDFFlags.BSSRDF)
                     & (Frame3f.cos_theta(bs.wo) < Float(0.0)))

        ##### Process for BSSRDF #####
        # TODO: Convert incident position into local coordinates of mesh of interested as tensor
        in_pos = ek.select(is_bssrdf, si.p, Vector3f(0))

        # Get transform matrix w.r.t each mesh of BSSDF
        mesh_id = bsdf.mesh_id(is_bssrdf)
        transform_mat = get_trans_mat(mesh_id)
        
        # TODO: Get properties, e.g., medium params and incident angle as tensor

        # TODO: Get height map around incident position as tensor

        # TODO: Estimate position and absorption probability with VAE as mitsuba types
        # pos_recon_local, abs_recon = bssrdf.estimate(in_pos, im, props, sigma_n, is_bssrdf)
        # TODO: Convert from mesh coordinates to world coordinates

        # TODO: Project estimated position onto nearest mesh

        # TODO: Apply absorption probability
        ##############################
        

        # Intersect the BSDF ray against the scene geometry
        rays = RayDifferential3f(si.spawn_ray(si.to_world(bs.wo)))
        si_bsdf = scene.ray_intersect(rays, active)

        # Determine probability of having sampled that same
        # direction using emitter sampling
        emitter = si_bsdf.emitter(scene, active)
        ds = DirectionSample3f(si_bsdf, si)
        ds.object = emitter

        delta = has_flag(bs.sampled_type, BSDFFlags.Delta)
        emitter_pdf = ek.select(delta, Float(0.0),
                                scene.pdf_emitter_direction(si, ds, active))
        emission_weight = mis_weight(bs.pdf, emitter_pdf)

        si = si_bsdf

    return result, valid_rays
