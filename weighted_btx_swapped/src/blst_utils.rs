//! Thin wrappers around BLST batch operations that `blstrs` does not expose.

use blst::{
    blst_p1, blst_p1_affine, blst_p1s_mult_pippenger, blst_p1s_mult_pippenger_scratch_sizeof,
    blst_p1s_to_affine, blst_p2, blst_p2_affine, blst_p2s_mult_pippenger,
    blst_p2s_mult_pippenger_scratch_sizeof, blst_p2s_to_affine,
};
use blstrs::{G1Affine, G1Projective, G2Affine, G2Projective, Scalar};
use group::Group;

pub(crate) fn batch_normalize_g2(projective: &[G2Projective], affine: &mut [G2Affine]) {
    assert_eq!(projective.len(), affine.len());
    if projective.is_empty() {
        return;
    }

    // BLST accepts an array of pointers to contiguous point blocks, terminated
    // by a null pointer. Both blstrs point types are transparent wrappers over
    // their corresponding BLST representations.
    let blocks: [*const blst_p2; 2] = [projective[0].as_ref(), std::ptr::null()];
    unsafe {
        blst_p2s_to_affine(affine[0].as_mut(), blocks.as_ptr(), projective.len());
    }
}

pub(crate) fn batch_normalize_g1(projective: &[G1Projective], affine: &mut [G1Affine]) {
    assert_eq!(projective.len(), affine.len());
    if projective.is_empty() {
        return;
    }

    let blocks: [*const blst_p1; 2] = [projective[0].as_ref(), std::ptr::null()];
    unsafe {
        blst_p1s_to_affine(affine[0].as_mut(), blocks.as_ptr(), projective.len());
    }
}

/// Perform a single-threaded raw-BLST MSM over affine G2 points.
///
/// Partial decryptions are parallelized across parties by the caller, so this
/// avoids launching BLST's separate global worker pool inside every Rayon
/// task.
pub(crate) fn g2_multi_exp_affine_bytes(points: &[G2Affine], scalar_bytes: &[u8]) -> G2Projective {
    assert_eq!(scalar_bytes.len(), points.len().saturating_mul(32));
    if points.is_empty() {
        return G2Projective::identity();
    }

    let point_blocks: [*const blst_p2_affine; 2] = [points[0].as_ref(), std::ptr::null()];
    let scalar_blocks: [*const u8; 2] = [scalar_bytes.as_ptr(), std::ptr::null()];
    let scratch_size = unsafe { blst_p2s_mult_pippenger_scratch_sizeof(points.len()) };
    let word_size = std::mem::size_of::<u64>();
    let mut scratch = vec![0u64; scratch_size.div_ceil(word_size)];

    let mut result = G2Projective::identity();
    unsafe {
        blst_p2s_mult_pippenger(
            result.as_mut(),
            point_blocks.as_ptr(),
            points.len(),
            scalar_blocks.as_ptr(),
            255,
            scratch.as_mut_ptr(),
        );
    }
    result
}

/// Perform a BLST Pippenger MSM directly from affine G1 points.
///
/// `blstrs::G1Projective::multi_exp` accepts projective inputs and first batch
/// normalizes them. Weighted BTX stores its large public decryption key in
/// affine form, so converting it back to projective only to normalize it again
/// is both slower and substantially more memory-hungry.
#[cfg(test)]
pub(crate) fn g1_multi_exp_affine(points: &[G1Affine], scalars: &[Scalar]) -> G1Projective {
    assert_eq!(points.len(), scalars.len());
    g1_multi_exp_affine_bytes(points, &scalars_to_le_bytes(scalars))
}

/// Perform a single-threaded raw-BLST affine MSM with pre-encoded scalars.
///
/// The weighted decryption loops are already parallelized with Rayon. Calling
/// BLST's Rust `MultiPoint` wrapper there would launch its independent global
/// pool for every MSM, oversubscribe parallel runs, and make a one-thread
/// Rayon configuration misleading. The C Pippenger entry point used here is
/// one task; Rayon controls concurrency at the call sites.
pub(crate) fn g1_multi_exp_affine_bytes(points: &[G1Affine], scalar_bytes: &[u8]) -> G1Projective {
    assert_eq!(scalar_bytes.len(), points.len().saturating_mul(32));
    if points.is_empty() {
        return G1Projective::identity();
    }

    // G1Affine is a transparent wrapper around blst_p1_affine. BLST's block
    // API consumes the first element of each contiguous block followed by a
    // null terminator; the scalars use the same layout.
    let point_blocks: [*const blst_p1_affine; 2] = [points[0].as_ref(), std::ptr::null()];
    g1_multi_exp_affine_raw(&point_blocks, points.len(), scalar_bytes)
}

/// Perform an affine MSM over a sparse selection without copying the points.
///
/// BLST accepts several point blocks. A one-point block for each selected
/// index lets the large exponent-major public-key block remain in place; the
/// scalar encodings are still one contiguous block in selection order.
pub(crate) fn g1_multi_exp_affine_indexed(
    points: &[G1Affine],
    indices: &[usize],
    scalar_bytes: &[u8],
) -> G1Projective {
    assert_eq!(scalar_bytes.len(), indices.len().saturating_mul(32));
    if indices.is_empty() {
        return G1Projective::identity();
    }

    let mut point_blocks = Vec::with_capacity(indices.len() + 1);
    point_blocks.extend(indices.iter().map(|index| {
        points
            .get(*index)
            .expect("affine MSM index is in bounds")
            .as_ref() as *const blst_p1_affine
    }));
    point_blocks.push(std::ptr::null());
    g1_multi_exp_affine_raw(&point_blocks, indices.len(), scalar_bytes)
}

fn g1_multi_exp_affine_raw(
    point_blocks: &[*const blst_p1_affine],
    point_count: usize,
    scalar_bytes: &[u8],
) -> G1Projective {
    debug_assert!(point_count > 0);
    debug_assert_eq!(scalar_bytes.len(), point_count * 32);

    let scalar_blocks: [*const u8; 2] = [scalar_bytes.as_ptr(), std::ptr::null()];
    let scratch_size = unsafe { blst_p1s_mult_pippenger_scratch_sizeof(point_count) };
    let word_size = std::mem::size_of::<u64>();
    let mut scratch = vec![0u64; scratch_size.div_ceil(word_size)];

    let mut result = G1Projective::identity();
    unsafe {
        blst_p1s_mult_pippenger(
            result.as_mut(),
            point_blocks.as_ptr(),
            point_count,
            scalar_blocks.as_ptr(),
            255,
            scratch.as_mut_ptr(),
        );
    }
    result
}

/// Concatenate canonical little-endian scalar encodings for BLST Pippenger.
pub(crate) fn scalars_to_le_bytes(scalars: &[Scalar]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(scalars.len().saturating_mul(32));
    for scalar in scalars {
        bytes.extend_from_slice(&scalar.to_bytes_le());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use blstrs::Scalar;
    use group::{Curve, Group};

    #[test]
    fn g2_batch_normalization_matches_individual_conversion() {
        let projective = (0..32)
            .map(|index| {
                if index % 11 == 0 {
                    G2Projective::identity()
                } else {
                    G2Projective::generator() * Scalar::from((index + 1) as u64)
                }
            })
            .collect::<Vec<_>>();
        let expected = projective.iter().map(Curve::to_affine).collect::<Vec<_>>();
        let mut actual = vec![G2Affine::default(); projective.len()];
        batch_normalize_g2(&projective, &mut actual);
        assert_eq!(actual, expected);
    }

    #[test]
    fn g1_batch_normalization_matches_individual_conversion() {
        let projective = (0..32)
            .map(|index| {
                if index % 11 == 0 {
                    G1Projective::identity()
                } else {
                    G1Projective::generator() * Scalar::from((index + 1) as u64)
                }
            })
            .collect::<Vec<_>>();
        let expected = projective.iter().map(Curve::to_affine).collect::<Vec<_>>();
        let mut actual = vec![G1Affine::default(); projective.len()];
        batch_normalize_g1(&projective, &mut actual);
        assert_eq!(actual, expected);
    }

    #[test]
    fn affine_g1_msm_matches_blstrs_pippenger() {
        let projective = (0..64)
            .map(|index| {
                if index % 17 == 0 {
                    G1Projective::identity()
                } else {
                    G1Projective::generator() * Scalar::from((3 * index + 1) as u64)
                }
            })
            .collect::<Vec<_>>();
        let affine = projective.iter().map(Curve::to_affine).collect::<Vec<_>>();
        let scalars = (0..projective.len())
            .map(|index| Scalar::from((5 * index + 2) as u64))
            .collect::<Vec<_>>();

        let expected = G1Projective::multi_exp(&projective, &scalars);
        assert_eq!(g1_multi_exp_affine(&affine, &scalars), expected);
        assert_eq!(g1_multi_exp_affine(&[], &[]), G1Projective::identity());
    }

    #[test]
    fn affine_g2_msm_matches_blstrs_pippenger() {
        let projective = (0..64)
            .map(|index| {
                if index % 17 == 0 {
                    G2Projective::identity()
                } else {
                    G2Projective::generator() * Scalar::from((3 * index + 1) as u64)
                }
            })
            .collect::<Vec<_>>();
        let affine = projective.iter().map(Curve::to_affine).collect::<Vec<_>>();
        let scalars = (0..projective.len())
            .map(|index| Scalar::from((5 * index + 2) as u64))
            .collect::<Vec<_>>();

        let expected = G2Projective::multi_exp(&projective, &scalars);
        assert_eq!(
            g2_multi_exp_affine_bytes(&affine, &scalars_to_le_bytes(&scalars)),
            expected
        );
        assert_eq!(
            g2_multi_exp_affine_bytes(&[], &[]),
            G2Projective::identity()
        );
    }

    #[test]
    fn indexed_affine_g1_msm_matches_selected_points() {
        let points = (0..80)
            .map(|index| {
                (G1Projective::generator() * Scalar::from((3 * index + 1) as u64)).to_affine()
            })
            .collect::<Vec<_>>();
        let indices = (0usize..80)
            .filter(|index| index % 3 != 1)
            .collect::<Vec<_>>();
        let scalars = indices
            .iter()
            .enumerate()
            .map(|(position, _)| Scalar::from((5 * position + 2) as u64))
            .collect::<Vec<_>>();
        let selected_projective = indices
            .iter()
            .map(|index| G1Projective::from(points[*index]))
            .collect::<Vec<_>>();
        let expected = G1Projective::multi_exp(&selected_projective, &scalars);

        assert_eq!(
            g1_multi_exp_affine_indexed(&points, &indices, &scalars_to_le_bytes(&scalars)),
            expected
        );
        assert_eq!(
            g1_multi_exp_affine_indexed(&points, &[], &[]),
            G1Projective::identity()
        );
    }

    #[test]
    fn scalar_bytes_are_concatenated_in_little_endian_order() {
        let scalars = [Scalar::from(1u64), Scalar::from(0x0102u64)];
        let bytes = scalars_to_le_bytes(&scalars);
        assert_eq!(bytes.len(), 64);
        assert_eq!(&bytes[..32], &scalars[0].to_bytes_le());
        assert_eq!(&bytes[32..], &scalars[1].to_bytes_le());
        assert_eq!(bytes[32], 0x02);
        assert_eq!(bytes[33], 0x01);
    }
}
