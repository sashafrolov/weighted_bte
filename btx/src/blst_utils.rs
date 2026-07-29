//! Thin wrappers around BLST batch operations that `blstrs` does not expose.

use blst::{blst_p1, blst_p1s_to_affine, blst_p2, blst_p2s_to_affine};
use blstrs::{G1Affine, G1Projective, G2Affine, G2Projective};

pub(crate) fn batch_normalize_g1(projective: &[G1Projective], affine: &mut [G1Affine]) {
    assert_eq!(projective.len(), affine.len());
    if projective.is_empty() {
        return;
    }

    // BLST accepts an array of pointers to contiguous point blocks, terminated
    // by a null pointer. Both blstrs point types are transparent wrappers over
    // their corresponding BLST representations.
    let blocks: [*const blst_p1; 2] = [projective[0].as_ref(), std::ptr::null()];
    unsafe {
        blst_p1s_to_affine(affine[0].as_mut(), blocks.as_ptr(), projective.len());
    }
}

pub(crate) fn batch_normalize_g2(projective: &[G2Projective], affine: &mut [G2Affine]) {
    assert_eq!(projective.len(), affine.len());
    if projective.is_empty() {
        return;
    }

    let blocks: [*const blst_p2; 2] = [projective[0].as_ref(), std::ptr::null()];
    unsafe {
        blst_p2s_to_affine(affine[0].as_mut(), blocks.as_ptr(), projective.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blstrs::Scalar;
    use group::{Curve, Group};

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
}
