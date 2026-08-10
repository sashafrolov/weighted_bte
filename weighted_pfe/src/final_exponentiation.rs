//! Low-level split final exponentiation for BLS12-381.
//!
//! A pairing final exponentiation factors as
//!
//! ```text
//! (p^12 - 1) / r
//!     = (p^6 - 1)(p^2 + 1) * (p^4 - p^2 + 1) / r
//!       \_________________/   \_____________________/
//!             easy                      hard
//! ```
//!
//! Applying the easy part puts a Miller-loop result in the cyclotomic
//! subgroup.  The hard part maps that subgroup into the prime-order target
//! group.  Keeping these steps separate lets a caller apply the easy part to
//! every frequency-domain pairing, perform a cyclotomic inverse FFT, and apply
//! the hard part only to retained coefficients.
//!
//! The hard-part addition chain below is a direct Rust transcription of
//! BLST's BLS12-381 final-exponentiation chain.  It deliberately uses only
//! public `blst` functions.

use std::fmt;

use blst::{
    blst_final_exp, blst_fp12, blst_fp12_conjugate, blst_fp12_cyclotomic_sqr,
    blst_fp12_frobenius_map, blst_fp12_inverse, blst_fp12_is_one, blst_fp12_mul, blst_fp12_one,
    blst_fp6, blst_miller_loop_lines, blst_miller_loop_n, blst_precompute_lines,
};
use blstrs::{Fp12, G1Affine, G2Affine, Gt};
use group::prime::PrimeCurveAffine;

/// Number of precomputed BLS12-381 Miller-loop line coefficients in BLST.
pub(crate) const MILLER_LINE_COUNT: usize = 68;

/// A nonzero, un-final-exponentiated Miller-loop result.
///
/// The private field records the invariant needed by the easy part: values
/// constructed through this module are nonzero elements of Fp12.
#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub(crate) struct RawMillerResult(blst_fp12);

impl fmt::Debug for RawMillerResult {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RawMillerResult")
            .finish_non_exhaustive()
    }
}

impl RawMillerResult {
    /// Construct from a raw value, rejecting zero.
    #[cfg(test)]
    pub(crate) fn from_raw(raw: blst_fp12) -> Option<Self> {
        (!fp12_is_zero(&raw)).then_some(Self(raw))
    }

    pub(crate) fn as_raw(&self) -> &blst_fp12 {
        &self.0
    }
}

/// An Fp12 element known to lie in the BLS12-381 cyclotomic subgroup.
///
/// This group is larger than the prime-order pairing target group.  In
/// particular, an easy-final-exponentiated value or an intermediate FFT value
/// must not be exposed as a `Gt` until [`hard_final_exponentiation`] has run.
#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub(crate) struct CyclotomicFp12(blst_fp12);

impl fmt::Debug for CyclotomicFp12 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CyclotomicFp12")
            .field("identity", &self.is_identity())
            .finish()
    }
}

impl CyclotomicFp12 {
    pub(crate) fn identity() -> Self {
        Self(fp12_one())
    }

    /// Treat a raw value as cyclotomic without checking it.
    ///
    /// # Safety
    ///
    /// The caller must ensure the input belongs to the cyclotomic subgroup.
    /// Arbitrary Fp12 values do not admit conjugation as their group inverse
    /// and are not valid inputs to BLST's cyclotomic squaring routine.
    pub(crate) unsafe fn from_raw_unchecked(raw: blst_fp12) -> Self {
        Self(raw)
    }

    #[cfg(test)]
    pub(crate) fn from_gt(value: Gt) -> Self {
        let fp12: Fp12 = value.into();
        Self(fp12.into())
    }

    pub(crate) fn as_raw(&self) -> &blst_fp12 {
        &self.0
    }

    #[cfg(test)]
    pub(crate) fn into_raw(self) -> blst_fp12 {
        self.0
    }

    /// Convert to `Gt` without applying the hard exponent.
    ///
    /// # Safety
    ///
    /// The caller must separately prove that this value lies in the
    /// prime-order r-torsion subgroup, not merely the cyclotomic subgroup.
    #[cfg(test)]
    pub(crate) unsafe fn into_gt_unchecked(self) -> Gt {
        Gt::from(Fp12::from(self.0))
    }

    pub(crate) fn is_identity(&self) -> bool {
        unsafe { blst_fp12_is_one(&self.0) }
    }

    /// Multiply two cyclotomic elements in place.
    #[inline]
    pub(crate) fn mul_assign(&mut self, rhs: &Self) {
        let lhs = std::ptr::addr_of!(self.0);
        unsafe {
            blst_fp12_mul(std::ptr::addr_of_mut!(self.0), lhs, rhs.as_raw());
        }
    }

    #[inline]
    pub(crate) fn product(mut self, rhs: &Self) -> Self {
        self.mul_assign(rhs);
        self
    }

    /// Cyclotomic inverse, which is just conjugation.
    #[inline]
    pub(crate) fn conjugate_assign(&mut self) {
        unsafe {
            blst_fp12_conjugate(&mut self.0);
        }
    }

    #[inline]
    pub(crate) fn inverse(mut self) -> Self {
        self.conjugate_assign();
        self
    }

    #[inline]
    pub(crate) fn square_assign(&mut self) {
        let input = std::ptr::addr_of!(self.0);
        unsafe {
            blst_fp12_cyclotomic_sqr(std::ptr::addr_of_mut!(self.0), input);
        }
    }
}

/// Contiguous, fixed-size BLST Miller-loop line table for one G2 point.
///
/// Unlike `blstrs::G2Prepared`, this type contains no per-point `Vec`.
/// Consequently `Box<[PreparedG2Lines]>` stores every line table in one
/// contiguous allocation.
#[repr(C, align(64))]
#[derive(Clone)]
pub(crate) struct PreparedG2Lines {
    lines: [blst_fp6; MILLER_LINE_COUNT],
    infinity: bool,
}

impl fmt::Debug for PreparedG2Lines {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedG2Lines")
            .field("infinity", &self.infinity)
            .finish_non_exhaustive()
    }
}

impl PreparedG2Lines {
    pub(crate) fn from_affine(point: &G2Affine) -> Self {
        let infinity = bool::from(point.is_identity());
        let mut lines = [blst_fp6::default(); MILLER_LINE_COUNT];
        if !infinity {
            unsafe {
                blst_precompute_lines(lines.as_mut_ptr(), point.as_ref());
            }
        }
        Self { lines, infinity }
    }

    #[cfg(test)]
    pub(crate) fn batch_from_affine(points: &[G2Affine]) -> Box<[Self]> {
        points
            .iter()
            .map(Self::from_affine)
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    /// Run one raw Miller loop using this prepared G2 point.
    pub(crate) fn miller_loop(&self, point: &G1Affine) -> RawMillerResult {
        if self.infinity || bool::from(point.is_identity()) {
            return RawMillerResult(fp12_one());
        }

        let mut output = blst_fp12::default();
        unsafe {
            blst_miller_loop_lines(&mut output, self.lines.as_ptr(), point.as_ref());
        }

        // A Miller-loop result is a product of nonzero field elements.
        RawMillerResult(output)
    }
}

/// Compute one unfinalized multi-pairing without retaining G2 line tables.
///
/// Weighted BTX uses a different G2 element for every accepted party and
/// output slot.  Caching BLST's 68-line table for all of them would use far
/// more memory than the affine public key itself.  BLST can consume two
/// contiguous affine arrays directly, sharing the Miller loop across all
/// terms while keeping only one Fp12 accumulator.
pub(crate) fn unprepared_multi_miller_loop(
    left: &[G1Affine],
    right: &[G2Affine],
) -> RawMillerResult {
    assert_eq!(left.len(), right.len());
    if left.is_empty() {
        return RawMillerResult(fp12_one());
    }

    // `blst_miller_loop_n` does not treat a G2 point at infinity as a neutral
    // term when it appears inside a mixed batch. Filter both source-group
    // identities explicitly. The ordinary hot path only pays this linear
    // scan and retains its allocation-free contiguous call.
    if left
        .iter()
        .zip(right)
        .any(|(left, right)| bool::from(left.is_identity() | right.is_identity()))
    {
        let (filtered_left, filtered_right): (Vec<_>, Vec<_>) = left
            .iter()
            .copied()
            .zip(right.iter().copied())
            .filter(|(left, right)| {
                !bool::from(left.is_identity()) && !bool::from(right.is_identity())
            })
            .unzip();
        return unprepared_multi_miller_loop(&filtered_left, &filtered_right);
    }

    // `blstrs` affine points are transparent wrappers around the matching
    // BLST structs.  BLST's block API accepts the first element of each
    // contiguous block followed by a null terminator.
    let right_blocks = [right[0].as_ref(), std::ptr::null()];
    let left_blocks = [left[0].as_ref(), std::ptr::null()];
    let mut output = blst_fp12::default();
    unsafe {
        blst_miller_loop_n(
            &mut output,
            right_blocks.as_ptr(),
            left_blocks.as_ptr(),
            left.len(),
        );
    }
    RawMillerResult(output)
}

/// Apply the easy part of the BLS12-381 final exponentiation.
pub(crate) fn easy_final_exponentiation(input: RawMillerResult) -> CyclotomicFp12 {
    easy_final_exponentiation_with_inverse(input.0, fp12_inverse(&input.0))
}

/// Apply the easy part to a batch, sharing one Fp12 inversion.
pub(crate) fn batch_easy_final_exponentiation(inputs: &[RawMillerResult]) -> Vec<CyclotomicFp12> {
    if inputs.is_empty() {
        return Vec::new();
    }

    let mut prefixes = Vec::with_capacity(inputs.len());
    let mut product = fp12_one();
    for input in inputs {
        prefixes.push(product);
        product = fp12_mul(&product, input.as_raw());
    }

    let mut product_inverse = fp12_inverse(&product);
    let mut inverses = vec![blst_fp12::default(); inputs.len()];
    for index in (0..inputs.len()).rev() {
        inverses[index] = fp12_mul(&product_inverse, &prefixes[index]);
        product_inverse = fp12_mul(&product_inverse, inputs[index].as_raw());
    }

    inputs
        .iter()
        .zip(inverses)
        .map(|(input, inverse)| easy_final_exponentiation_with_inverse(input.0, inverse))
        .collect()
}

/// Apply the hard part and produce an r-torsion target-group element.
pub(crate) fn hard_final_exponentiation(input: CyclotomicFp12) -> Gt {
    let mut result = input.0;

    // This is the hard part of BLST's final_exp() in pairing.c.
    let y0 = cyclotomic_square(&result);
    let mut y1 = raise_to_z(&y0);
    let mut y2 = raise_to_z_div_by_2(&y1);
    let mut y3 = result;
    fp12_conjugate_assign(&mut y3);
    y1 = fp12_mul(&y1, &y3);
    fp12_conjugate_assign(&mut y1);
    y1 = fp12_mul(&y1, &y2);
    y2 = raise_to_z(&y1);
    y3 = raise_to_z(&y2);
    fp12_conjugate_assign(&mut y1);
    y3 = fp12_mul(&y3, &y1);
    fp12_conjugate_assign(&mut y1);
    y1 = fp12_frobenius_map(&y1, 3);
    y2 = fp12_frobenius_map(&y2, 2);
    y1 = fp12_mul(&y1, &y2);
    y2 = raise_to_z(&y3);
    y2 = fp12_mul(&y2, &y0);
    y2 = fp12_mul(&y2, &result);
    y1 = fp12_mul(&y1, &y2);
    y2 = fp12_frobenius_map(&y3, 1);
    result = fp12_mul(&y1, &y2);

    Gt::from(Fp12::from(result))
}

#[cfg(test)]
pub(crate) fn batch_hard_final_exponentiation(inputs: &[CyclotomicFp12]) -> Vec<Gt> {
    inputs
        .iter()
        .copied()
        .map(hard_final_exponentiation)
        .collect()
}

/// Reference full final exponentiation through BLST.
pub(crate) fn full_final_exponentiation(input: RawMillerResult) -> Gt {
    let mut output = blst_fp12::default();
    unsafe {
        blst_final_exp(&mut output, input.as_raw());
    }
    Gt::from(Fp12::from(output))
}

fn easy_final_exponentiation_with_inverse(
    input: blst_fp12,
    input_inverse: blst_fp12,
) -> CyclotomicFp12 {
    // input^(p^6 - 1)
    let mut conjugate = input;
    fp12_conjugate_assign(&mut conjugate);
    let mut result = fp12_mul(&conjugate, &input_inverse);

    // input^((p^6 - 1)(p^2 + 1))
    let frobenius = fp12_frobenius_map(&result, 2);
    result = fp12_mul(&result, &frobenius);

    // The easy exponent maps every nonzero input into the cyclotomic subgroup.
    CyclotomicFp12(result)
}

#[inline]
fn fp12_one() -> blst_fp12 {
    unsafe { *blst_fp12_one() }
}

#[inline]
#[cfg(test)]
fn fp12_zero() -> blst_fp12 {
    blst_fp12 {
        fp6: [blst_fp6::default(); 2],
    }
}

#[inline]
#[cfg(test)]
fn fp12_is_zero(value: &blst_fp12) -> bool {
    value.fp6 == fp12_zero().fp6
}

#[inline]
fn fp12_mul(left: &blst_fp12, right: &blst_fp12) -> blst_fp12 {
    let mut output = blst_fp12::default();
    unsafe {
        blst_fp12_mul(&mut output, left, right);
    }
    output
}

#[inline]
fn fp12_inverse(input: &blst_fp12) -> blst_fp12 {
    let mut output = blst_fp12::default();
    unsafe {
        blst_fp12_inverse(&mut output, input);
    }
    output
}

#[inline]
fn fp12_conjugate_assign(value: &mut blst_fp12) {
    unsafe {
        blst_fp12_conjugate(value);
    }
}

#[inline]
fn fp12_frobenius_map(input: &blst_fp12, power: usize) -> blst_fp12 {
    assert!(
        (1..=3).contains(&power),
        "BLST's Fp12 Frobenius map requires a power in 1..=3"
    );
    let mut output = blst_fp12::default();
    unsafe {
        blst_fp12_frobenius_map(&mut output, input, power);
    }
    output
}

#[inline]
fn cyclotomic_square(input: &blst_fp12) -> blst_fp12 {
    let mut output = blst_fp12::default();
    unsafe {
        blst_fp12_cyclotomic_sqr(&mut output, input);
    }
    output
}

fn multiply_then_square(
    accumulator: blst_fp12,
    multiplicand: &blst_fp12,
    square_count: usize,
) -> blst_fp12 {
    let mut result = fp12_mul(&accumulator, multiplicand);
    for _ in 0..square_count {
        result = cyclotomic_square(&result);
    }
    result
}

/// Raise a cyclotomic element to half the absolute BLS12-381 parameter.
fn raise_to_z_div_by_2(input: &blst_fp12) -> blst_fp12 {
    let mut result = cyclotomic_square(input);
    result = multiply_then_square(result, input, 2);
    result = multiply_then_square(result, input, 3);
    result = multiply_then_square(result, input, 9);
    result = multiply_then_square(result, input, 32);
    result = multiply_then_square(result, input, 15);
    fp12_conjugate_assign(&mut result);
    result
}

/// Raise a cyclotomic element to the (negative) BLS12-381 parameter.
fn raise_to_z(input: &blst_fp12) -> blst_fp12 {
    cyclotomic_square(&raise_to_z_div_by_2(input))
}

#[cfg(test)]
mod tests {
    use blstrs::{pairing, G1Projective, G2Projective, Scalar};
    use ff::Field;
    use group::{Curve, Group};
    use rand_core::OsRng;

    use super::*;

    fn sample_pair() -> (G1Affine, G2Affine) {
        let p = G1Projective::generator() * Scalar::random(OsRng);
        let q = G2Projective::generator() * Scalar::random(OsRng);
        (p.to_affine(), q.to_affine())
    }

    #[test]
    fn unprepared_multi_miller_loop_matches_pairing_sum() {
        let mut pairs = (0..9).map(|_| sample_pair()).collect::<Vec<_>>();
        pairs.push((G1Affine::identity(), sample_pair().1));
        pairs.push((sample_pair().0, G2Affine::identity()));
        pairs.push((G1Affine::identity(), G2Affine::identity()));
        let left = pairs.iter().map(|(left, _)| *left).collect::<Vec<_>>();
        let right = pairs.iter().map(|(_, right)| *right).collect::<Vec<_>>();
        let expected = pairs
            .iter()
            .map(|(left, right)| pairing(left, right))
            .sum::<Gt>();

        assert_eq!(
            full_final_exponentiation(unprepared_multi_miller_loop(&left, &right)),
            expected
        );
        assert_eq!(
            full_final_exponentiation(unprepared_multi_miller_loop(&[], &[])),
            Gt::identity()
        );
    }

    #[test]
    fn split_final_exponentiation_matches_blst() {
        for _ in 0..8 {
            let (p, q) = sample_pair();
            let prepared = PreparedG2Lines::from_affine(&q);
            let raw = prepared.miller_loop(&p);

            let reference = full_final_exponentiation(raw);
            let split = hard_final_exponentiation(easy_final_exponentiation(raw));

            assert_eq!(reference, split);
            assert_eq!(reference, pairing(&p, &q));
        }
    }

    #[test]
    fn identity_pairings_match() {
        let identity_g1 = G1Affine::identity();
        let identity_g2 = G2Affine::identity();
        let (p, q) = sample_pair();

        for (left, right) in [(identity_g1, q), (p, identity_g2)] {
            let prepared = PreparedG2Lines::from_affine(&right);
            let raw = prepared.miller_loop(&left);
            assert_eq!(
                hard_final_exponentiation(easy_final_exponentiation(raw)),
                Gt::identity()
            );
        }
    }

    #[test]
    fn batch_easy_part_matches_individual_easy_parts() {
        let raw = (0..16)
            .map(|_| {
                let (p, q) = sample_pair();
                PreparedG2Lines::from_affine(&q).miller_loop(&p)
            })
            .collect::<Vec<_>>();

        let individual = raw
            .iter()
            .copied()
            .map(easy_final_exponentiation)
            .collect::<Vec<_>>();
        let batched = batch_easy_final_exponentiation(&raw);

        assert_eq!(individual, batched);

        let individual_hard = batch_hard_final_exponentiation(&individual);
        let batched_hard = batch_hard_final_exponentiation(&batched);
        assert_eq!(individual_hard, batched_hard);
    }

    #[test]
    fn easy_then_hard_is_homomorphic() {
        for _ in 0..4 {
            let (p0, q0) = sample_pair();
            let (p1, q1) = sample_pair();
            let raw0 = PreparedG2Lines::from_affine(&q0).miller_loop(&p0);
            let raw1 = PreparedG2Lines::from_affine(&q1).miller_loop(&p1);

            let easy0 = easy_final_exponentiation(raw0);
            let easy1 = easy_final_exponentiation(raw1);
            let product = easy0.product(&easy1);
            let quotient = easy0.product(&easy1.inverse());

            assert_eq!(
                hard_final_exponentiation(product),
                pairing(&p0, &q0) + pairing(&p1, &q1)
            );
            assert_eq!(
                hard_final_exponentiation(quotient),
                pairing(&p0, &q0) - pairing(&p1, &q1)
            );
        }
    }

    #[test]
    fn cyclotomic_in_place_operations_are_alias_safe() {
        let (p, q) = sample_pair();
        let easy = easy_final_exponentiation(PreparedG2Lines::from_affine(&q).miller_loop(&p));

        let mut squared = easy;
        squared.square_assign();
        assert_eq!(squared, easy.product(&easy));

        assert!(easy.product(&easy.inverse()).is_identity());
    }

    #[test]
    fn raw_zero_is_rejected() {
        assert!(RawMillerResult::from_raw(fp12_zero()).is_none());
        assert!(RawMillerResult::from_raw(fp12_one()).is_some());
    }

    #[test]
    fn prepared_line_tables_are_contiguous() {
        let points = (0..3).map(|_| sample_pair().1).collect::<Vec<G2Affine>>();
        let prepared = PreparedG2Lines::batch_from_affine(&points);

        let first = std::ptr::addr_of!(prepared[0]) as usize;
        let second = std::ptr::addr_of!(prepared[1]) as usize;
        assert_eq!(second - first, std::mem::size_of::<PreparedG2Lines>());
        assert_eq!(first % 64, 0);
    }
}
