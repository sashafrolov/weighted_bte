//! Radix-2 FFTs over BLS12-381 scalars and scalar-multipliable groups.
//!
//! PFE needs transforms in G1, G2, and GT.  The GT path uses BLST's
//! cyclotomic square and a width-5 NAF exponentiation instead of `blstrs`'
//! generic binary `Gt * Scalar` implementation.

use blst::{blst_fp12, blst_fp12_conjugate, blst_fp12_cyclotomic_sqr, blst_fp12_mul};
use blstrs::{Fp12, Gt, Scalar};
use ff::{Field, PrimeField};
use group::Group;
use rayon::prelude::*;
use std::ops::{Add, Mul, Sub};

use crate::error::{Error, Result};

const PARALLEL_FFT_THRESHOLD: usize = 256;

#[derive(Clone, Copy, Debug)]
pub struct Radix2Domain {
    size: usize,
    generator: Scalar,
    generator_inverse: Scalar,
    size_inverse: Scalar,
}

impl Radix2Domain {
    pub fn new(size: usize) -> Result<Self> {
        if size == 0 || !size.is_power_of_two() {
            return Err(Error::InvalidBatchSize(size));
        }

        let log_size = size.trailing_zeros();
        if log_size > Scalar::S {
            return Err(Error::InvalidBatchSize(size));
        }

        let mut generator = Scalar::ROOT_OF_UNITY;
        for _ in log_size..Scalar::S {
            generator = generator.square();
        }

        let generator_inverse =
            Option::<Scalar>::from(generator.invert()).expect("a root of unity is nonzero");
        let size_inverse = Option::<Scalar>::from(Scalar::from(size as u64).invert())
            .expect("the FFT domain size is nonzero in the scalar field");

        Ok(Self {
            size,
            generator,
            generator_inverse,
            size_inverse,
        })
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn generator(&self) -> Scalar {
        self.generator
    }

    pub fn generator_inverse(&self) -> Scalar {
        self.generator_inverse
    }

    pub fn size_inverse(&self) -> Scalar {
        self.size_inverse
    }

    pub fn fft<G>(&self, values: &mut [G])
    where
        G: Copy + Send + Sync + Add<Output = G> + Sub<Output = G> + Mul<Scalar, Output = G>,
    {
        assert_eq!(values.len(), self.size);
        fft_with_generator(values, self.generator);
    }

    pub fn ifft<G>(&self, values: &mut [G])
    where
        G: Copy + Send + Sync + Add<Output = G> + Sub<Output = G> + Mul<Scalar, Output = G>,
    {
        assert_eq!(values.len(), self.size);
        self.ifft_unscaled(values);

        if values.len() >= PARALLEL_FFT_THRESHOLD && rayon::current_num_threads() > 1 {
            values
                .par_iter_mut()
                .for_each(|value| *value = *value * self.size_inverse);
        } else {
            values
                .iter_mut()
                .for_each(|value| *value = *value * self.size_inverse);
        }
    }

    /// Apply the inverse-root transform without multiplying by `1 / size`.
    ///
    /// PFE folds this scale into its transformed scalar convolution kernel,
    /// avoiding an additional group-scalar multiplication after each IFFT.
    pub fn ifft_unscaled<G>(&self, values: &mut [G])
    where
        G: Copy + Send + Sync + Add<Output = G> + Sub<Output = G> + Mul<Scalar, Output = G>,
    {
        assert_eq!(values.len(), self.size);
        fft_with_generator(values, self.generator_inverse);
    }

    pub fn fft_gt(&self, values: &mut [Gt]) {
        assert_eq!(values.len(), self.size);
        fft_gt_with_generator(values, self.generator);
    }

    pub fn ifft_gt_unscaled(&self, values: &mut [Gt]) {
        assert_eq!(values.len(), self.size);
        fft_gt_with_generator(values, self.generator_inverse);
    }
}

fn fft_with_generator<G>(values: &mut [G], generator: Scalar)
where
    G: Copy + Send + Sync + Add<Output = G> + Sub<Output = G> + Mul<Scalar, Output = G>,
{
    bit_reverse(values);

    let size = values.len();
    let mut butterfly_size = 2;
    while butterfly_size <= size {
        let half = butterfly_size / 2;
        let twiddle_step = scalar_pow(generator, (size / butterfly_size) as u64);

        if size >= PARALLEL_FFT_THRESHOLD && rayon::current_num_threads() > 1 {
            values
                .par_chunks_mut(butterfly_size)
                .for_each(|chunk| butterfly_chunk(chunk, half, twiddle_step));
        } else {
            values
                .chunks_mut(butterfly_size)
                .for_each(|chunk| butterfly_chunk(chunk, half, twiddle_step));
        }

        butterfly_size *= 2;
    }
}

fn fft_gt_with_generator(values: &mut [Gt], generator: Scalar) {
    bit_reverse(values);

    let size = values.len();
    let mut butterfly_size = 2;
    while butterfly_size <= size {
        let half = butterfly_size / 2;
        let twiddle_step = scalar_pow(generator, (size / butterfly_size) as u64);

        if size >= PARALLEL_FFT_THRESHOLD && rayon::current_num_threads() > 1 {
            values
                .par_chunks_mut(butterfly_size)
                .for_each(|chunk| gt_butterfly_chunk(chunk, half, twiddle_step));
        } else {
            values
                .chunks_mut(butterfly_size)
                .for_each(|chunk| gt_butterfly_chunk(chunk, half, twiddle_step));
        }

        butterfly_size *= 2;
    }
}

#[inline]
fn butterfly_chunk<G>(chunk: &mut [G], half: usize, twiddle_step: Scalar)
where
    G: Copy + Add<Output = G> + Sub<Output = G> + Mul<Scalar, Output = G>,
{
    let (left, right) = chunk.split_at_mut(half);
    let u = left[0];
    let v = right[0];
    left[0] = u + v;
    right[0] = u - v;

    let mut twiddle = twiddle_step;
    for (lhs, rhs) in left.iter_mut().zip(right.iter_mut()).skip(1) {
        let u = *lhs;
        let v = *rhs * twiddle;
        *lhs = u + v;
        *rhs = u - v;
        twiddle *= twiddle_step;
    }
}

#[inline]
fn gt_butterfly_chunk(chunk: &mut [Gt], half: usize, twiddle_step: Scalar) {
    let (left, right) = chunk.split_at_mut(half);
    let u = left[0];
    let v = right[0];
    left[0] = u + v;
    right[0] = u - v;

    let mut twiddle = twiddle_step;
    for (lhs, rhs) in left.iter_mut().zip(right.iter_mut()).skip(1) {
        let u = *lhs;
        let v = mul_gt(*rhs, twiddle);
        *lhs = u + v;
        *rhs = u - v;
        twiddle *= twiddle_step;
    }
}

#[inline]
pub(crate) fn mul_gt(value: Gt, scalar: Scalar) -> Gt {
    let value_fp12: Fp12 = value.into();
    let base: blst_fp12 = value_fp12.into();
    let identity_fp12: Fp12 = Gt::identity().into();
    let mut accumulator: blst_fp12 = identity_fp12.into();

    let (digits, digit_count) = width_five_naf(scalar);

    let mut two_base = base;
    let input = two_base;
    unsafe {
        blst_fp12_cyclotomic_sqr(&mut two_base, &input);
    }
    let mut odd_multiples = [base; 8];
    for index in 1..odd_multiples.len() {
        let previous = odd_multiples[index - 1];
        unsafe {
            blst_fp12_mul(&mut odd_multiples[index], &previous, &two_base);
        }
    }

    for digit in digits[..digit_count].iter().rev() {
        let input = accumulator;
        unsafe {
            blst_fp12_cyclotomic_sqr(&mut accumulator, &input);
        }
        if *digit != 0 {
            let table_index = ((digit.unsigned_abs() as usize) - 1) / 2;
            let mut multiple = odd_multiples[table_index];
            if *digit < 0 {
                unsafe {
                    blst_fp12_conjugate(&mut multiple);
                }
            }
            let input = accumulator;
            unsafe {
                blst_fp12_mul(&mut accumulator, &input, &multiple);
            }
        }
    }

    Gt::from(Fp12::from(accumulator))
}

#[inline]
pub(crate) fn mul_gt_signed_small(value: Gt, coefficient: i64) -> Gt {
    if coefficient == 0 {
        return Gt::identity();
    }

    let value_fp12: Fp12 = value.into();
    let mut base: blst_fp12 = value_fp12.into();
    let identity_fp12: Fp12 = Gt::identity().into();
    let mut accumulator: blst_fp12 = identity_fp12.into();
    let mut magnitude = coefficient.unsigned_abs();

    while magnitude != 0 {
        if magnitude & 1 == 1 {
            let left = accumulator;
            unsafe {
                blst_fp12_mul(&mut accumulator, &left, &base);
            }
        }
        magnitude >>= 1;
        if magnitude != 0 {
            let input = base;
            unsafe {
                blst_fp12_cyclotomic_sqr(&mut base, &input);
            }
        }
    }

    if coefficient < 0 {
        unsafe {
            blst_fp12_conjugate(&mut accumulator);
        }
    }

    Gt::from(Fp12::from(accumulator))
}

fn width_five_naf(scalar: Scalar) -> ([i8; 257], usize) {
    let bytes = scalar.to_bytes_le();
    let mut limbs = [
        u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
        u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
        u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
        u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
    ];
    let mut digits = [0i8; 257];
    let mut position = 0;

    while limbs.iter().any(|limb| *limb != 0) {
        if limbs[0] & 1 == 1 {
            let mut digit = (limbs[0] & 31) as i16;
            if digit >= 16 {
                digit -= 32;
            }
            digits[position] = digit as i8;
            if digit > 0 {
                sub_small(&mut limbs, digit as u64);
            } else {
                add_small(&mut limbs, (-digit) as u64);
            }
        }
        shift_right_one(&mut limbs);
        position += 1;
    }

    (digits, position)
}

fn sub_small(limbs: &mut [u64; 4], value: u64) {
    let (first, mut borrow) = limbs[0].overflowing_sub(value);
    limbs[0] = first;
    for limb in limbs.iter_mut().skip(1) {
        if !borrow {
            break;
        }
        let (next, next_borrow) = limb.overflowing_sub(1);
        *limb = next;
        borrow = next_borrow;
    }
}

fn add_small(limbs: &mut [u64; 4], value: u64) {
    let (first, mut carry) = limbs[0].overflowing_add(value);
    limbs[0] = first;
    for limb in limbs.iter_mut().skip(1) {
        if !carry {
            break;
        }
        let (next, next_carry) = limb.overflowing_add(1);
        *limb = next;
        carry = next_carry;
    }
}

fn shift_right_one(limbs: &mut [u64; 4]) {
    let mut carry = 0u64;
    for limb in limbs.iter_mut().rev() {
        let next_carry = *limb << 63;
        *limb = (*limb >> 1) | carry;
        carry = next_carry;
    }
}

fn bit_reverse<G>(values: &mut [G]) {
    if values.len() <= 1 {
        return;
    }
    let bits = values.len().trailing_zeros();
    for index in 0..values.len() {
        let reversed = index.reverse_bits() >> (usize::BITS - bits);
        if index < reversed {
            values.swap(index, reversed);
        }
    }
}

pub(crate) fn scalar_pow(mut base: Scalar, mut exponent: u64) -> Scalar {
    let mut result = Scalar::ONE;
    while exponent != 0 {
        if exponent & 1 == 1 {
            result *= base;
        }
        base = base.square();
        exponent >>= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use blstrs::{G1Projective, G2Projective};
    use group::Group;

    #[test]
    fn scalar_round_trip() {
        let domain = Radix2Domain::new(16).unwrap();
        let original = (0..16)
            .map(|value| Scalar::from((value + 1) as u64))
            .collect::<Vec<_>>();
        let mut transformed = original.clone();
        domain.fft(&mut transformed);
        domain.ifft(&mut transformed);
        assert_eq!(transformed, original);
    }

    #[test]
    fn source_groups_round_trip() {
        let domain = Radix2Domain::new(8).unwrap();
        let g1 = (0..8)
            .map(|value| G1Projective::generator() * Scalar::from((value + 1) as u64))
            .collect::<Vec<_>>();
        let g2 = (0..8)
            .map(|value| G2Projective::generator() * Scalar::from((value + 1) as u64))
            .collect::<Vec<_>>();

        let mut transformed = g1.clone();
        domain.fft(&mut transformed);
        domain.ifft(&mut transformed);
        assert_eq!(transformed, g1);

        let mut transformed = g2.clone();
        domain.fft(&mut transformed);
        domain.ifft(&mut transformed);
        assert_eq!(transformed, g2);
    }

    #[test]
    fn optimized_target_group_round_trip() {
        let domain = Radix2Domain::new(8).unwrap();
        let original = (0..8)
            .map(|value| Gt::generator() * Scalar::from((value + 1) as u64))
            .collect::<Vec<_>>();
        let mut transformed = original.clone();
        domain.fft_gt(&mut transformed);
        domain.ifft_gt_unscaled(&mut transformed);
        transformed
            .iter_mut()
            .for_each(|value| *value = mul_gt(*value, domain.size_inverse()));
        assert_eq!(transformed, original);
    }

    #[test]
    fn optimized_gt_multiplication_matches_blstrs() {
        let values = [
            Scalar::ZERO,
            Scalar::ONE,
            -Scalar::ONE,
            Scalar::from(17u64),
            Scalar::ROOT_OF_UNITY,
        ];
        let base = Gt::generator() * Scalar::from(29u64);
        for scalar in values {
            assert_eq!(mul_gt(base, scalar), base * scalar);
        }
        for coefficient in [-17i64, -1, 0, 1, 2, 19] {
            let scalar = if coefficient < 0 {
                -Scalar::from(coefficient.unsigned_abs())
            } else {
                Scalar::from(coefficient as u64)
            };
            assert_eq!(mul_gt_signed_small(base, coefficient), base * scalar);
        }
    }
}
