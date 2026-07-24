//! Radix-2 FFTs over BLS12-381 scalars and scalar-multipliable groups.
//!
//! BTX needs transforms in G1, G2, and GT.  The butterfly is the usual
//! Cooley–Tukey butterfly, with field multiplication replaced by scalar
//! multiplication when the coefficients are group elements.

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
    /// BTX scales its fixed G2 kernel during one-time preprocessing, allowing
    /// the online GT transform to use this method and avoid `size` expensive
    /// target-group scalar multiplications.
    pub fn ifft_unscaled<G>(&self, values: &mut [G])
    where
        G: Copy + Send + Sync + Add<Output = G> + Sub<Output = G> + Mul<Scalar, Output = G>,
    {
        assert_eq!(values.len(), self.size);
        fft_with_generator(values, self.generator_inverse);
    }

    /// Unnormalized inverse transform specialized for GT.
    ///
    /// `blstrs::Gt` currently implements doubling with a general Fp12 square.
    /// FFT coefficients always remain in the cyclotomic target subgroup, so
    /// BLST's specialized cyclotomic square is both valid and faster.
    pub fn ifft_gt_unscaled(&self, values: &mut [Gt]) {
        assert_eq!(values.len(), self.size);
        bit_reverse(values);

        let mut butterfly_size = 2;
        while butterfly_size <= self.size {
            let half = butterfly_size / 2;
            let twiddle_step =
                scalar_pow(self.generator_inverse, (self.size / butterfly_size) as u64);

            if self.size >= PARALLEL_FFT_THRESHOLD && rayon::current_num_threads() > 1 {
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

    pub(crate) fn size_inverse(&self) -> Scalar {
        self.size_inverse
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
        let v = gt_mul_scalar(*rhs, twiddle);
        *lhs = u + v;
        *rhs = u - v;
        twiddle *= twiddle_step;
    }
}

#[inline]
fn gt_mul_scalar(value: Gt, scalar: Scalar) -> Gt {
    let value_fp12: Fp12 = value.into();
    let base: blst_fp12 = value_fp12.into();
    let identity_fp12: Fp12 = Gt::identity().into();
    let mut accumulator: blst_fp12 = identity_fp12.into();

    // Width-5 non-adjacent form cuts the number of Fp12 multiplications from
    // roughly 127 to roughly 43 for a full-width scalar.
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
    let bits = values.len().trailing_zeros();
    for index in 0..values.len() {
        let reversed = index.reverse_bits() >> (usize::BITS - bits);
        if index < reversed {
            values.swap(index, reversed);
        }
    }
}

fn scalar_pow(mut base: Scalar, mut exponent: u64) -> Scalar {
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
    use blstrs::{G1Projective, Gt};
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
    fn source_group_round_trip() {
        let domain = Radix2Domain::new(8).unwrap();
        let original = (0..8)
            .map(|value| G1Projective::generator() * Scalar::from((value + 1) as u64))
            .collect::<Vec<_>>();
        let mut transformed = original.clone();
        domain.fft(&mut transformed);
        domain.ifft(&mut transformed);
        assert_eq!(transformed, original);
    }

    #[test]
    fn target_group_round_trip() {
        let domain = Radix2Domain::new(8).unwrap();
        let original = (0..8)
            .map(|value| Gt::generator() * Scalar::from((value + 1) as u64))
            .collect::<Vec<_>>();
        let mut transformed = original.clone();
        domain.fft(&mut transformed);
        domain.ifft(&mut transformed);
        assert_eq!(transformed, original);
    }
}
