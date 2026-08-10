//! Radix-2 FFTs over BLS12-381 scalars and scalar-multipliable groups.
//!
//! BTX needs transforms in G1, G2, and GT.  The butterfly is the usual
//! Cooley–Tukey butterfly, with field multiplication replaced by scalar
//! multiplication when the coefficients are group elements.

use blst::{
    blst_fp12, blst_fp12_conjugate, blst_fp12_cyclotomic_sqr, blst_fp12_frobenius_map,
    blst_fp12_mul,
};
#[cfg(test)]
use blstrs::Gt;
use blstrs::{Fp12, Scalar};
use ff::{Field, PrimeField};
#[cfg(test)]
use group::Group;
use rayon::prelude::*;
use std::ops::{Add, Mul, Sub};

use crate::{
    error::{Error, Result},
    final_exponentiation::CyclotomicFp12,
};

const PARALLEL_FFT_THRESHOLD: usize = 256;
const BLS12_381_X_ABS: u64 = 0xd201_0000_0001_0000;
const GT_DECOMPOSITION_COMPONENTS: usize = 4;
const GT_WNAF_WIDTH: usize = 4;
const GT_WNAF_TABLE_SIZE: usize = 1 << (GT_WNAF_WIDTH - 2);
const GT_WNAF_DIGITS: usize = 65;

/// Public-scalar control flow for one decomposed cyclotomic exponentiation.
///
/// FFT chunks in the same stage use the same twiddle at each butterfly
/// offset. Keeping the scalar-only work in a reusable plan avoids repeating
/// the base-x divisions and wNAF conversion for every chunk.
struct CyclotomicScalarPlan {
    nafs: [[i8; GT_WNAF_DIGITS]; GT_DECOMPOSITION_COMPONENTS],
    digit_count: usize,
}

impl CyclotomicScalarPlan {
    fn new(scalar: Scalar) -> Self {
        let scalar_components = decompose_scalar_base_x(scalar);
        let mut nafs = [[0i8; GT_WNAF_DIGITS]; GT_DECOMPOSITION_COMPONENTS];
        let mut digit_count = 0;
        for (naf, component) in nafs.iter_mut().zip(scalar_components) {
            let (component_naf, component_digits) = width_w_naf_u64(component);
            *naf = component_naf;
            digit_count = digit_count.max(component_digits);
        }
        Self { nafs, digit_count }
    }
}

/// Apply a scalar through the four-way BLS-x/Frobenius decomposition.
///
/// For BLS12-381, `p = x (mod r)` and `r = x^4 - x^2 + 1`, where `x` is
/// negative.  Writing the canonical scalar exactly as `sum d_i |x|^i` gives
/// four 64-bit components.  Frobenius^i followed by conjugation for odd i acts
/// as exponentiation by `|x|^i` after projection into r-torsion.  For [`Gt`]
/// inputs that relation already holds exactly.
///
/// A split final exponentiation may call this on a cyclotomic value before its
/// hard projection.  In that case the returned raw value can differ from
/// ordinary integer exponentiation, but their hard projections are equal.
#[inline]
pub(crate) fn cyclotomic_mul_scalar_decomposed(
    value: CyclotomicFp12,
    scalar: Scalar,
) -> CyclotomicFp12 {
    let plan = CyclotomicScalarPlan::new(scalar);
    cyclotomic_mul_scalar_with_plan(value, &plan)
}

#[inline]
fn cyclotomic_mul_scalar_with_plan(
    value: CyclotomicFp12,
    plan: &CyclotomicScalarPlan,
) -> CyclotomicFp12 {
    if plan.digit_count == 0 {
        return CyclotomicFp12::identity();
    }

    // Precompute odd multiples only once.  Frobenius and conjugation are
    // group endomorphisms, so the other three tables are derived from the
    // first instead of spending another 3 * (2^(w-2)-1) Fp12 products.
    let base = *value.as_raw();
    let mut two_base = base;
    let input = two_base;
    unsafe {
        blst_fp12_cyclotomic_sqr(&mut two_base, &input);
    }
    let mut base_table = [base; GT_WNAF_TABLE_SIZE];
    for index in 1..base_table.len() {
        let previous = base_table[index - 1];
        unsafe {
            blst_fp12_mul(&mut base_table[index], &previous, &two_base);
        }
    }

    let mut tables = [base_table; GT_DECOMPOSITION_COMPONENTS];
    for (power, table) in tables.iter_mut().enumerate().skip(1) {
        for (entry, base_multiple) in table.iter_mut().zip(base_table) {
            unsafe {
                blst_fp12_frobenius_map(entry, &base_multiple, power);
                if power & 1 == 1 {
                    // x is negative, so phi^i = [-|x|^i] for odd i.
                    blst_fp12_conjugate(entry);
                }
            }
        }
    }

    // Interleave the four width-4 NAFs.  This shares one sequence of at most
    // 64 cyclotomic squarings across all four 64-bit components, instead of
    // performing a 255-bit exponentiation.
    let mut accumulator: blst_fp12 = Fp12::ONE.into();
    let mut started = false;
    for position in (0..plan.digit_count).rev() {
        if started {
            let input = accumulator;
            unsafe {
                blst_fp12_cyclotomic_sqr(&mut accumulator, &input);
            }
        }

        for (component, table) in tables.iter().enumerate() {
            let digit = plan.nafs[component][position];
            if digit == 0 {
                continue;
            }

            let table_index = ((digit.unsigned_abs() as usize) - 1) / 2;
            let mut multiple = table[table_index];
            if digit < 0 {
                unsafe {
                    blst_fp12_conjugate(&mut multiple);
                }
            }

            if started {
                let input = accumulator;
                unsafe {
                    blst_fp12_mul(&mut accumulator, &input, &multiple);
                }
            } else {
                accumulator = multiple;
                started = true;
            }
        }
    }

    // Every operation above preserves the caller-provided cyclotomic
    // invariant.
    unsafe { CyclotomicFp12::from_raw_unchecked(accumulator) }
}

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

    /// Forward transform specialized for target-group values.
    ///
    /// Every twiddle multiplication uses the four-way BLS-x/Frobenius
    /// decomposition below, sharing the scalar plan across equal butterfly
    /// positions rather than invoking `Gt * Scalar` independently.
    #[cfg(test)]
    pub(crate) fn fft_gt(&self, values: &mut [Gt]) {
        assert_eq!(values.len(), self.size);
        bit_reverse(values);

        let mut twiddle_plans = Vec::new();
        let mut butterfly_size = 2;
        while butterfly_size <= self.size {
            let half = butterfly_size / 2;
            let twiddle_step = scalar_pow(self.generator, (self.size / butterfly_size) as u64);
            fill_cyclotomic_twiddle_plans(&mut twiddle_plans, half, twiddle_step);
            let thread_count = rayon::current_num_threads();

            if self.size >= PARALLEL_FFT_THRESHOLD && thread_count > 1 {
                let chunk_count = self.size / butterfly_size;
                if chunk_count < thread_count {
                    values
                        .par_chunks_mut(butterfly_size)
                        .for_each(|chunk| gt_butterfly_chunk_parallel(chunk, half, &twiddle_plans));
                } else {
                    values
                        .par_chunks_mut(butterfly_size)
                        .for_each(|chunk| gt_butterfly_chunk(chunk, half, &twiddle_plans));
                }
            } else {
                values
                    .chunks_mut(butterfly_size)
                    .for_each(|chunk| gt_butterfly_chunk(chunk, half, &twiddle_plans));
            }

            butterfly_size *= 2;
        }
    }

    /// Unnormalized inverse transform specialized for GT.
    ///
    /// `blstrs::Gt` currently implements doubling with a general Fp12 square.
    /// FFT coefficients always remain in the cyclotomic target subgroup, so
    /// BLST's specialized cyclotomic square is both valid and faster.
    #[cfg(test)]
    pub(crate) fn ifft_gt_unscaled(&self, values: &mut [Gt]) {
        assert_eq!(values.len(), self.size);
        bit_reverse(values);

        let mut twiddle_plans = Vec::new();
        let mut butterfly_size = 2;
        while butterfly_size <= self.size {
            let half = butterfly_size / 2;
            let twiddle_step =
                scalar_pow(self.generator_inverse, (self.size / butterfly_size) as u64);
            fill_cyclotomic_twiddle_plans(&mut twiddle_plans, half, twiddle_step);
            let thread_count = rayon::current_num_threads();

            if self.size >= PARALLEL_FFT_THRESHOLD && thread_count > 1 {
                let chunk_count = self.size / butterfly_size;
                if chunk_count < thread_count {
                    values
                        .par_chunks_mut(butterfly_size)
                        .for_each(|chunk| gt_butterfly_chunk_parallel(chunk, half, &twiddle_plans));
                } else {
                    values
                        .par_chunks_mut(butterfly_size)
                        .for_each(|chunk| gt_butterfly_chunk(chunk, half, &twiddle_plans));
                }
            } else {
                values
                    .chunks_mut(butterfly_size)
                    .for_each(|chunk| gt_butterfly_chunk(chunk, half, &twiddle_plans));
            }

            butterfly_size *= 2;
        }
    }

    /// Forward transform over easy-final-exponentiated cyclotomic values.
    ///
    /// The inputs need not yet be in the prime-order target subgroup. Every
    /// scalar action is chosen so its hard projection agrees with the usual
    /// FFT, allowing callers to defer the hard final exponentiation until
    /// after subsequent linear combinations.
    pub(crate) fn fft_cyclotomic(&self, values: &mut [CyclotomicFp12]) {
        self.fft_cyclotomic_with_generator(values, self.generator);
    }

    /// Unnormalized inverse transform over cyclotomic values.
    pub(crate) fn ifft_cyclotomic_unscaled(&self, values: &mut [CyclotomicFp12]) {
        self.fft_cyclotomic_with_generator(values, self.generator_inverse);
    }

    fn fft_cyclotomic_with_generator(&self, values: &mut [CyclotomicFp12], generator: Scalar) {
        assert_eq!(values.len(), self.size);
        bit_reverse(values);

        let mut twiddle_plans = Vec::new();
        let mut butterfly_size = 2;
        while butterfly_size <= self.size {
            let half = butterfly_size / 2;
            let twiddle_step = scalar_pow(generator, (self.size / butterfly_size) as u64);
            fill_cyclotomic_twiddle_plans(&mut twiddle_plans, half, twiddle_step);
            let thread_count = rayon::current_num_threads();

            if self.size >= PARALLEL_FFT_THRESHOLD && thread_count > 1 {
                let chunk_count = self.size / butterfly_size;
                if chunk_count < thread_count {
                    values.par_chunks_mut(butterfly_size).for_each(|chunk| {
                        cyclotomic_butterfly_chunk_parallel(chunk, half, &twiddle_plans)
                    });
                } else {
                    values
                        .par_chunks_mut(butterfly_size)
                        .for_each(|chunk| cyclotomic_butterfly_chunk(chunk, half, &twiddle_plans));
                }
            } else {
                values
                    .chunks_mut(butterfly_size)
                    .for_each(|chunk| cyclotomic_butterfly_chunk(chunk, half, &twiddle_plans));
            }

            butterfly_size *= 2;
        }
    }

    /// Unnormalized inverse transform over the cyclotomic subgroup, retaining
    /// only an output prefix.
    ///
    /// BTX needs at most the first half of a `2B`-point inverse transform.
    /// Every stage before the final crossing remains necessary, but the final
    /// stage can omit the unused right outputs. The values need not yet be in
    /// r-torsion: the decomposed twiddle action is guaranteed to agree after
    /// the caller applies the hard final exponentiation.
    #[cfg(test)]
    pub(crate) fn ifft_cyclotomic_prefix_unscaled(
        &self,
        values: &mut [CyclotomicFp12],
        output_count: usize,
    ) {
        assert_eq!(values.len(), self.size);
        assert!(output_count <= self.size / 2);
        bit_reverse(values);

        let mut twiddle_plans = Vec::new();
        let mut butterfly_size = 2;
        while butterfly_size < self.size {
            let half = butterfly_size / 2;
            let twiddle_step =
                scalar_pow(self.generator_inverse, (self.size / butterfly_size) as u64);
            fill_cyclotomic_twiddle_plans(&mut twiddle_plans, half, twiddle_step);
            let thread_count = rayon::current_num_threads();

            if self.size >= PARALLEL_FFT_THRESHOLD && thread_count > 1 {
                let chunk_count = self.size / butterfly_size;
                if chunk_count < thread_count {
                    values.par_chunks_mut(butterfly_size).for_each(|chunk| {
                        cyclotomic_butterfly_chunk_parallel(chunk, half, &twiddle_plans)
                    });
                } else {
                    values
                        .par_chunks_mut(butterfly_size)
                        .for_each(|chunk| cyclotomic_butterfly_chunk(chunk, half, &twiddle_plans));
                }
            } else {
                values
                    .chunks_mut(butterfly_size)
                    .for_each(|chunk| cyclotomic_butterfly_chunk(chunk, half, &twiddle_plans));
            }

            butterfly_size *= 2;
        }

        let half = self.size / 2;
        let twiddle_step = self.generator_inverse;
        let (left, right) = values.split_at_mut(half);
        if output_count == 0 {
            return;
        }
        left[0] = left[0].product(&right[0]);

        fill_cyclotomic_twiddle_plans(&mut twiddle_plans, output_count, twiddle_step);
        let left = &mut left[1..output_count];
        let right = &right[1..output_count];
        if self.size >= PARALLEL_FFT_THRESHOLD && rayon::current_num_threads() > 1 {
            left.par_iter_mut()
                .zip(right.par_iter())
                .zip(twiddle_plans.par_iter())
                .for_each(|((lhs, rhs), plan)| {
                    let value = cyclotomic_mul_scalar_with_plan(*rhs, plan);
                    *lhs = lhs.product(&value);
                });
        } else {
            left.iter_mut()
                .zip(right)
                .zip(&twiddle_plans)
                .for_each(|((lhs, rhs), plan)| {
                    let value = cyclotomic_mul_scalar_with_plan(*rhs, plan);
                    *lhs = lhs.product(&value);
                });
        }
    }

    #[cfg(test)]
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
        let thread_count = rayon::current_num_threads();

        if size >= PARALLEL_FFT_THRESHOLD && thread_count > 1 {
            let chunk_count = size / butterfly_size;
            if chunk_count < thread_count {
                let twiddles = stage_twiddles(half, twiddle_step);
                values
                    .par_chunks_mut(butterfly_size)
                    .for_each(|chunk| butterfly_chunk_parallel(chunk, half, &twiddles));
            } else {
                values
                    .par_chunks_mut(butterfly_size)
                    .for_each(|chunk| butterfly_chunk(chunk, half, twiddle_step));
            }
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
fn butterfly_chunk_parallel<G>(chunk: &mut [G], half: usize, twiddles: &[Scalar])
where
    G: Copy + Send + Sync + Add<Output = G> + Sub<Output = G> + Mul<Scalar, Output = G>,
{
    let (left, right) = chunk.split_at_mut(half);
    let u = left[0];
    let v = right[0];
    left[0] = u + v;
    right[0] = u - v;

    left[1..]
        .par_iter_mut()
        .zip(right[1..].par_iter_mut())
        .zip(twiddles.par_iter())
        .for_each(|((lhs, rhs), twiddle)| {
            let u = *lhs;
            let v = *rhs * *twiddle;
            *lhs = u + v;
            *rhs = u - v;
        });
}

#[inline]
#[cfg(test)]
fn gt_butterfly_chunk(chunk: &mut [Gt], half: usize, twiddle_plans: &[CyclotomicScalarPlan]) {
    let (left, right) = chunk.split_at_mut(half);
    let u = left[0];
    let v = right[0];
    left[0] = u + v;
    right[0] = u - v;

    left[1..]
        .iter_mut()
        .zip(&mut right[1..])
        .zip(twiddle_plans)
        .for_each(|((lhs, rhs), plan)| {
            let u = *lhs;
            let v = gt_mul_scalar_with_plan(*rhs, plan);
            *lhs = u + v;
            *rhs = u - v;
        });
}

#[inline]
#[cfg(test)]
fn gt_butterfly_chunk_parallel(
    chunk: &mut [Gt],
    half: usize,
    twiddle_plans: &[CyclotomicScalarPlan],
) {
    let (left, right) = chunk.split_at_mut(half);
    let u = left[0];
    let v = right[0];
    left[0] = u + v;
    right[0] = u - v;

    left[1..]
        .par_iter_mut()
        .zip(right[1..].par_iter_mut())
        .zip(twiddle_plans.par_iter())
        .for_each(|((lhs, rhs), plan)| {
            let u = *lhs;
            let v = gt_mul_scalar_with_plan(*rhs, plan);
            *lhs = u + v;
            *rhs = u - v;
        });
}

#[inline]
fn cyclotomic_butterfly_chunk(
    chunk: &mut [CyclotomicFp12],
    half: usize,
    twiddle_plans: &[CyclotomicScalarPlan],
) {
    let (left, right) = chunk.split_at_mut(half);
    let u = left[0];
    let v = right[0];
    left[0] = u.product(&v);
    right[0] = u.product(&v.inverse());

    left[1..]
        .iter_mut()
        .zip(&mut right[1..])
        .zip(twiddle_plans)
        .for_each(|((lhs, rhs), plan)| {
            let u = *lhs;
            let v = cyclotomic_mul_scalar_with_plan(*rhs, plan);
            *lhs = u.product(&v);
            *rhs = u.product(&v.inverse());
        });
}

#[inline]
fn cyclotomic_butterfly_chunk_parallel(
    chunk: &mut [CyclotomicFp12],
    half: usize,
    twiddle_plans: &[CyclotomicScalarPlan],
) {
    let (left, right) = chunk.split_at_mut(half);
    let u = left[0];
    let v = right[0];
    left[0] = u.product(&v);
    right[0] = u.product(&v.inverse());

    left[1..]
        .par_iter_mut()
        .zip(right[1..].par_iter_mut())
        .zip(twiddle_plans.par_iter())
        .for_each(|((lhs, rhs), plan)| {
            let u = *lhs;
            let v = cyclotomic_mul_scalar_with_plan(*rhs, plan);
            *lhs = u.product(&v);
            *rhs = u.product(&v.inverse());
        });
}

#[inline]
#[cfg(test)]
pub(crate) fn mul_gt(value: Gt, scalar: Scalar) -> Gt {
    let plan = CyclotomicScalarPlan::new(scalar);
    gt_mul_scalar_with_plan(value, &plan)
}

/// Multiply by a public signed machine integer using cyclotomic double/add.
/// Cauchy's closed-form spectrum uses only values in `[-B, B]`, so this is
/// substantially cheaper than converting them to near-modulus field scalars.
#[inline]
#[cfg(test)]
pub(crate) fn mul_gt_signed_small(value: Gt, coefficient: i64) -> Gt {
    if coefficient == 0 {
        return Gt::identity();
    }

    let value_fp12: Fp12 = value.into();
    let mut base: blst_fp12 = value_fp12.into();
    let mut accumulator: blst_fp12 = Fp12::ONE.into();
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

/// Multiply a cyclotomic value by a public signed machine integer.
#[inline]
pub(crate) fn mul_cyclotomic_signed_small(
    value: CyclotomicFp12,
    coefficient: i64,
) -> CyclotomicFp12 {
    if coefficient == 0 {
        return CyclotomicFp12::identity();
    }

    let mut magnitude = coefficient.unsigned_abs();
    let mut base = value;
    let mut result = CyclotomicFp12::identity();
    while magnitude != 0 {
        if magnitude & 1 == 1 {
            result.mul_assign(&base);
        }
        magnitude >>= 1;
        if magnitude != 0 {
            base.square_assign();
        }
    }
    if coefficient < 0 {
        result.inverse()
    } else {
        result
    }
}

#[inline]
#[cfg(test)]
fn gt_mul_scalar_with_plan(value: Gt, plan: &CyclotomicScalarPlan) -> Gt {
    let result = cyclotomic_mul_scalar_with_plan(CyclotomicFp12::from_gt(value), plan);
    // A Gt input is already in r-torsion, and the decomposed action preserves
    // that subgroup.
    unsafe { result.into_gt_unchecked() }
}

fn decompose_scalar_base_x(scalar: Scalar) -> [u64; GT_DECOMPOSITION_COMPONENTS] {
    let bytes = scalar.to_bytes_le();
    let mut limbs = [
        u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
        u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
        u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
        u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
    ];
    let mut components = [0u64; GT_DECOMPOSITION_COMPONENTS];
    for component in &mut components {
        *component = div_rem_u256_by_u64(&mut limbs, BLS12_381_X_ABS);
    }
    debug_assert_eq!(limbs, [0; 4]);
    components
}

fn div_rem_u256_by_u64(limbs: &mut [u64; 4], divisor: u64) -> u64 {
    let mut remainder = 0u128;
    for limb in limbs.iter_mut().rev() {
        let dividend = (remainder << 64) | *limb as u128;
        *limb = (dividend / divisor as u128) as u64;
        remainder = dividend % divisor as u128;
    }
    remainder as u64
}

fn width_w_naf_u64(value: u64) -> ([i8; GT_WNAF_DIGITS], usize) {
    // A negative wNAF digit can temporarily carry into bit 64 for inputs near
    // u64::MAX. The extra bit is why GT_WNAF_DIGITS is 65.
    let mut value = value as u128;
    let mut digits = [0i8; GT_WNAF_DIGITS];
    let mut position = 0;

    while value != 0 {
        if value & 1 == 1 {
            let mut digit = (value & ((1u128 << GT_WNAF_WIDTH) - 1)) as i16;
            if digit >= (1 << (GT_WNAF_WIDTH - 1)) {
                digit -= 1 << GT_WNAF_WIDTH;
            }
            digits[position] = digit as i8;
            if digit > 0 {
                value -= digit as u128;
            } else {
                value += (-digit) as u128;
            }
        }
        value >>= 1;
        position += 1;
    }

    (digits, position)
}

fn fill_cyclotomic_twiddle_plans(
    plans: &mut Vec<CyclotomicScalarPlan>,
    pair_count: usize,
    twiddle_step: Scalar,
) {
    plans.clear();
    plans.reserve(pair_count.saturating_sub(1));

    let mut twiddle = twiddle_step;
    for _ in 1..pair_count {
        plans.push(CyclotomicScalarPlan::new(twiddle));
        twiddle *= twiddle_step;
    }
}

fn stage_twiddles(pair_count: usize, twiddle_step: Scalar) -> Vec<Scalar> {
    let mut twiddles = Vec::with_capacity(pair_count.saturating_sub(1));
    let mut twiddle = twiddle_step;
    for _ in 1..pair_count {
        twiddles.push(twiddle);
        twiddle *= twiddle_step;
    }
    twiddles
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
    use crate::final_exponentiation::{
        easy_final_exponentiation, full_final_exponentiation, hard_final_exponentiation,
        PreparedG2Lines,
    };
    use blstrs::{G1Projective, G2Projective, Gt};
    use ff::Field;
    use group::{Curve, Group};
    use rand_core::OsRng;

    // (p^4 - p^2 + 1) / r, little-endian u64 limbs.  This is the hard
    // final-exponentiation projection from the cyclotomic subgroup to GT.
    const HARD_EXPONENT: [u64; 20] = [
        0xe516_c3f4_38e3_ba79,
        0xfa99_12aa_e208_ccf1,
        0x905c_e937_335d_5b68,
        0xc71a_2629_b0de_a236,
        0x8377_4940_9967_54c8,
        0x21d1_60ae_b6a1_e799,
        0x2ed0_b283_ed23_7db4,
        0x915c_97f3_6c6f_1821,
        0x67f1_7fcb_de78_3765,
        0x2378_b903_9096_d1b7,
        0x7988_f876_1bdc_51dc,
        0x2076_9950_03fc_77a1,
        0x827e_ca0b_a621_315b,
        0xe5a7_2bce_8d63_cb9f,
        0xf68f_7764_c28b_6f8a,
        0x2f23_0063_cf08_1517,
        0x9450_6632_528d_6a9a,
        0xd3cd_e88e_eb99_6ca3,
        0xc0bd_38c3_195c_899e,
        0x000f_686b_3d80_7d01,
    ];

    fn scalar_limbs(scalar: Scalar) -> [u64; 4] {
        let bytes = scalar.to_bytes_le();
        [
            u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
        ]
    }

    fn random_cyclotomic(rng: &mut OsRng) -> Fp12 {
        loop {
            let value = Fp12::random(&mut *rng);
            if let Some(inverse) = Option::<Fp12>::from(value.invert()) {
                // Easy final exponentiation:
                // value^((p^6 - 1)(p^2 + 1)).
                let mut conjugate = value;
                conjugate.conjugate();
                let easy_first = conjugate * inverse;
                let mut easy_first_p2 = easy_first;
                easy_first_p2.frobenius_map(2);
                return easy_first * easy_first_p2;
            }
        }
    }

    fn hard_project(value: Fp12) -> Fp12 {
        value.pow_vartime(HARD_EXPONENT)
    }

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

    #[test]
    fn scalar_base_x_decomposition_recomposes() {
        let x_abs = Scalar::from(BLS12_381_X_ABS);
        let deterministic = [
            Scalar::ZERO,
            Scalar::ONE,
            Scalar::from(2u64),
            x_abs - Scalar::ONE,
            x_abs,
            x_abs + Scalar::ONE,
            x_abs.square(),
            x_abs.square() * x_abs,
            Scalar::ROOT_OF_UNITY,
            -Scalar::ONE,
        ];

        let mut rng = OsRng;
        for scalar in deterministic
            .into_iter()
            .chain((0..64).map(|_| Scalar::random(&mut rng)))
        {
            let components = decompose_scalar_base_x(scalar);
            assert!(components
                .iter()
                .all(|component| *component < BLS12_381_X_ABS));
            let recomposed = components
                .iter()
                .rev()
                .fold(Scalar::ZERO, |accumulator, component| {
                    accumulator * x_abs + Scalar::from(*component)
                });
            assert_eq!(recomposed, scalar);
        }
    }

    #[test]
    fn alternating_frobenius_bases_are_x_powers_in_gt() {
        let value = Gt::generator() * Scalar::from(0x1234_5678_9abc_def0);
        let value_fp12: Fp12 = value.into();
        let raw: blst_fp12 = value_fp12.into();
        let x_abs = Scalar::from(BLS12_381_X_ABS);
        let mut x_power = Scalar::ONE;

        for power in 0..GT_DECOMPOSITION_COMPONENTS {
            let mut transformed = raw;
            if power != 0 {
                unsafe {
                    blst_fp12_frobenius_map(&mut transformed, &raw, power);
                    if power & 1 == 1 {
                        blst_fp12_conjugate(&mut transformed);
                    }
                }
            }
            assert_eq!(
                Gt::from(Fp12::from(transformed)),
                value * x_power,
                "component {power}"
            );
            x_power *= x_abs;
        }
    }

    #[test]
    fn decomposed_gt_scalar_mul_matches_ordinary_mul() {
        let x_abs = Scalar::from(BLS12_381_X_ABS);
        let deterministic_scalars = [
            Scalar::ZERO,
            Scalar::ONE,
            Scalar::from(2u64),
            x_abs - Scalar::ONE,
            x_abs,
            x_abs + Scalar::ONE,
            x_abs.square(),
            x_abs.square() * x_abs,
            Scalar::ROOT_OF_UNITY,
            -Scalar::ONE,
        ];
        let generator = Gt::generator();
        let deterministic_bases = [
            Gt::identity(),
            generator,
            -generator,
            generator * Scalar::from(42u64),
        ];

        for base in deterministic_bases {
            for scalar in deterministic_scalars {
                assert_eq!(mul_gt(base, scalar), base * scalar);
            }
        }

        let mut rng = OsRng;
        for _ in 0..64 {
            let base = Gt::random(&mut rng);
            let scalar = Scalar::random(&mut rng);
            assert_eq!(mul_gt(base, scalar), base * scalar);
        }

        let base = generator * Scalar::from(29u64);
        for coefficient in [-17i64, -1, 0, 1, 2, 19] {
            let scalar = if coefficient < 0 {
                -Scalar::from(coefficient.unsigned_abs())
            } else {
                Scalar::from(coefficient as u64)
            };
            assert_eq!(mul_gt_signed_small(base, coefficient), base * scalar);
        }
    }

    #[test]
    fn raw_cyclotomic_action_agrees_after_hard_projection() {
        let mut rng = OsRng;
        for _ in 0..4 {
            let value = random_cyclotomic(&mut rng);
            let scalar = Scalar::random(&mut rng);
            let ordinary = value.pow_vartime(scalar_limbs(scalar));
            let value_raw: blst_fp12 = value.into();
            let decomposed = cyclotomic_mul_scalar_decomposed(
                unsafe { CyclotomicFp12::from_raw_unchecked(value_raw) },
                scalar,
            );
            let decomposed = Fp12::from(decomposed.into_raw());

            assert_eq!(hard_project(decomposed), hard_project(ordinary));
        }
    }

    #[test]
    fn specialized_gt_ifft_matches_generic_ifft() {
        let mut rng = OsRng;
        for size in [1, 2, 4, 8, 16, 32, 64] {
            let domain = Radix2Domain::new(size).unwrap();
            let input = (0..size).map(|_| Gt::random(&mut rng)).collect::<Vec<_>>();
            let mut generic = input.clone();
            let mut specialized = input;

            domain.ifft_unscaled(&mut generic);
            domain.ifft_gt_unscaled(&mut specialized);
            assert_eq!(specialized, generic, "size {size}");
        }
    }

    #[test]
    fn specialized_gt_fft_round_trip() {
        let mut rng = OsRng;
        for size in [1, 2, 4, 8, 16, 32, 64] {
            let domain = Radix2Domain::new(size).unwrap();
            let original = (0..size).map(|_| Gt::random(&mut rng)).collect::<Vec<_>>();
            let mut transformed = original.clone();

            domain.fft_gt(&mut transformed);
            domain.ifft_gt_unscaled(&mut transformed);
            transformed
                .iter_mut()
                .for_each(|value| *value *= domain.size_inverse());
            assert_eq!(transformed, original, "size {size}");
        }
    }

    #[test]
    fn cyclotomic_prefix_ifft_matches_full_gt_ifft() {
        let size = 16;
        let domain = Radix2Domain::new(size).unwrap();
        let mut rng = OsRng;
        let miller_results = (0..size)
            .map(|index| {
                let (g1, g2) = match index {
                    0 => (G1Projective::identity(), G2Projective::identity()),
                    1 => (G1Projective::identity(), G2Projective::generator()),
                    2 => (G1Projective::generator(), G2Projective::identity()),
                    3 => (G1Projective::generator(), G2Projective::generator()),
                    4..=7 => (
                        G1Projective::generator() * Scalar::from((index + 1) as u64),
                        G2Projective::generator() * Scalar::from((2 * index + 1) as u64),
                    ),
                    _ => (
                        G1Projective::random(&mut rng),
                        G2Projective::random(&mut rng),
                    ),
                };
                let g1 = g1.to_affine();
                PreparedG2Lines::from_affine(&g2.to_affine()).miller_loop(&g1)
            })
            .collect::<Vec<_>>();

        let easy_inputs = miller_results
            .iter()
            .copied()
            .map(easy_final_exponentiation)
            .collect::<Vec<_>>();
        let hard_inputs = easy_inputs
            .iter()
            .copied()
            .map(hard_final_exponentiation)
            .collect::<Vec<_>>();
        let full_inputs = miller_results
            .iter()
            .copied()
            .map(full_final_exponentiation)
            .collect::<Vec<_>>();
        assert_eq!(hard_inputs, full_inputs);

        let mut expected = hard_inputs;
        domain.ifft_gt_unscaled(&mut expected);
        for output_count in [0, 1, 3, size / 2] {
            let mut actual = easy_inputs.clone();
            domain.ifft_cyclotomic_prefix_unscaled(&mut actual, output_count);
            let actual = actual[..output_count]
                .iter()
                .copied()
                .map(hard_final_exponentiation)
                .collect::<Vec<_>>();
            assert_eq!(
                actual,
                expected[..output_count],
                "output count {output_count}"
            );
        }
    }

    #[test]
    fn parallel_fft_scheduling_matches_single_thread() {
        let size = PARALLEL_FFT_THRESHOLD;
        let domain = Radix2Domain::new(size).unwrap();
        let single_thread = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let four_threads = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let mut rng = OsRng;

        let source_input = (0..size)
            .map(|_| G1Projective::random(&mut rng))
            .collect::<Vec<_>>();
        let mut source_single = source_input.clone();
        let mut source_parallel = source_input;
        single_thread.install(|| domain.fft(&mut source_single));
        four_threads.install(|| domain.fft(&mut source_parallel));
        assert_eq!(source_parallel, source_single);

        let target_input = (0..size).map(|_| Gt::random(&mut rng)).collect::<Vec<_>>();
        let mut target_single = target_input.clone();
        let mut target_parallel = target_input.clone();
        single_thread.install(|| domain.ifft_gt_unscaled(&mut target_single));
        four_threads.install(|| domain.ifft_gt_unscaled(&mut target_parallel));
        assert_eq!(target_parallel, target_single);

        let output_count = size / 2 - 7;
        let cyclotomic_input = target_input
            .into_iter()
            .map(CyclotomicFp12::from_gt)
            .collect::<Vec<_>>();
        let mut cyclotomic_single = cyclotomic_input.clone();
        let mut cyclotomic_parallel = cyclotomic_input;
        single_thread.install(|| {
            domain.ifft_cyclotomic_prefix_unscaled(&mut cyclotomic_single, output_count)
        });
        four_threads.install(|| {
            domain.ifft_cyclotomic_prefix_unscaled(&mut cyclotomic_parallel, output_count)
        });
        assert_eq!(cyclotomic_parallel, cyclotomic_single);
    }
}
