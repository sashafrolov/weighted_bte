//! Fast Lagrange coefficients for subsets of the weighted Shamir domain.
//!
//! Setup assigns virtual shares to a prefix of a radix-2 root-of-unity
//! domain.  Small accepted sets use the direct quadratic formula.  For larger
//! sets, we form the vanishing polynomial
//!
//! `P(X) = product_i (X - x_i)`
//!
//! with a parallel product tree, evaluate `P'` over the enclosing FFT domain,
//! and use `L_i(0) = -P(0) / (x_i P'(x_i))`.

use blstrs::Scalar;
use ff::{BatchInvert, Field};
use rayon::prelude::*;

use crate::{error::Error, fft::Radix2Domain, Result};

/// Below this size, the direct formula is faster and avoids product-tree
/// allocations.  The fast-path tests intentionally straddle this boundary.
const DIRECT_INTERPOLATION_CUTOFF: usize = 64;

/// Polynomial products at the bottom of the tree are cheaper as ordinary
/// coefficient convolutions than as three FFTs.
const NAIVE_MULTIPLICATION_CUTOFF: usize = 32;

/// Compute Lagrange-at-zero coefficients in `selected_indices` order.
///
/// `domain` must be the prefix used by setup: the first `domain.len()` points
/// of the radix-2 domain of size `domain.len().next_power_of_two()`.  The
/// caller owns that invariant; this function validates the selected index set.
pub(crate) fn lagrange_at_zero(
    domain: &[Scalar],
    selected_indices: &[usize],
) -> Result<Vec<Scalar>> {
    validate_selection(domain.len(), selected_indices)?;

    // If every point of a complete M-th-root domain is selected, then
    // P(X) = X^M - 1 and every coefficient at zero is 1/M.  Handling this
    // directly also avoids trying to represent the degree-M product in an
    // M-coefficient FFT buffer.
    if domain.len().is_power_of_two() && selected_indices.len() == domain.len() {
        let size_inverse = Option::<Scalar>::from(Scalar::from(domain.len() as u64).invert())
            .ok_or(Error::InvalidInterpolationSet)?;
        return Ok(vec![size_inverse; selected_indices.len()]);
    }

    if selected_indices.len() <= DIRECT_INTERPOLATION_CUTOFF {
        return direct_lagrange_at_zero(domain, selected_indices);
    }

    fast_lagrange_at_zero(domain, selected_indices)
}

fn validate_selection(domain_len: usize, selected_indices: &[usize]) -> Result<()> {
    if domain_len == 0 || selected_indices.is_empty() {
        return Err(Error::InvalidInterpolationSet);
    }

    let mut seen = vec![false; domain_len];
    for &index in selected_indices {
        if index >= domain_len || seen[index] {
            return Err(Error::InvalidInterpolationSet);
        }
        seen[index] = true;
    }
    Ok(())
}

fn direct_lagrange_at_zero(domain: &[Scalar], selected_indices: &[usize]) -> Result<Vec<Scalar>> {
    let points = selected_indices
        .iter()
        .map(|&index| domain[index])
        .collect::<Vec<_>>();
    let mut numerators = Vec::with_capacity(points.len());
    let mut denominators = Vec::with_capacity(points.len());

    for (point_index, point) in points.iter().enumerate() {
        let mut numerator = Scalar::ONE;
        let mut denominator = Scalar::ONE;
        for (other_index, other) in points.iter().enumerate() {
            if point_index != other_index {
                numerator *= -*other;
                denominator *= *point - other;
            }
        }
        numerators.push(numerator);
        denominators.push(denominator);
    }

    if denominators.iter().any(|value| bool::from(value.is_zero())) {
        return Err(Error::InvalidInterpolationSet);
    }
    denominators.iter_mut().batch_invert();
    Ok(numerators
        .into_iter()
        .zip(denominators)
        .map(|(numerator, denominator_inverse)| numerator * denominator_inverse)
        .collect())
}

fn fast_lagrange_at_zero(domain: &[Scalar], selected_indices: &[usize]) -> Result<Vec<Scalar>> {
    let enclosing_size = domain
        .len()
        .checked_next_power_of_two()
        .ok_or(Error::InvalidInterpolationSet)?;
    let evaluation_domain =
        Radix2Domain::new(enclosing_size).map_err(|_| Error::InvalidInterpolationSet)?;

    let leaves = selected_indices
        .par_iter()
        .map(|&index| vec![-domain[index], Scalar::ONE])
        .collect::<Vec<_>>();
    let product = product_tree(leaves)?;
    debug_assert_eq!(product.len(), selected_indices.len() + 1);

    let value_at_zero = product[0];
    let mut derivative_evaluations = vec![Scalar::ZERO; enclosing_size];
    for (degree, coefficient) in product.iter().copied().enumerate().skip(1) {
        derivative_evaluations[degree - 1] = coefficient * Scalar::from(degree as u64);
    }
    evaluation_domain.fft(&mut derivative_evaluations);

    let mut inverse_denominators = selected_indices
        .iter()
        .map(|&index| domain[index] * derivative_evaluations[index])
        .collect::<Vec<_>>();
    if inverse_denominators
        .iter()
        .any(|value| bool::from(value.is_zero()))
    {
        return Err(Error::InvalidInterpolationSet);
    }
    inverse_denominators.iter_mut().batch_invert();

    let negative_value_at_zero = -value_at_zero;
    Ok(inverse_denominators
        .into_iter()
        .map(|inverse| negative_value_at_zero * inverse)
        .collect())
}

fn product_tree(mut level: Vec<Vec<Scalar>>) -> Result<Vec<Scalar>> {
    debug_assert!(!level.is_empty());
    while level.len() > 1 {
        level = level
            .par_chunks(2)
            .map(|pair| {
                if pair.len() == 1 {
                    Ok(pair[0].clone())
                } else {
                    polynomial_multiply(&pair[0], &pair[1])
                }
            })
            .collect::<Result<Vec<_>>>()?;
    }
    Ok(level.pop().expect("a nonempty product tree has a root"))
}

fn polynomial_multiply(left: &[Scalar], right: &[Scalar]) -> Result<Vec<Scalar>> {
    let result_len = left
        .len()
        .checked_add(right.len())
        .and_then(|sum| sum.checked_sub(1))
        .ok_or(Error::InvalidInterpolationSet)?;

    if left.len().min(right.len()) <= NAIVE_MULTIPLICATION_CUTOFF {
        let mut result = vec![Scalar::ZERO; result_len];
        for (left_degree, left_coefficient) in left.iter().copied().enumerate() {
            for (right_degree, right_coefficient) in right.iter().copied().enumerate() {
                result[left_degree + right_degree] += left_coefficient * right_coefficient;
            }
        }
        return Ok(result);
    }

    let transform_size = result_len
        .checked_next_power_of_two()
        .ok_or(Error::InvalidInterpolationSet)?;
    let transform =
        Radix2Domain::new(transform_size).map_err(|_| Error::InvalidInterpolationSet)?;
    let mut left_values = vec![Scalar::ZERO; transform_size];
    let mut right_values = vec![Scalar::ZERO; transform_size];
    left_values[..left.len()].copy_from_slice(left);
    right_values[..right.len()].copy_from_slice(right);

    rayon::join(
        || transform.fft(&mut left_values),
        || transform.fft(&mut right_values),
    );
    left_values
        .par_iter_mut()
        .zip(right_values.par_iter())
        .for_each(|(left_value, right_value)| *left_value *= right_value);
    transform.ifft(&mut left_values);
    left_values.truncate(result_len);
    Ok(left_values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_core::OsRng;

    fn setup_domain(weight: usize) -> Vec<Scalar> {
        let size = weight.next_power_of_two();
        let transform = Radix2Domain::new(size).unwrap();
        let mut points = vec![Scalar::ZERO; size];
        if size == 1 {
            points[0] = Scalar::ONE;
        } else {
            points[1] = Scalar::ONE;
            transform.fft(&mut points);
        }
        points.truncate(weight);
        points
    }

    fn reference(domain: &[Scalar], selected: &[usize]) -> Vec<Scalar> {
        let points = selected
            .iter()
            .map(|&index| domain[index])
            .collect::<Vec<_>>();
        let mut numerators = Vec::with_capacity(points.len());
        let mut denominators = Vec::with_capacity(points.len());
        for (point_index, point) in points.iter().enumerate() {
            let mut numerator = Scalar::ONE;
            let mut denominator = Scalar::ONE;
            for (other_index, other) in points.iter().enumerate() {
                if point_index != other_index {
                    numerator *= -*other;
                    denominator *= *point - other;
                }
            }
            numerators.push(numerator);
            denominators.push(denominator);
        }
        denominators.iter_mut().batch_invert();
        numerators
            .into_iter()
            .zip(denominators)
            .map(|(numerator, inverse)| numerator * inverse)
            .collect()
    }

    fn evaluate(coefficients: &[Scalar], point: Scalar) -> Scalar {
        coefficients
            .iter()
            .rev()
            .fold(Scalar::ZERO, |value, coefficient| {
                value * point + coefficient
            })
    }

    #[test]
    fn rejects_empty_duplicate_and_out_of_range_selections() {
        let domain = setup_domain(7);
        assert!(matches!(
            lagrange_at_zero(&[], &[0]),
            Err(Error::InvalidInterpolationSet)
        ));
        assert!(matches!(
            lagrange_at_zero(&domain, &[]),
            Err(Error::InvalidInterpolationSet)
        ));
        assert!(matches!(
            lagrange_at_zero(&domain, &[1, 1]),
            Err(Error::InvalidInterpolationSet)
        ));
        assert!(matches!(
            lagrange_at_zero(&domain, &[domain.len()]),
            Err(Error::InvalidInterpolationSet)
        ));
    }

    #[test]
    fn exhaustive_irregular_small_domain_matches_reference_in_both_orders() {
        let domain = setup_domain(7);
        for mask in 1usize..(1 << domain.len()) {
            let selected = (0..domain.len())
                .filter(|index| mask & (1 << index) != 0)
                .collect::<Vec<_>>();
            let mut reversed = selected.clone();
            reversed.reverse();
            for order in [&selected, &reversed] {
                assert_eq!(
                    lagrange_at_zero(&domain, order).unwrap(),
                    reference(&domain, order),
                    "mask {mask:#x}, order {order:?}"
                );
            }
        }
    }

    #[test]
    fn fft_path_matches_reference_for_irregular_domains_and_orders() {
        for weight in [65usize, 73, 95, 127, 129, 191] {
            let domain = setup_domain(weight);
            let mut selections = vec![
                (0..weight).collect::<Vec<_>>(),
                (0..weight).filter(|index| index % 3 != 1).collect(),
                (0..weight).filter(|index| index % 4 != 2).collect(),
            ];
            for selected in &mut selections {
                if selected.len() <= DIRECT_INTERPOLATION_CUTOFF {
                    let needed = DIRECT_INTERPOLATION_CUTOFF + 1 - selected.len();
                    let additions = (0..weight)
                        .filter(|index| !selected.contains(index))
                        .take(needed)
                        .collect::<Vec<_>>();
                    selected.extend(additions);
                }
                let rotation = weight % selected.len();
                selected.rotate_left(rotation);
                selected.reverse();
                assert_eq!(
                    lagrange_at_zero(&domain, selected).unwrap(),
                    reference(&domain, selected),
                    "weight {weight}, selected {}",
                    selected.len()
                );
            }
        }
    }

    #[test]
    fn complete_root_domain_has_uniform_coefficients_in_any_order() {
        for size in [1usize, 2, 64, 128, 256] {
            let domain = setup_domain(size);
            let mut selected = (0..size).collect::<Vec<_>>();
            selected.rotate_left(size / 3);
            selected.reverse();
            let expected = Option::<Scalar>::from(Scalar::from(size as u64).invert()).unwrap();
            assert_eq!(
                lagrange_at_zero(&domain, &selected).unwrap(),
                vec![expected; size]
            );
        }
    }

    #[test]
    fn coefficients_reconstruct_random_low_degree_polynomials() {
        let mut rng = OsRng;
        for (weight, selected_count) in [(17usize, 11usize), (97, 65), (151, 103)] {
            let domain = setup_domain(weight);
            let mut selected = (0..weight)
                .filter(|index| (index * 5 + 3) % 7 != 0)
                .take(selected_count)
                .collect::<Vec<_>>();
            assert_eq!(selected.len(), selected_count);
            selected.rotate_left(selected_count / 4);
            selected.reverse();
            let coefficients = lagrange_at_zero(&domain, &selected).unwrap();

            for degree_bound in [1usize, selected_count / 2, selected_count] {
                let polynomial = (0..degree_bound)
                    .map(|_| Scalar::random(&mut rng))
                    .collect::<Vec<_>>();
                let reconstructed = selected
                    .iter()
                    .zip(coefficients.iter())
                    .map(|(&index, coefficient)| evaluate(&polynomial, domain[index]) * coefficient)
                    .fold(Scalar::ZERO, |sum, value| sum + value);
                assert_eq!(reconstructed, polynomial[0]);
            }
        }
    }
}
