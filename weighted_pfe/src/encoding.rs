use blstrs::{Compress, Gt};
use group::Group;

const COMPRESSED_GT_BYTES: usize = 6 * 48;

/// Append a canonical, identity-safe GT encoding.
///
/// `blstrs` torus compression is undefined for the identity, so every value
/// receives a tag and a fixed 288-byte body. The identity body is all zero;
/// non-identity bodies contain the canonical torus encoding. Transcript and
/// batch-digest callers must use this helper rather than calling
/// `write_compressed` directly.
pub(crate) fn append_gt(value: &Gt, output: &mut Vec<u8>) {
    if bool::from(value.is_identity()) {
        output.push(0);
        output.resize(output.len() + COMPRESSED_GT_BYTES, 0);
    } else {
        output.push(1);
        let body_start = output.len();
        (*value)
            .write_compressed(&mut *output)
            .expect("writing to a byte vector cannot fail");
        debug_assert_eq!(output.len() - body_start, COMPRESSED_GT_BYTES);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_encoding_is_fixed_width_and_identity_safe() {
        let mut identity = Vec::new();
        append_gt(&Gt::identity(), &mut identity);
        let mut generator = Vec::new();
        append_gt(&Gt::generator(), &mut generator);

        assert_eq!(identity.len(), 1 + COMPRESSED_GT_BYTES);
        assert_eq!(generator.len(), 1 + COMPRESSED_GT_BYTES);
        assert_eq!(identity[0], 0);
        assert_eq!(generator[0], 1);
        assert!(identity[1..].iter().all(|byte| *byte == 0));
        assert_ne!(identity, generator);
    }
}
