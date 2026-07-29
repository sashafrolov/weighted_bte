use blstrs::{Compress, Gt};
use group::Group;

/// Append a canonical, identity-safe GT encoding.
///
/// `blstrs` torus compression is undefined for the identity, so it receives a
/// dedicated tag.
pub(crate) fn append_gt(value: &Gt, output: &mut Vec<u8>) {
    if bool::from(value.is_identity()) {
        output.push(0);
    } else {
        output.push(1);
        (*value)
            .write_compressed(output)
            .expect("writing to a byte vector cannot fail");
    }
}
