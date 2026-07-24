use core::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Error {
    BatchIsEmpty,
    BatchTooLarge {
        batch_size: usize,
        max_batch_size: usize,
    },
    InvalidBatchSize(usize),
    InvalidCommittee {
        server_count: usize,
        threshold: usize,
    },
    InvalidProof,
    InvalidServerIndex(usize),
    InvalidShare,
    InsufficientShares {
        supplied: usize,
        required: usize,
    },
    DuplicateServerIndex(usize),
    MismatchedBatchSize {
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BatchIsEmpty => write!(f, "the ciphertext batch is empty"),
            Self::BatchTooLarge {
                batch_size,
                max_batch_size,
            } => write!(
                f,
                "batch size {batch_size} exceeds the configured maximum {max_batch_size}"
            ),
            Self::InvalidBatchSize(size) => {
                write!(f, "batch size {size} cannot be represented by the FFT domain")
            }
            Self::InvalidCommittee {
                server_count,
                threshold,
            } => write!(
                f,
                "invalid committee: threshold {threshold} must be smaller than server count {server_count}"
            ),
            Self::InvalidProof => write!(f, "ciphertext proof is invalid"),
            Self::InvalidServerIndex(index) => write!(f, "invalid server index {index}"),
            Self::InvalidShare => write!(f, "decryption share verification failed"),
            Self::InsufficientShares { supplied, required } => write!(
                f,
                "insufficient decryption shares: supplied {supplied}, require {required}"
            ),
            Self::DuplicateServerIndex(index) => {
                write!(f, "duplicate server index {index}")
            }
            Self::MismatchedBatchSize { expected, actual } => write!(
                f,
                "mismatched batch size: expected {expected}, got {actual}"
            ),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = core::result::Result<T, Error>;
