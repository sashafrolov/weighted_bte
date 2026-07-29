use core::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Error {
    BatchIsEmpty,
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
            Self::InvalidBatchSize(size) => write!(
                f,
                "PFE batch size {size} must be a supported power of two"
            ),
            Self::InvalidCommittee {
                server_count,
                threshold,
            } => write!(
                f,
                "invalid committee: corruption threshold {threshold} must be smaller than server count {server_count}"
            ),
            Self::InvalidProof => write!(f, "a ciphertext proof is invalid"),
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
