use core::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Error {
    BatchIsEmpty,
    BatchTooLarge {
        batch_size: usize,
        max_batch_size: usize,
    },
    InvalidBatchSize(usize),
    InvalidIndexSpace(usize),
    InvalidCiphertextIndex(usize),
    DuplicateCiphertextIndex(usize),
    EmptyCommittee,
    InvalidPartyWeight {
        party_index: usize,
        weight: usize,
    },
    TotalWeightOverflow,
    ParameterSizeOverflow,
    InvalidEvaluationDomain {
        total_weight: usize,
    },
    InvalidInterpolationSet,
    InvalidCommittee {
        party_count: usize,
        total_weight: usize,
        threshold_weight: usize,
    },
    InvalidPartyIndex(usize),
    InsufficientWeight {
        accepted: usize,
        required: usize,
        rejected_parties: Vec<usize>,
    },
    WeightOverflow,
    TooManyShares {
        supplied: usize,
        party_count: usize,
    },
    DuplicatePartyIndex(usize),
    MismatchedBatchSize {
        expected: usize,
        actual: usize,
    },
    MismatchedMessageLength,
    MismatchedBatchDigest,
    MismatchedSetup,
    MismatchedCommittee,
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
            Self::InvalidIndexSpace(size) => write!(
                f,
                "index space size {size} is invalid; the construction requires at least two indices"
            ),
            Self::InvalidCiphertextIndex(index) => {
                write!(f, "ciphertext index {index} is outside the configured index space")
            }
            Self::DuplicateCiphertextIndex(index) => {
                write!(f, "ciphertext index {index} occurs more than once in the valid batch")
            }
            Self::EmptyCommittee => write!(f, "the weighted committee is empty"),
            Self::InvalidPartyWeight {
                party_index,
                weight,
            } => write!(
                f,
                "party {party_index} has invalid weight {weight}; every represented party must have positive weight"
            ),
            Self::TotalWeightOverflow => write!(f, "the sum of party weights overflows usize"),
            Self::ParameterSizeOverflow => write!(
                f,
                "the configured public-parameter dimensions overflow addressable memory"
            ),
            Self::InvalidEvaluationDomain { total_weight } => write!(
                f,
                "total weight {total_weight} cannot be represented by a radix-2 scalar-field domain"
            ),
            Self::InvalidInterpolationSet => {
                write!(f, "the interpolation points must be nonempty and distinct")
            }
            Self::InvalidCommittee {
                party_count,
                total_weight,
                threshold_weight,
            } => write!(
                f,
                "invalid weighted committee of {party_count} parties: threshold weight {threshold_weight} must be smaller than total weight {total_weight}"
            ),
            Self::InvalidPartyIndex(index) => write!(f, "invalid party index {index}"),
            Self::InsufficientWeight {
                accepted,
                required,
                rejected_parties,
            } => {
                write!(
                    f,
                    "insufficient decryption weight: accepted {accepted}, require {required}"
                )?;
                if !rejected_parties.is_empty() {
                    write!(f, "; rejected parties {rejected_parties:?}")?;
                }
                Ok(())
            }
            Self::WeightOverflow => write!(f, "accepted decryption weight overflows usize"),
            Self::TooManyShares {
                supplied,
                party_count,
            } => write!(
                f,
                "too many decryption shares: supplied {supplied} for {party_count} parties"
            ),
            Self::DuplicatePartyIndex(index) => write!(f, "duplicate party index {index}"),
            Self::MismatchedBatchSize { expected, actual } => write!(
                f,
                "mismatched batch size: expected {expected}, got {actual}"
            ),
            Self::MismatchedMessageLength => {
                write!(f, "ciphertext payload lengths do not match the requested operation")
            }
            Self::MismatchedBatchDigest => {
                write!(f, "decryption material belongs to a different ciphertext batch")
            }
            Self::MismatchedSetup => {
                write!(f, "cryptographic material belongs to a different setup")
            }
            Self::MismatchedCommittee => {
                write!(f, "decryption material belongs to a different accepted committee")
            }
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = core::result::Result<T, Error>;
