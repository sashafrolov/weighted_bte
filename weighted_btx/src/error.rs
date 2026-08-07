use core::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Error {
    BatchIsEmpty,
    BatchTooLarge {
        batch_size: usize,
        max_batch_size: usize,
    },
    InvalidBatchSize(usize),
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
    InvalidPower {
        power: usize,
        max_power: usize,
    },
    InsufficientWeight {
        accepted: usize,
        required: usize,
        /// Parties whose submitted shares were rejected before authorization
        /// failed. Empty for non-verification callers.
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
            Self::EmptyCommittee => write!(f, "the weighted committee is empty"),
            Self::InvalidPartyWeight {
                party_index,
                weight,
            } => write!(
                f,
                "party {party_index} has invalid weight {weight}; every party must have positive weight"
            ),
            Self::TotalWeightOverflow => {
                write!(f, "the sum of the party weights overflows usize")
            }
            Self::ParameterSizeOverflow => {
                write!(f, "the configured public-key dimensions overflow addressable memory")
            }
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
            Self::InvalidPower { power, max_power } => write!(
                f,
                "invalid key power {power}; the supported positive magnitude is 1 through {max_power}"
            ),
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
            Self::WeightOverflow => {
                write!(f, "accepted decryption weight overflows usize")
            }
            Self::TooManyShares {
                supplied,
                party_count,
            } => write!(
                f,
                "too many decryption shares: supplied {supplied} for {party_count} parties"
            ),
            Self::DuplicatePartyIndex(index) => {
                write!(f, "duplicate party index {index}")
            }
            Self::MismatchedBatchSize { expected, actual } => write!(
                f,
                "mismatched batch size: expected {expected}, got {actual}"
            ),
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
