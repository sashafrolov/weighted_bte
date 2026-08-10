use core::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Error {
    BatchIsEmpty,
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
    InvalidProof,
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
    MismatchedBatchDigest,
    MismatchedSetup,
    MismatchedCommittee,
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BatchIsEmpty => write!(formatter, "the ciphertext batch is empty"),
            Self::InvalidBatchSize(size) => write!(
                formatter,
                "batch size {size} must be a nonzero radix-2 FFT-domain size"
            ),
            Self::EmptyCommittee => write!(formatter, "the weighted committee is empty"),
            Self::InvalidPartyWeight {
                party_index,
                weight,
            } => write!(
                formatter,
                "party {party_index} has invalid weight {weight}; every represented party must have positive weight"
            ),
            Self::TotalWeightOverflow => {
                write!(formatter, "the sum of the party weights overflows usize")
            }
            Self::ParameterSizeOverflow => write!(
                formatter,
                "the configured public-key dimensions overflow addressable memory"
            ),
            Self::InvalidEvaluationDomain { total_weight } => write!(
                formatter,
                "total weight {total_weight} cannot be represented by a radix-2 scalar-field domain"
            ),
            Self::InvalidInterpolationSet => write!(
                formatter,
                "the interpolation points must be nonempty, distinct, and in range"
            ),
            Self::InvalidCommittee {
                party_count,
                total_weight,
                threshold_weight,
            } => write!(
                formatter,
                "invalid weighted committee of {party_count} parties: threshold weight {threshold_weight} must be smaller than total weight {total_weight}"
            ),
            Self::InvalidPartyIndex(index) => write!(formatter, "invalid party index {index}"),
            Self::InvalidProof => write!(formatter, "the ciphertext batch contains an invalid proof"),
            Self::InsufficientWeight {
                accepted,
                required,
                rejected_parties,
            } => {
                write!(
                    formatter,
                    "insufficient decryption weight: accepted {accepted}, require {required}"
                )?;
                if !rejected_parties.is_empty() {
                    write!(formatter, "; rejected parties {rejected_parties:?}")?;
                }
                Ok(())
            }
            Self::WeightOverflow => {
                write!(formatter, "accepted decryption weight overflows usize")
            }
            Self::TooManyShares {
                supplied,
                party_count,
            } => write!(
                formatter,
                "too many decryption shares: supplied {supplied} for {party_count} parties"
            ),
            Self::DuplicatePartyIndex(index) => {
                write!(formatter, "duplicate party index {index}")
            }
            Self::MismatchedBatchSize { expected, actual } => write!(
                formatter,
                "mismatched batch size: expected {expected}, got {actual}"
            ),
            Self::MismatchedBatchDigest => write!(
                formatter,
                "decryption material belongs to a different ciphertext batch"
            ),
            Self::MismatchedSetup => write!(
                formatter,
                "cryptographic material belongs to a different setup"
            ),
            Self::MismatchedCommittee => write!(
                formatter,
                "decryption material belongs to a different accepted committee"
            ),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = core::result::Result<T, Error>;
