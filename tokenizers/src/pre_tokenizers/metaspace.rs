use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::tokenizer::{Decoder, PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior};

fn bytes_char() -> HashMap<u8, char> {
    let mut bs: Vec<u8> = vec![];
    bs.extend(b'!'..=b'~');
    bs.extend(b'\xA1'..=b'\xAC');
    bs.extend(b'\xAE'..=b'\xFF');

    let mut cs: Vec<u32> = bs.iter().map(|i| *i as u32).collect();
    let mut n = 0;

    for b in 0..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(u32::pow(2, 8) + n);
            n += 1;
        }
    }

    bs.into_iter()
        .zip(cs)
        .map(|(f, t)| (f, unsafe { std::char::from_u32_unchecked(t) }))
        .collect()
}

lazy_static! {
    static ref BYTES_CHAR: HashMap<u8, char> = bytes_char();
    static ref CHAR_BYTES: HashMap<char, u8> =
        bytes_char().into_iter().map(|(c, b)| (b, c)).collect();
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Replaces all the whitespaces by the provided meta character and then
/// splits on this character
#[serde(tag = "type", from = "MetaspaceDeserializer")]
pub struct Metaspace {
    replacement: char,
    pub add_prefix_space: bool,
    pub use_byte_level: bool,
    #[serde(skip)]
    str_rep: String,
}

#[doc(hidden)]
#[derive(Deserialize)]
#[serde(tag = "type")]
pub struct MetaspaceDeserializer {
    replacement: char,
    add_prefix_space: bool,
    #[serde(default)]
    use_byte_level: bool,
}

impl From<MetaspaceDeserializer> for Metaspace {
    fn from(v: MetaspaceDeserializer) -> Metaspace {
        let mut obj = Metaspace::new(v.replacement, v.add_prefix_space);
        obj.use_byte_level = v.use_byte_level;
        obj
    }
}

impl Metaspace {
    pub fn new(replacement: char, add_prefix_space: bool) -> Self {
        Self {
            replacement,
            str_rep: replacement.to_string(),
            add_prefix_space,
            use_byte_level: false,
        }
    }

    pub fn with_byte_level(mut self, use_byte_level: bool) -> Self {
        self.use_byte_level = use_byte_level;
        self
    }

    pub fn get_replacement(&self) -> char {
        self.replacement
    }

    pub fn set_replacement(&mut self, replacement: char) {
        self.replacement = replacement;
        self.str_rep = replacement.to_string();
    }
}

impl Default for Metaspace {
    fn default() -> Self {
        Self::new('▁', true)
    }
}

impl PreTokenizer for Metaspace {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, mut normalized| {
            normalized.replace(' ', &self.str_rep)?;
            if self.add_prefix_space && !normalized.get().starts_with(self.replacement) {
                normalized.prepend(&self.str_rep);
            }

            normalized.split(self.replacement, SplitDelimiterBehavior::MergedWithNext)
        })?;
        if self.use_byte_level {
            pretokenized.normalize(|normalized| {
                let s = normalized.get();
                let mut transformations: Vec<(char, isize)> = Vec::with_capacity(s.len());
                let mut i = 0;
                for cur_char in s.chars() {
                    let size = cur_char.len_utf8();
                    let bytes = s[i..i + size].as_bytes();
                    i += size;
                    transformations.extend(
                        bytes
                            .iter()
                            .enumerate()
                            .map(|(i, b)| (BYTES_CHAR[b], if i > 0 { 1 } else { 0 })),
                    );
                }
                normalized.transform(transformations.into_iter(), 0);
                Ok(())
            })?;
        }
        Ok(())
    }
}

impl Decoder for Metaspace {
    fn decode(&self, tokens: Vec<String>) -> Result<String> {
        let detokenized = if self.use_byte_level {
            let toks = tokens
                .into_iter()
                .flat_map(|t| {
                    t.chars()
                        .try_fold(vec![], |mut acc, c| {
                            CHAR_BYTES.get(&c).map(|b| {
                                acc.push(*b);
                                acc
                            })
                        })
                        .unwrap_or_else(|| t.as_bytes().to_vec())
                })
                .collect::<Vec<_>>();
            String::from_utf8_lossy(&toks).into_owned()
        } else {
            tokens.iter().flat_map(|t| t.chars()).collect::<String>()
        };
        Ok(detokenized
            .chars()
            .enumerate()
            .filter_map(|(i, c)| {
                if c == self.replacement {
                    if i == 0 && self.add_prefix_space {
                        None
                    } else {
                        Some(' ')
                    }
                } else {
                    Some(c)
                }
            })
            .collect::<String>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OffsetReferential, OffsetType};

    #[test]
    fn serialization() {
        let metaspace = Metaspace::new('_', true);
        let metaspace_s = r#"{"type":"Metaspace","replacement":"_","add_prefix_space":true,"use_byte_level":false}"#;
        assert_eq!(serde_json::to_string(&metaspace).unwrap(), metaspace_s);
        assert_eq!(
            serde_json::from_str::<Metaspace>(metaspace_s).unwrap(),
            metaspace
        );

        // Also check it can deserialize previous versions
        let metaspace = Metaspace::new('_', true);
        let metaspace_s =
            r#"{"type":"Metaspace","str_rep":"_","replacement":"_","add_prefix_space":true}"#;
        assert_eq!(
            serde_json::from_str::<Metaspace>(metaspace_s).unwrap(),
            metaspace
        );
    }

    #[test]
    fn byte_level() {
        let metaspace = Metaspace::new('_', true).with_byte_level();
        let metaspace_s = r#"{"type":"Metaspace","replacement":"_","add_prefix_space":true,"use_byte_level":true}"#;
        assert_eq!(serde_json::to_string(&metaspace).unwrap(), metaspace_s);
        assert_eq!(
            serde_json::from_str::<Metaspace>(metaspace_s).unwrap(),
            metaspace
        );
    }

    #[test]
    fn basic() {
        let pretok = Metaspace::new('▁', true);
        let mut pretokenized = PreTokenizedString::from("Hey friend!");
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Normalized, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![("▁Hey", (0, 6)), ("▁friend!", (6, 16))]
        );
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![("▁Hey", (0, 3)), ("▁friend!", (3, 11))]
        );
    }

    #[test]
    fn multiple_spaces() {
        let pretok = Metaspace::new('▁', true);
        let mut pretokenized = PreTokenizedString::from("Hey   friend!");
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Normalized, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![
                ("▁Hey", (0, 6)),
                ("▁", (6, 9)),
                ("▁", (9, 12)),
                ("▁friend!", (12, 22)),
            ]
        );
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![
                ("▁Hey", (0, 3)),
                ("▁", (3, 4)),
                ("▁", (4, 5)),
                ("▁friend!", (5, 13)),
            ]
        );
    }

    #[test]
    fn decode() {
        let decoder = Metaspace::new('▁', true);
        let res = decoder
            .decode(vec!["▁Hey".into(), "▁friend!".into()])
            .unwrap();
        assert_eq!(&res, "Hey friend!")
    }
}
