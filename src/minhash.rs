use std::{
    collections::{HashMap, HashSet},
    str::CharIndices,
};

use itertools::{Itertools, MultiPeek};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

use crate::{
    error::Error,
    traits::{BucketSearch, Train},
};

#[derive(Debug)]
pub struct Minhash<T> {
    shingle_index: HashMap<T, usize>,
    hash_functions: Vec<HashMap<usize, usize>>,
    bands: HashMap<Vec<usize>, Vec<usize>>,
    n_hashes: usize,
    band_size: usize,
    k: usize,
    rng: StdRng,
    trained: bool,
}

type Id = usize;

impl<'a> Minhash<&'a str> {
    pub fn new(k: usize, n_hashes: usize, band_size: usize) -> Self {
        Self::from_rng(k, n_hashes, band_size, StdRng::from_entropy())
    }

    pub fn from_rng(k: usize, n_hashes: usize, band_size: usize, rng: impl Into<StdRng>) -> Self {
        assert!(n_hashes % band_size == 0);

        Self {
            shingle_index: HashMap::new(),
            hash_functions: vec![],
            bands: HashMap::new(),
            n_hashes,
            k,
            band_size,
            rng: rng.into(),
            trained: false,
        }
    }

    pub fn minhash(&self, text: &str) -> Vec<usize> {
        let shingle_indicator_vec = self.shingle_indicator_vec(text);

        self.hash_functions
            .iter()
            .map(|hf| self.apply_hash_function(hf, &shingle_indicator_vec))
            .collect()
    }

    fn shingle_indicator_vec(&self, text: &str) -> Vec<bool> {
        let mut vec = vec![false; self.shingle_index.len()];

        for shingle in text.shingles(self.k) {
            vec[self.shingle_index[shingle]] = true;
        }

        vec
    }

    fn build_shingle_index(&mut self, corpus: &'a [&str]) {
        self.shingle_index = self
            .build_vocab(corpus)
            .iter()
            .enumerate()
            .map(|(index, shingle)| (*shingle, index))
            .collect()
    }

    fn build_vocab(&self, corpus: &'a [&str]) -> Vec<&'a str> {
        corpus
            .iter()
            .flat_map(|text| text.shingles(self.k))
            .unique()
            .collect()
    }

    fn generate_hash_functions(&mut self) {
        self.hash_functions = (0..self.n_hashes)
            .map(|_| self.generate_hash_function())
            .collect()
    }

    fn generate_hash_function(&mut self) -> HashMap<usize, usize> {
        let iter = 0..self.shingle_index.len();
        let mut perm: Vec<_> = iter.clone().collect();
        perm.shuffle(&mut self.rng);
        iter.zip(perm).collect()
    }

    fn apply_hash_function(
        &self,
        hash_function: &HashMap<usize, usize>,
        shingle_indicator_vec: &[bool],
    ) -> usize {
        for idx in 0..self.shingle_index.len() {
            let permuted_idx = hash_function[&idx];
            if shingle_indicator_vec[permuted_idx] {
                return permuted_idx;
            }
        }

        panic!("Text does not contain any shingles that are in the vocabulary")
    }
}

impl<'a> Train<'a, &str> for Minhash<&'a str> {
    fn train(&mut self, corpus: &'a [&str]) -> Result<(), Error> {
        if self.trained {
            return Err(Error::ModelAlreadyTrained);
        };

        self.build_shingle_index(corpus);
        self.generate_hash_functions();

        let signatures: Vec<_> = corpus.iter().map(|text| self.minhash(text)).collect();

        for (index, signature) in signatures.iter().enumerate() {
            for band in signature.windows(self.band_size) {
                self.bands
                    .entry(band.to_vec())
                    .or_insert(vec![])
                    .push(index)
            }
        }

        self.trained = true;
        Ok(())
    }
}

impl<'a> BucketSearch<'a, &str, Id> for Minhash<&'a str> {
    fn search(&self, query: &&str) -> Result<Vec<Id>, Error> {
        if !self.trained {
            return Err(Error::ModelNotTrained);
        }

        let ids = self
            .minhash(query)
            .windows(self.band_size)
            .flat_map(|band| self.bands.get(&band.to_vec()).unwrap_or(&vec![]).to_vec())
            .sorted()
            .dedup()
            .collect();

        Ok(ids)
    }
}

trait Shingles<'a> {
    fn shingles(&self, k: usize) -> ShingleIter<'a>;
}

impl<'a> Shingles<'a> for &'a str {
    fn shingles(&self, window_size: usize) -> ShingleIter<'a> {
        ShingleIter::new(self, window_size)
    }
}
struct ShingleIter<'a> {
    s: &'a str,
    char_indices: MultiPeek<CharIndices<'a>>,
    k: usize,
}

impl<'a> ShingleIter<'a> {
    fn new(s: &'a str, k: usize) -> Self {
        assert!(k > 0 && k <= s.len());
        Self {
            s,
            char_indices: s.char_indices().multipeek(),
            k,
        }
    }
}

impl<'a> Iterator for ShingleIter<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        match self.char_indices.next() {
            Some((start, _)) => {
                // Peek at the shingle's chars
                for _ in 0..self.k - 1 {
                    // If any of the chars are `None`, then we have iterated past the final shingle
                    self.char_indices.peek()?;
                }
                // Peek at the char after the shingle's final char
                match self.char_indices.peek() {
                    Some((end, _)) => Some(&self.s[start..*end]),
                    // If there is no next char, then index to the end of the string
                    None => Some(&self.s[start..]),
                }
            }
            None => None,
        }
    }
}

pub fn jaccard(x: &[usize], y: &[usize]) -> f64 {
    let x: HashSet<_> = x.iter().collect();
    let y: HashSet<_> = y.iter().collect();
    x.intersection(&y).count() as f64 / x.union(&y).count() as f64
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_pinecone_example() {
        // Example from:
        // https://www.pinecone.io/learn/locality-sensitive-hashing/
        let a = "flying fish flew by the space station";
        let b = "we will not allow you to bring your sticks of dynamite and pet armadillo";
        let c = "he figured a few sticks of dynamite were easier than a fishing pole to catch an armadillo";

        // let corpus = vec![a.clone(), b.clone(), c.clone()];
        let corpus = vec![a, b, c];
        let k = 2;
        let n_hashes = 20;
        let band_size = 2;
        let rng = StdRng::seed_from_u64(42);
        let mut mh = Minhash::from_rng(k, n_hashes, band_size, rng);

        assert_eq!(mh.search(&a).unwrap_err(), Error::ModelNotTrained);

        mh.train(&corpus).unwrap();
        assert!(mh.trained);

        assert_eq!(mh.train(&corpus).unwrap_err(), Error::ModelAlreadyTrained);

        let a_sig = mh.minhash(&a);
        let b_sig = mh.minhash(&b);
        let c_sig = mh.minhash(&c);

        let a_b = jaccard(&a_sig, &b_sig);
        let a_c = jaccard(&a_sig, &c_sig);
        let b_c = jaccard(&b_sig, &c_sig);
        assert!(a_b < a_c && a_c < b_c);

        assert_eq!(mh.search(&a).unwrap(), vec![0]);
        assert_eq!(mh.search(&b).unwrap(), vec![1, 2]);
        assert_eq!(mh.search(&c).unwrap(), vec![1, 2]);
    }

    #[test]
    fn test_shingles_k_1() {
        let s = "abc";
        let mut iter = s.shingles(1);
        assert_eq!(iter.next(), Some("a"));
        assert_eq!(iter.next(), Some("b"));
        assert_eq!(iter.next(), Some("c"));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_shingles_k_2() {
        let s = "abcd";
        let mut iter = s.shingles(2);
        assert_eq!(iter.next(), Some("ab"));
        assert_eq!(iter.next(), Some("bc"));
        assert_eq!(iter.next(), Some("cd"));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_shingles_k_3() {
        let s = "abcde";
        let mut iter = s.shingles(3);
        assert_eq!(iter.next(), Some("abc"));
        assert_eq!(iter.next(), Some("bcd"));
        assert_eq!(iter.next(), Some("cde"));
        assert_eq!(iter.next(), None);

        let s = "👌🤙🤣😘";
        let mut iter = s.shingles(3);
        assert_eq!(iter.next(), Some("👌🤙🤣"));
        assert_eq!(iter.next(), Some("🤙🤣😘"));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_shingles_k_s() {
        let s = "abc";
        let mut iter = s.shingles(3);
        assert_eq!(iter.next(), Some("abc"));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_trained_error_handling() {
        let mut mh = Minhash::new(1, 1, 1);
        assert_eq!(mh.search(&"query").unwrap_err(), Error::ModelNotTrained);
        mh.train(&["first time"]).unwrap();
        assert_eq!(
            mh.train(&["second time"]).unwrap_err(),
            Error::ModelAlreadyTrained
        );
    }
}
