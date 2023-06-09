use std::collections::{HashMap, HashSet};

use itertools::Itertools;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

use crate::traits::{BucketSearch, Train};

#[derive(Debug)]
pub struct Minhash {
    shingle_index: HashMap<String, usize>,
    hash_functions: Vec<HashMap<usize, usize>>,
    bands: HashMap<Vec<usize>, Vec<usize>>,
    n_hashes: usize,
    band_size: usize,
    k: usize,
    rng: StdRng,
    // TODO add trained flag
}

type Id = usize;

impl Minhash {
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

        let chars: Vec<_> = text.chars().collect();

        for window in chars.windows(self.k) {
            let shingle: String = window.into_iter().collect();
            let index = self.shingle_index[&shingle];
            vec[index] = true;
        }

        vec
    }

    fn build_shingle_index(&mut self, corpus: &[String]) {
        self.shingle_index = self
            .build_vocab(corpus)
            .into_iter()
            .enumerate()
            .map(|(index, shingle)| (shingle, index))
            .collect()
    }

    fn build_vocab(&self, corpus: &[String]) -> Vec<String> {
        corpus
            .iter()
            .flat_map(|text| {
                let chars: Vec<_> = text.chars().collect();
                chars
                    .windows(self.k)
                    .map(|window| window.into_iter().collect())
                    .collect::<Vec<_>>()
            })
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
        shingle_indicator_vec: &Vec<bool>,
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

impl Train<String> for Minhash {
    fn train(&mut self, corpus: &[String]) {
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
    }
}

impl BucketSearch<String, Id> for Minhash {
    fn search(&self, query: &String) -> Vec<Id> {
        self.minhash(query)
            .windows(self.band_size)
            .flat_map(|band| self.bands.get(&band.to_vec()).unwrap_or(&vec![]).to_vec())
            .unique()
            .sorted()
            .collect()
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
        let a = "flying fish flew by the space station".to_string();
        let b =
            "we will not allow you to bring your sticks of dynamite and pet armadillo".to_string();
        let c = "he figured a few sticks of dynamite were easier than a fishing pole to catch an armadillo".to_string();

        let corpus = vec![a.clone(), b.clone(), c.clone()];

        let k = 2;
        let n_hashes = 20;
        let band_size = 2;
        let rng = StdRng::seed_from_u64(42);
        let mut mh = Minhash::from_rng(k, n_hashes, band_size, rng);

        mh.train(&corpus);

        let a_sig = mh.minhash(&a);
        let b_sig = mh.minhash(&b);
        let c_sig = mh.minhash(&c);

        let a_b = jaccard(&a_sig, &b_sig);
        let a_c = jaccard(&a_sig, &c_sig);
        let b_c = jaccard(&b_sig, &c_sig);
        assert!(a_b < a_c && a_c < b_c);

        assert_eq!(mh.search(&a), vec![0]);
        assert_eq!(mh.search(&b), vec![1, 2]);
        assert_eq!(mh.search(&c), vec![1, 2]);
    }
}
