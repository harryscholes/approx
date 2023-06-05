use std::collections::HashMap;

use ndarray::{arr1, Array2, LinalgScalar};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use num::Float;

#[derive(Clone)]
pub struct RandomProjection<T> {
    plane_norms: Array2<T>,
}

impl<T> RandomProjection<T>
where
    T: Float + LinalgScalar + SampleUniform,
{
    pub fn new(dims: usize, bits: usize) -> Self {
        Self {
            plane_norms: Array2::random(
                (bits, dims),
                Uniform::new(T::from(-0.5).unwrap(), T::from(0.5).unwrap()),
            ),
        }
    }

    pub fn from(plane_norms: Array2<T>) -> Self {
        Self { plane_norms }
    }

    pub fn project(&self, vec: &[T]) -> Vec<T> {
        self.plane_norms.dot(&arr1(vec)).to_vec()
    }

    pub fn hash(&self, vec: &[T]) -> Vec<bool> {
        self.project(vec)
            .into_iter()
            .map(|x| x > T::zero())
            .collect()
    }

    pub fn buckets(&self) -> RandomProjectionBuckets<T> {
        RandomProjectionBuckets::new(self)
    }
}

pub struct RandomProjectionBuckets<T> {
    rp: RandomProjection<T>,
    buckets: HashMap<Vec<bool>, Vec<Vec<T>>>,
}

impl<T> RandomProjectionBuckets<T>
where
    T: Float + LinalgScalar + SampleUniform,
{
    pub fn new(rp: &RandomProjection<T>) -> Self {
        Self {
            rp: rp.clone(),
            buckets: HashMap::new(),
        }
    }

    pub fn insert(&mut self, vec: &[T]) {
        let h = self.rp.hash(vec);
        self.buckets.entry(h).or_insert(vec![]).push(vec.to_vec());
    }

    pub fn get(&self, k: &[bool]) -> Option<&Vec<Vec<T>>> {
        self.buckets.get(k)
    }
}

pub fn hamming_distance(x: &[bool], y: &[bool]) -> usize {
    x.iter().zip(y).filter(|(x_i, y_i)| x_i != y_i).count()
}

#[cfg(test)]
mod tests {
    use ndarray::arr2;

    use super::*;

    #[test]
    fn test_pinecone_example() {
        // Example from:
        // https://www.pinecone.io/learn/locality-sensitive-hashing-random-projection/
        let plane_norms = arr2(&[
            [-0.26623211, 0.34055181],
            [0.3388499, -0.33368453],
            [0.34768572, -0.37184437],
            [-0.11170635, -0.0242341],
        ]);
        let rp = RandomProjection::from(plane_norms);

        let a = vec![1., 2.];
        let b = vec![2., 1.];
        let c = vec![3., 1.];

        let data = vec![a.clone(), b.clone(), c.clone()];

        assert_eq!(
            rp.project(&a),
            vec![0.41487151, -0.32851916, -0.39600302, -0.16017455]
        );
        assert_eq!(rp.hash(&a), vec![true, false, false, false]);
        assert_eq!(rp.hash(&b), vec![false, true, true, false]);
        assert_eq!(rp.hash(&c), vec![false, true, true, false]);

        let mut rp_buckets = rp.buckets();
        for vec in &data {
            rp_buckets.insert(vec);
        }

        assert_eq!(
            rp_buckets.get(&vec![true, false, false, false]),
            Some(&vec![a])
        );
        assert_eq!(
            rp_buckets.get(&vec![false, true, true, false]),
            Some(&vec![b, c])
        );
        assert!(rp_buckets.get(&vec![true, false, false, true]).is_none());
    }

    #[test]
    fn test_hamming_distance() {
        let x = vec![true, false, false, false];
        let y = vec![false, true, true, false];
        assert_eq!(hamming_distance(&x, &y), 3);
    }
}
