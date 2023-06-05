use std::collections::HashMap;

use ndarray::{arr1, Array2, LinalgScalar};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use num::Float;

type Buckets<T> = HashMap<Vec<bool>, Vec<Vec<T>>>;

pub struct RandomProjection<T> {
    plane_norms: Array2<T>,
    buckets: Buckets<T>,
}

impl<T> RandomProjection<T>
where
    T: Float + LinalgScalar + SampleUniform,
{
    pub fn new(dims: usize, bits: usize) -> Self {
        Self::from(Array2::random(
            (bits, dims),
            Uniform::new(T::from(-0.5).unwrap(), T::from(0.5).unwrap()),
        ))
    }

    pub fn from(plane_norms: Array2<T>) -> Self {
        Self {
            plane_norms,
            buckets: HashMap::new(),
        }
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

    pub fn hash_to_bucket(&mut self, vec: &[T]) -> Vec<bool> {
        let hash = self.hash(vec);
        self.insert_into_bucket(hash.clone(), vec.to_vec());
        hash
    }

    fn insert_into_bucket(&mut self, hash: Vec<bool>, vec: Vec<T>) {
        self.buckets.entry(hash).or_insert(vec![]).push(vec);
    }

    pub fn get_bucket(&self, hash: &[bool]) -> Option<&Vec<Vec<T>>> {
        self.buckets.get(hash)
    }

    pub fn nearest_neighbours(&self, vec: &[T]) -> Option<&Vec<Vec<T>>> {
        self.get_bucket(&self.hash(vec))
    }

    pub fn buckets(&self) -> &Buckets<T> {
        &self.buckets
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

        let mut rp = RandomProjection::from(plane_norms);

        let a = vec![1., 2.];
        let b = vec![2., 1.];
        let c = vec![3., 1.];

        assert_eq!(
            rp.project(&a),
            vec![0.41487151, -0.32851916, -0.39600302, -0.16017455]
        );
        assert_eq!(rp.hash(&a), vec![true, false, false, false]);
        assert_eq!(rp.hash(&b), vec![false, true, true, false]);
        assert_eq!(rp.hash(&c), vec![false, true, true, false]);

        rp.hash_to_bucket(&a);
        rp.hash_to_bucket(&b);
        rp.hash_to_bucket(&c);

        assert_eq!(
            rp.get_bucket(&vec![true, false, false, false]),
            Some(&vec![a])
        );
        assert_eq!(
            rp.get_bucket(&vec![false, true, true, false]),
            Some(&vec![b.clone(), c.clone()])
        );
        assert!(rp.get_bucket(&vec![true, false, false, true]).is_none());

        assert_eq!(rp.nearest_neighbours(&vec![2.5, 1.]), Some(&vec![b, c]));

        let query = vec![true, true, false, false];
        let buckets = rp.buckets();
        let keys: Vec<&Vec<bool>> = buckets.keys().collect();
        let mut distances: Vec<(&Vec<bool>, usize)> = keys
            .iter()
            .map(|bucket| (*bucket, hamming_distance(bucket, &query)))
            .collect();
        distances.sort_by(|a, b| a.1.cmp(&b.1));
        assert_eq!(
            distances,
            vec![
                (&vec![true, false, false, false], 1),
                (&vec![false, true, true, false], 2),
            ]
        );
    }

    #[test]
    fn test_hamming_distance() {
        let x = vec![true, false, false, false];
        let y = vec![false, true, true, false];
        assert_eq!(hamming_distance(&x, &y), 3);
    }
}
