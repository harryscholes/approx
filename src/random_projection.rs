use std::collections::HashMap;

use ndarray::{arr1, Array2, LinalgScalar};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use num::Float;

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

    pub fn project(&self, vec: &[T]) -> Vec<T> {
        self.plane_norms.dot(&arr1(vec)).to_vec()
    }

    pub fn hash(&self, vec: &[T]) -> Vec<bool> {
        self.project(vec)
            .into_iter()
            .map(|x| x > T::zero())
            .collect()
    }

    pub fn bucket(&self, vecs: &[Vec<T>]) -> HashMap<Vec<bool>, Vec<Vec<T>>> {
        let mut map = HashMap::new();
        for vec in vecs {
            let h = self.hash(vec);
            map.entry(h).or_insert(vec![]).push(vec.to_vec());
        }
        map
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
        let rp = RandomProjection {
            plane_norms: arr2(&[
                [-0.26623211, 0.34055181],
                [0.3388499, -0.33368453],
                [0.34768572, -0.37184437],
                [-0.11170635, -0.0242341],
            ]),
        };

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

        let buckets = rp.bucket(&data);
        assert_eq!(buckets[&vec![true, false, false, false]], vec![a]);
        assert_eq!(buckets[&vec![false, true, true, false]], vec![b, c]);
    }

    #[test]
    fn test_hamming_distance() {
        let x = vec![true, false, false, false];
        let y = vec![false, true, true, false];
        assert_eq!(hamming_distance(&x, &y), 3);
    }
}
