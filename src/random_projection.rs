use std::collections::HashMap;

use ndarray::{arr1, Array2, LinalgScalar};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use crate::traits::ANNModel;

// TODO serde
pub struct RandomProjection<T> {
    plane_norms: Array2<T>,
    buckets: Buckets,
    id: Id,
}

type Id = usize;
type Bucket = Vec<Id>;
type Buckets = HashMap<Vec<bool>, Bucket>;

impl<T> RandomProjection<T>
where
    T: LinalgScalar + SampleUniform + PartialOrd,
{
    pub fn new(dims: usize, bits: usize, low: T, high: T) -> Self {
        let dbn = Uniform::new(low, high);
        let plane_norms = Array2::random((bits, dims), dbn);
        Self::from_plane_norms(plane_norms)
    }

    pub fn from_plane_norms(plane_norms: Array2<T>) -> Self {
        Self {
            plane_norms,
            buckets: HashMap::new(),
            id: 0,
        }
    }

    pub fn project(&self, vec: &[T]) -> Vec<T> {
        self.plane_norms.dot(&arr1(vec)).to_vec()
    }

    pub fn hash(&self, vec: &[T]) -> Vec<bool> {
        let projection = self.project(vec);
        projection.into_iter().map(|x| x > T::zero()).collect()
    }

    fn hash_to_bucket(&mut self, vec: &[T]) -> Id {
        let id = self.id;
        self.buckets
            .entry(self.hash(vec))
            .or_insert(vec![])
            .push(id);
        self.id += 1;
        id
    }

    fn get_bucket(&self, hash: &[bool]) -> Option<&Bucket> {
        self.buckets.get(hash)
    }

    pub fn buckets(&self) -> &Buckets {
        &self.buckets
    }
}

impl<T> ANNModel<T, usize> for RandomProjection<T>
where
    T: LinalgScalar + SampleUniform + PartialOrd,
{
    fn train(&mut self, vec: &[T]) -> Id {
        self.hash_to_bucket(vec)
    }

    fn search(&self, query: &[T], k: usize) -> Vec<Id> {
        let hash = self.hash(query);
        match self.get_bucket(&hash) {
            Some(ids) => ids.iter().take(k).cloned().collect(),
            None => vec![],
        }
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

        let mut rp = RandomProjection::from_plane_norms(plane_norms);

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

        let a_id = rp.hash_to_bucket(&a);
        let b_id = rp.hash_to_bucket(&b);
        let c_id = rp.hash_to_bucket(&c);

        assert_eq!(
            rp.get_bucket(&vec![true, false, false, false]),
            Some(&vec![a_id])
        );
        assert_eq!(
            rp.get_bucket(&vec![false, true, true, false]),
            Some(&vec![b_id, c_id])
        );
        assert!(rp.get_bucket(&vec![true, false, false, true]).is_none());

        let k = 2;
        let query = vec![2.5, 1.];
        assert_eq!(rp.search(&query, k), vec![1, 2]);

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
    fn test_integer_eltypes() {
        let rp = RandomProjection::new(2, 4, -1, 1);
        let v = vec![1, 2];
        rp.project(&v);
        rp.hash(&v);
    }

    #[test]
    fn test_hamming_distance() {
        let x = vec![true, false, false, false];
        let y = vec![false, true, true, false];
        assert_eq!(hamming_distance(&x, &y), 3);
    }
}
