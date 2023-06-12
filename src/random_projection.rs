use std::collections::HashMap;

use ndarray::{arr1, Array2, LinalgScalar};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};

use crate::{
    error::Error,
    traits::{BucketSearch, Train},
};

pub struct RandomProjection<T> {
    plane_norms: PlaneNorms<T>,
    buckets: Buckets,
    trained: bool,
}

type Id = usize;
type Bucket = Vec<Id>;
type Buckets = HashMap<Vec<bool>, Bucket>;

impl<T> RandomProjection<T> {
    pub fn from_plane_norms(arr: impl Into<PlaneNorms<T>>) -> Self {
        Self {
            plane_norms: arr.into(),
            buckets: HashMap::new(),
            trained: false,
        }
    }
}

impl<T> RandomProjection<T>
where
    T: LinalgScalar + SampleUniform + PartialOrd,
{
    pub fn new(dims: usize, bits: usize, low: T, high: T) -> Self {
        Self::from_plane_norms(PlaneNorms::new(dims, bits, low, high))
    }

    pub fn project(&self, vec: &[T]) -> Vec<T> {
        self.plane_norms.arr.dot(&arr1(vec)).to_vec()
    }

    pub fn hash(&self, vec: &[T]) -> Vec<bool> {
        let projection = self.project(vec);
        projection.into_iter().map(|x| x > T::zero()).collect()
    }

    fn hash_to_bucket(&mut self, vec: &[T], id: Id) {
        self.buckets
            .entry(self.hash(vec))
            .or_insert(vec![])
            .push(id);
    }

    fn get_bucket(&self, hash: &[bool]) -> Option<&Bucket> {
        self.buckets.get(hash)
    }

    pub fn buckets(&self) -> &Buckets {
        &self.buckets
    }

    pub fn plane_norms(&self) -> &PlaneNorms<T> {
        &self.plane_norms
    }
}

impl<T> Train<'_, Vec<T>> for RandomProjection<T>
where
    T: LinalgScalar + SampleUniform + PartialOrd,
{
    fn train(&mut self, vecs: &[Vec<T>]) -> Result<(), Error> {
        if self.trained {
            return Err(Error::ModelAlreadyTrained);
        };

        for (id, vec) in vecs.iter().enumerate() {
            self.hash_to_bucket(vec, id);
        }

        self.trained = true;
        Ok(())
    }
}

impl<T> BucketSearch<'_, Vec<T>, usize> for RandomProjection<T>
where
    T: LinalgScalar + SampleUniform + PartialOrd,
{
    fn search(&self, query: &Vec<T>) -> Result<Vec<Id>, Error> {
        if !self.trained {
            return Err(Error::ModelNotTrained);
        }

        let hash = self.hash(query);
        let ids = match self.get_bucket(&hash) {
            Some(ids) => ids.clone(),
            None => vec![],
        };

        Ok(ids)
    }
}

impl<T> From<Array2<T>> for RandomProjection<T> {
    fn from(arr: Array2<T>) -> Self {
        RandomProjection::from_plane_norms(arr)
    }
}

impl<T> From<PlaneNorms<T>> for RandomProjection<T> {
    fn from(plane_norms: PlaneNorms<T>) -> Self {
        RandomProjection::from_plane_norms(plane_norms)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct PlaneNorms<T> {
    arr: Array2<T>,
}

impl<T> PlaneNorms<T>
where
    T: SampleUniform,
{
    pub fn new(dims: usize, bits: usize, low: T, high: T) -> Self {
        let dbn = Uniform::new(low, high);
        let arr = Array2::random((bits, dims), dbn);
        PlaneNorms { arr }
    }
}

impl<T> From<Array2<T>> for PlaneNorms<T> {
    fn from(arr: Array2<T>) -> Self {
        Self { arr }
    }
}

pub fn hamming_distance(x: &[bool], y: &[bool]) -> usize {
    x.iter().zip(y).filter(|(x_i, y_i)| x_i != y_i).count()
}

#[cfg(test)]
mod tests {
    use ndarray::arr2;
    use serde_json;

    use super::*;

    #[test]
    fn test_pinecone_example() {
        // Example from:
        // https://www.pinecone.io/learn/locality-sensitive-hashing-random-projection/
        let mut rp: RandomProjection<_> = arr2(&[
            [-0.26623211, 0.34055181],
            [0.3388499, -0.33368453],
            [0.34768572, -0.37184437],
            [-0.11170635, -0.0242341],
        ])
        .into();

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

        let vecs = vec![a.clone(), b.clone(), c.clone()];

        assert_eq!(rp.search(&a).unwrap_err(), Error::ModelNotTrained);

        rp.train(&vecs).unwrap();
        assert!(rp.trained);

        assert_eq!(rp.train(&vecs).unwrap_err(), Error::ModelAlreadyTrained);

        assert_eq!(
            rp.get_bucket(&vec![true, false, false, false]),
            Some(&vec![0])
        );
        assert_eq!(
            rp.get_bucket(&vec![false, true, true, false]),
            Some(&vec![1, 2])
        );
        assert!(rp.get_bucket(&vec![true, false, false, true]).is_none());

        assert_eq!(rp.search(&a).unwrap(), vec![0]);
        assert_eq!(rp.search(&b).unwrap(), vec![1, 2]);
        assert_eq!(rp.search(&c).unwrap(), vec![1, 2]);

        let query = vec![2.5, 1.];
        assert_eq!(rp.search(&query).unwrap(), vec![1, 2]);

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

    #[test]
    fn test_serialize_deserialize_i64() {
        let plane_norms = PlaneNorms::new(1024, 100, -1, 1);
        let serialized = serde_json::to_string(&plane_norms).unwrap();
        let deserialized = serde_json::from_str::<PlaneNorms<i64>>(&serialized).unwrap();
        assert_eq!(plane_norms, deserialized);
    }

    #[test]
    fn test_serialize_deserialize_f32() {
        // Note that floating point precision errors occur when serde-ing f64s
        let plane_norms = PlaneNorms::new(1024, 100, -0.5, 0.5);
        let serialized = serde_json::to_string(&plane_norms).unwrap();
        let deserialized = serde_json::from_str::<PlaneNorms<f32>>(&serialized).unwrap();
        assert_eq!(plane_norms, deserialized);
    }
}
