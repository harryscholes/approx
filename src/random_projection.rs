use std::collections::HashMap;

use ndarray::{arr1, Array2, LinalgScalar};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use num::{Float, ToPrimitive};

type Buckets<T> = HashMap<Vec<bool>, Vec<Vec<T>>>;

pub struct RandomProjection<T, F> {
    plane_norms: Array2<F>,
    buckets: Buckets<T>,
}

impl<T, F> RandomProjection<T, F>
where
    T: Clone + ToPrimitive,
    F: Float + LinalgScalar + SampleUniform,
{
    pub fn new(dims: usize, bits: usize, low: F, high: F) -> Self {
        Self::from_plane_norms(Array2::random((bits, dims), Uniform::new(low, high)))
    }

    pub fn from_plane_norms(plane_norms: Array2<F>) -> Self {
        Self {
            plane_norms,
            buckets: HashMap::new(),
        }
    }

    pub fn project(&self, vec: &[T]) -> Result<Vec<F>, Error> {
        let f_vec: Result<Vec<F>, Error> = vec
            .iter()
            .cloned()
            .map(|x| F::from(x).ok_or(Error::TypeConversionError))
            .collect();
        let f_vec = f_vec?;
        Ok(self.plane_norms.dot(&arr1(&f_vec)).to_vec())
    }

    pub fn hash(&self, vec: &[T]) -> Result<Vec<bool>, Error> {
        let projection = self.project(vec)?;
        let hash = projection.into_iter().map(|x| x > F::zero()).collect();
        Ok(hash)
    }

    pub fn hash_to_bucket(&mut self, vec: &[T]) -> Result<Vec<bool>, Error> {
        let hash = self.hash(vec)?;
        self.insert_into_bucket(hash.clone(), vec.to_vec());
        Ok(hash)
    }

    fn insert_into_bucket(&mut self, hash: Vec<bool>, vec: Vec<T>) {
        self.buckets.entry(hash).or_insert(vec![]).push(vec);
    }

    pub fn get_bucket(&self, hash: &[bool]) -> Option<&Vec<Vec<T>>> {
        self.buckets.get(hash)
    }

    pub fn nearest_neighbours(&self, vec: &[T]) -> Result<Option<&Vec<Vec<T>>>, Error> {
        let hash = self.hash(vec)?;
        Ok(self.get_bucket(&hash))
    }

    pub fn buckets(&self) -> &Buckets<T> {
        &self.buckets
    }
}

#[derive(Debug)]
pub enum Error {
    TypeConversionError,
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
            rp.project(&a).unwrap(),
            vec![0.41487151, -0.32851916, -0.39600302, -0.16017455]
        );
        assert_eq!(rp.hash(&a).unwrap(), vec![true, false, false, false]);
        assert_eq!(rp.hash(&b).unwrap(), vec![false, true, true, false]);
        assert_eq!(rp.hash(&c).unwrap(), vec![false, true, true, false]);

        rp.hash_to_bucket(&a).unwrap();
        rp.hash_to_bucket(&b).unwrap();
        rp.hash_to_bucket(&c).unwrap();

        assert_eq!(
            rp.get_bucket(&vec![true, false, false, false]),
            Some(&vec![a])
        );
        assert_eq!(
            rp.get_bucket(&vec![false, true, true, false]),
            Some(&vec![b.clone(), c.clone()])
        );
        assert!(rp.get_bucket(&vec![true, false, false, true]).is_none());

        assert_eq!(
            rp.nearest_neighbours(&vec![2.5, 1.]).unwrap(),
            Some(&vec![b, c])
        );

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
    fn test_type_conversion_i32_to_f64() {
        let rp = RandomProjection::new(2, 4, -0.5f64, 0.5);
        let v = vec![1i32, 2];
        rp.project(&v).unwrap();
        rp.hash(&v).unwrap();
    }

    #[test]
    fn test_hamming_distance() {
        let x = vec![true, false, false, false];
        let y = vec![false, true, true, false];
        assert_eq!(hamming_distance(&x, &y), 3);
    }
}
