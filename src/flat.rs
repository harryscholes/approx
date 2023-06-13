use num::traits::real::Real;

use crate::{
    error::Error,
    traits::{Search, Train},
};

pub struct Flat<'a, T> {
    data: Vec<&'a T>,
    trained: bool,
}

type Id = usize;

impl<'a, T> Flat<'a, T> {
    pub fn new() -> Self {
        Self {
            data: vec![],
            trained: false,
        }
    }
}

impl<'a, T> Train<'a, T> for Flat<'a, T> {
    fn train(&mut self, vecs: &'a [T]) -> Result<(), Error> {
        if self.trained {
            return Err(Error::ModelAlreadyTrained);
        }

        self.data.extend(vecs);

        self.trained = true;
        Ok(())
    }
}

impl<'a, T, I: 'a> Search<'a, T, Id> for Flat<'a, T>
where
    &'a T: IntoIterator<Item = &'a I>,
    I: Real + std::iter::Sum,
{
    fn search(&self, query: &'a T, k: usize) -> Result<Vec<Id>, Error> {
        if !self.trained {
            return Err(Error::ModelNotTrained);
        }

        let mut distances: Vec<_> = self
            .data
            .iter()
            .enumerate()
            .map(|(id, vec)| (euclidean_distance(query, vec), id))
            .collect();
        distances.sort_by(|(x, _), (y, _)| x.partial_cmp(y).unwrap());
        let ids = distances
            .into_iter()
            .take(k)
            .map(|(_dist, id)| id)
            .collect();

        Ok(ids)
    }
}

impl<'a, T> Default for Flat<'a, T> {
    fn default() -> Self {
        Self::new()
    }
}

fn euclidean_distance<'a, T, I: 'a>(x: &'a T, y: &'a T) -> I
where
    &'a T: IntoIterator<Item = &'a I>,
    I: Real + std::iter::Sum,
{
    let squared_diff_sum: I = x
        .into_iter()
        .zip(y.into_iter())
        .map(|(&x, &y)| {
            let diff = x - y;
            diff * diff
        })
        .sum();
    squared_diff_sum.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let mut f = Flat::new();

        let a = vec![1., 2.];
        let b = vec![2., 1.];
        let c = vec![3., 1.];
        let vecs = vec![a, b, c];

        let query = vec![0., 0.];
        assert_eq!(f.search(&query, 10).unwrap_err(), Error::ModelNotTrained);

        f.train(&vecs).unwrap();
        assert!(f.trained);

        assert_eq!(f.train(&vecs).unwrap_err(), Error::ModelAlreadyTrained);

        let query = vec![1.1, 1.9];
        assert_eq!(f.search(&query, 1).unwrap(), vec![0]);
        assert_eq!(f.search(&query, 2).unwrap(), vec![0, 1]);
        assert_eq!(f.search(&query, 3).unwrap(), vec![0, 1, 2]);

        let query = vec![1.9, 1.5];
        assert_eq!(f.search(&query, 1).unwrap(), vec![1]);
        assert_eq!(f.search(&query, 2).unwrap(), vec![1, 0]);
        assert_eq!(f.search(&query, 3).unwrap(), vec![1, 0, 2]);
    }

    #[test]
    fn test_trained_error_handling() {
        let mut f = Flat::new();
        assert_eq!(f.search(&vec![0.], 1).unwrap_err(), Error::ModelNotTrained);
        let data = vec![vec![0.]];
        f.train(&data).unwrap();
        assert_eq!(f.train(&data).unwrap_err(), Error::ModelAlreadyTrained);
    }
}
