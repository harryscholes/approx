use num::traits::NumOps;

use crate::traits::{Search, Train};

pub struct Flat<T> {
    data: Vec<Vec<T>>,
}

type Id = usize;

impl<T> Flat<T> {
    pub fn new() -> Self {
        Self { data: vec![] }
    }
}

impl<T> Train<&[T]> for Flat<T>
where
    T: Copy + Clone + NumOps<T, T> + std::iter::Sum + Into<f64>,
{
    fn train(&mut self, vecs: &[&[T]]) {
        self.data.extend(vecs.iter().map(|s| (*s).to_vec()));
    }
}

impl<T> Search<&[T], Id> for Flat<T>
where
    T: Copy + Clone + NumOps<T, T> + std::iter::Sum + Into<f64>,
{
    fn search(&self, query: &&[T], k: usize) -> Vec<Id> {
        let mut distances: Vec<_> = self
            .data
            .iter()
            .enumerate()
            .map(|(id, vec)| (euclidean_distance(query, vec), id))
            .filter(|(d, _)| !d.is_nan())
            .collect();
        distances.sort_by(|(x, _), (y, _)| x.partial_cmp(y).unwrap());
        distances
            .into_iter()
            .take(k)
            .map(|(_dist, id)| id)
            .collect()
    }
}

impl<T> Default for Flat<T> {
    fn default() -> Self {
        Self::new()
    }
}

fn euclidean_distance<T>(x: &[T], y: &[T]) -> f64
where
    T: Copy + NumOps<T, T> + std::iter::Sum + Into<f64>,
{
    let squared_diff_sum: T = x
        .iter()
        .zip(y.iter())
        .map(|(&x, &y)| {
            let diff = x - y;
            diff * diff
        })
        .sum();
    squared_diff_sum.into().sqrt()
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

        let query = vec![1.1, 1.9];
        assert!(f.search(&query.as_slice(), 10).is_empty());

        let vecs = vec![a.as_slice(), b.as_slice(), c.as_slice()];
        f.train(&vecs);

        let query = vec![1.1, 1.9];
        assert_eq!(f.search(&query.as_slice(), 1), vec![0]);
        assert_eq!(f.search(&query.as_slice(), 2), vec![0, 1]);
        assert_eq!(f.search(&query.as_slice(), 3), vec![0, 1, 2]);

        let query = vec![1.9, 1.5];
        assert_eq!(f.search(&query.as_slice(), 1), vec![1]);
        assert_eq!(f.search(&query.as_slice(), 2), vec![1, 0]);
        assert_eq!(f.search(&query.as_slice(), 3), vec![1, 0, 2]);
    }
}
