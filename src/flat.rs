use num::traits::real::Real;

use crate::traits::{Search, Train};

pub struct Flat<'a, T> {
    data: Vec<&'a T>,
}

type Id = usize;

impl<'a, T> Flat<'a, T> {
    pub fn new() -> Self {
        Self { data: vec![] }
    }
}

impl<'a, T> Train<'a, T> for Flat<'a, T> {
    fn train(&mut self, vecs: &'a [T]) {
        self.data.extend(vecs);
    }
}

impl<'a, T, I: 'a> Search<'a, T, Id> for Flat<'a, T>
where
    &'a T: IntoIterator<Item = &'a I>,
    I: Real + std::iter::Sum,
{
    fn search(&self, query: &'a T, k: usize) -> Vec<Id> {
        let mut distances: Vec<_> = self
            .data
            .iter()
            .enumerate()
            .map(|(id, vec)| (euclidean_distance(query, vec), id))
            .collect();
        distances.sort_by(|(x, _), (y, _)| x.partial_cmp(y).unwrap());
        distances
            .into_iter()
            .take(k)
            .map(|(_dist, id)| id)
            .collect()
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

        let query = vec![1.1, 1.9];
        assert!(f.search(&query, 10).is_empty());

        let vecs = vec![a, b, c];
        f.train(&vecs);

        let query = vec![1.1, 1.9];
        assert_eq!(f.search(&query, 1), vec![0]);
        assert_eq!(f.search(&query, 2), vec![0, 1]);
        assert_eq!(f.search(&query, 3), vec![0, 1, 2]);

        let query = vec![1.9, 1.5];
        assert_eq!(f.search(&query, 1), vec![1]);
        assert_eq!(f.search(&query, 2), vec![1, 0]);
        assert_eq!(f.search(&query, 3), vec![1, 0, 2]);
    }
}
