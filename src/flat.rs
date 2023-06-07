use num::traits::NumOps;

use crate::traits::ANNModel;

pub struct Flat<T> {
    data: Vec<(Vec<T>, Id)>,
    id: Id,
}

type Id = usize;

impl<T> Flat<T> {
    pub fn new() -> Self {
        Self {
            data: vec![],
            id: 0,
        }
    }
}

impl<T> ANNModel<T, Id> for Flat<T>
where
    T: Copy + Clone + NumOps<T, T> + std::iter::Sum + Into<f64>,
{
    fn train(&mut self, vec: &[T]) -> Id {
        let id = self.id;
        self.data.push((vec.to_vec(), id));
        self.id += 1;
        id
    }

    fn search(&self, query: &[T], k: usize) -> Vec<Id> {
        let mut distances: Vec<_> = self
            .data
            .iter()
            .map(|(x, id)| (euclidean_distance(query, x), id))
            .filter(|(d, _)| !d.is_nan())
            .collect();
        distances.sort_by(|(x, _), (y, _)| x.partial_cmp(y).unwrap());
        distances
            .into_iter()
            .take(k)
            .map(|(_dist, id)| *id)
            .collect()
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
            let diff_2 = diff * diff;
            diff_2
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
        assert!(f.search(&query, 10).is_empty());

        let a_id = f.train(&a);
        let b_id = f.train(&b);
        let c_id = f.train(&c);

        let query = vec![1.1, 1.9];
        assert_eq!(f.search(&query, 1), vec![a_id.clone()]);
        assert_eq!(f.search(&query, 2), vec![a_id.clone(), b_id.clone()]);
        assert_eq!(
            f.search(&query, 3),
            vec![a_id.clone(), b_id.clone(), c_id.clone()]
        );

        let query = vec![1.9, 1.5];
        assert_eq!(f.search(&query, 1), vec![b_id.clone()]);
        assert_eq!(f.search(&query, 2), vec![b_id.clone(), a_id.clone()]);
        assert_eq!(
            f.search(&query, 3),
            vec![b_id.clone(), a_id.clone(), c_id.clone()]
        );
    }
}
