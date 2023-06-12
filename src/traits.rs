use crate::error::Error;

pub trait Train<'a, T> {
    fn train(&mut self, items: &'a [T]) -> Result<(), Error>;
}

pub trait Search<'a, T, I>: Train<'a, T> {
    fn search(&self, query: &'a T, k: usize) -> Result<Vec<I>, Error>;
}

pub trait BucketSearch<'a, T, I>: Train<'a, T> {
    fn search(&self, query: &'a T) -> Result<Vec<I>, Error>;
}
