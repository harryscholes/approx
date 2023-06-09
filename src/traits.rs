pub trait Train<T> {
    fn train(&mut self, items: &[T]);
}

pub trait Search<T, I>: Train<T> {
    fn search(&self, query: &T, k: usize) -> Vec<I>;
}

pub trait BucketSearch<T, I>: Train<T> {
    fn search(&self, query: &T) -> Vec<I>;
}
