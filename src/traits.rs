pub trait Train<'a, T> {
    fn train(&mut self, items: &'a [T]);
}

pub trait Search<'a, T, I>: Train<'a, T> {
    fn search(&self, query: &'a T, k: usize) -> Vec<I>;
}

pub trait BucketSearch<'a, T, I>: Train<'a, T> {
    fn search(&self, query: &'a T) -> Vec<I>;
}
