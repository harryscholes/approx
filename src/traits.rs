pub trait ANNModel<T, Id> {
    fn train(&mut self, vec: &[T]) -> Id;

    fn search(&self, query: &[T], k: usize) -> Vec<Id>;
}
