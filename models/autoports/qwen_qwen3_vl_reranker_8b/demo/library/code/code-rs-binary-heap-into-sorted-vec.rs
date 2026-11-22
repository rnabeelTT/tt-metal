    /// Consumes the `BinaryHeap` and returns a vector in sorted
    /// (ascending) order.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use std::collections::BinaryHeap;
    ///
    /// let mut heap = BinaryHeap::from([1, 2, 4, 5, 7]);
    /// heap.push(6);
    /// heap.push(3);
    ///
    /// let vec = heap.into_sorted_vec();
    /// assert_eq!(vec, [1, 2, 3, 4, 5, 6, 7]);
    /// ```
    #[must_use = "`self` will be dropped if the result is not used"]
    #[stable(feature = "binary_heap_extras_15", since = "1.5.0")]
    pub fn into_sorted_vec(mut self) -> Vec<T, A> {
        let mut end = self.len();
        while end > 1 {
            end -= 1;
            // SAFETY: `end` goes from `self.len() - 1` to 1 (both included),
            //  so it's always a valid index to access.
            //  It is safe to access index 0 (i.e. `ptr`), because
            //  1 <= end < self.len(), which means self.len() >= 2.
            unsafe {
                let ptr = self.data.as_mut_ptr();
                ptr::swap(ptr, ptr.add(end));
            }
            // SAFETY: `end` goes from `self.len() - 1` to 1 (both included) so:
            //  0 < 1 <= end <= self.len() - 1 < self.len()
            //  Which means 0 < end and end < self.len().
            unsafe { self.sift_down_range(0, end) };
        }
        self.into_vec()
    }
