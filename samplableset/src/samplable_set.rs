// MIT License
//
// Copyright (c) 2025 Jai Veilleux
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// use
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg32 as RNGType;

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::Hash;

use crate::BinaryTree;
use crate::HashPropensity;

type GroupIndex = usize;
type InGroupIndex = usize;
type SSetPosition = (GroupIndex, InGroupIndex);

type SSetResult<T> = Result<T, SSetError>;

type PropensityGroup<T> = Vec<(T, f64)>;

// TODO add option for below
// For consistency with original C++ implementation
// All instances share the same rng stream (per thread)
thread_local! {
    static GEN: RefCell<RNGType> = RefCell::new(RNGType::from_seed(rand::random()));
}

/// Private helper function
/// If any instance of SamplableSet calls this,
/// RNG will be updated for **all** instances
// fn seed(seed: u64) {
//     GEN.with(|g|
//     *g.borrow_mut() = RNGType::seed_from_u64(seed)
//     );
// }

enum SSetError {
    NotFound,
    EmptySet,
    WeightOutOfRange { w: f64, min: f64, max: f64 },
    InconsistentState(&'static str),
}

#[derive(Debug, Clone)]
pub struct SamplableSet<T>
where
    T: Clone + Eq + Hash,
{
    min_weight_: f64,
    max_weight_: f64,

    // random_01_: Uniform<f64>,
    // Cloning results in identical behavior
    // because the state of RNG is preserved
    // when using #[derive(Clone)]
    // TODO implement Clone manually and reinitialize the RNG
    rng_: RNGType,

    hash_: HashPropensity,
    // TODO remove this variable
    num_groups_: u32,
    max_propensity_vec_: Vec<f64>,

    pos_map_: HashMap<T, SSetPosition>,
    sampling_tree_: BinaryTree,

    propensity_group_vec_: Vec<PropensityGroup<T>>,
    // iterator_: Option<InGroupIndex>,
    // iterator_group_index_: Option<GroupIndex>
}

impl<T> SamplableSet<T>
where
    T: Clone + Eq + Hash,
{
    pub fn new(min_weight: f64, max_weight: f64) -> Self {
        assert!(min_weight > 0.0 && max_weight.is_finite() && max_weight > min_weight);

        let hash = HashPropensity::new(min_weight, max_weight);
        let num_groups = (hash.operator(max_weight) as u32) + 1;

        let mut max_propensity_vec = vec![2.0 * min_weight; num_groups as usize];
        if num_groups > 2 {
            for i in 0..(num_groups - 2) {
                let idx = i as usize;
                max_propensity_vec[idx + 1] = max_propensity_vec[idx] * 2.0;
            }
        }
        if let Some(last) = max_propensity_vec.last_mut() {
            *last = max_weight;
        }

        let sampling_tree = BinaryTree::from(num_groups);
        let propensity_group_vec = vec![Vec::<(T, f64)>::new(); num_groups as usize];

        SamplableSet {
            min_weight_: min_weight,
            max_weight_: max_weight,
            // random_01_: Uniform::new(0.0, 1.0).unwrap(),
            rng_: RNGType::from_os_rng(),
            hash_: hash,
            num_groups_: num_groups,
            max_propensity_vec_: max_propensity_vec,
            pos_map_: HashMap::new(),
            sampling_tree_: sampling_tree,
            propensity_group_vec_: propensity_group_vec,
        }
    }

    pub fn size(&self) -> usize {
        self.pos_map_.len()
    }

    pub fn empty(&self) -> bool {
        self.pos_map_.is_empty()
    }

    #[inline]
    pub fn exists(&self, element: &T) -> bool {
        self.pos_map_.contains_key(element)
    }

    pub fn sample(&mut self) -> Option<(T, f64)> {
        if self.empty() {
            return None;
        }

        let total = self.total_weight();
        // TODO: this shouldn't be possible, ensure guarantees then remove
        if !total.is_finite() || total <= 0.0 {
            return None;
        }

        let r_grp: f64 = self.rng_.random_range(0.0..1.0);
        let grp_idx: GroupIndex = self.sampling_tree_.get_leaf_idx(Some(r_grp));

        // In valid structure, groups indexed by leaves are non-empty
        let grp = &self.propensity_group_vec_[grp_idx];
        let grp_len = grp.len();
        if grp_len == 0 {
            // If this ever happens, the tree/groups are out of sync.
            // TODO panic or error instead? should never have an empty SamplableSet
            return None;
        }

        let m_k = self.max_propensity_vec_[grp_idx];
        loop {
            let u: f64 = self.rng_.random_range(0.0..grp_len as f64);
            let in_grp_idx: InGroupIndex = (u as f64).floor() as InGroupIndex;

            let (elem, weight) = {
                let (e, w) = &grp[in_grp_idx];
                (e.clone(), *w)
            };

            let u_acc: f64 = self.rng_.random_range(0.0..1.0);
            if u_acc < (weight / m_k) {
                return Some((elem, weight));
            }
            // Expected O(1) retries
        }
    }

    pub fn sample_ext_rng<R>(&mut self, generator: &mut R) -> Option<(T, f64)>
    where
        R: Rng + ?Sized,
    {
        if self.empty() {
            return None;
        }

        let total = self.total_weight();
        // TODO: this shouldn't be possible, ensure guarantees then remove
        if !total.is_finite() || total <= 0.0 {
            return None;
        }

        let r_grp: f64 = generator.random_range(0.0..=1.0);
        let grp_idx: GroupIndex = self.sampling_tree_.get_leaf_idx(Some(r_grp));

        let grp = &self.propensity_group_vec_[grp_idx];
        let grp_len = grp.len();
        if grp_len == 0 {
            return None;
        }

        let m_k = self.max_propensity_vec_[grp_idx];
        loop {
            let u: f64 = generator.random_range(0.0..=grp_len as f64);
            let in_grp_idx: InGroupIndex = (u * grp_len as f64).floor() as InGroupIndex;

            let (elem, weight) = {
                let (e, w) = &grp[in_grp_idx];
                (e.clone(), *w)
            };

            let u_acc: f64 = generator.random_range(0.0..1.0);
            if u_acc < (weight / m_k) {
                return Some((elem, weight));
            }
        }
    }

    // TODO this requires a guarantee that cur_node is always root_
    pub fn total_weight(&self) -> f64 {
        self.sampling_tree_.get_val().unwrap()
    }

    // TODO add errors instead of panics
    pub fn get_weight(&self, element: &T) -> f64 {
        let &(g, i) = self
            .pos_map_
            .get(element)
            .expect("element not found in SamplableSet");
        return self.propensity_group_vec_[g as usize][i as usize].1;
    }

    // TODO add errors and Oks
    pub fn insert(&mut self, element: &T, weight: f64) {
        self.weight_check(weight);

        if self.pos_map_.contains_key(element) {
            ()
        }

        let grp_idx: GroupIndex = self.hash_.operator(weight);
        let grp = &mut self.propensity_group_vec_[grp_idx];
        let in_grp_idx: InGroupIndex = grp.len();

        grp.push((element.clone(), weight));
        self.pos_map_.insert(element.clone(), (grp_idx, in_grp_idx));

        self.sampling_tree_.update_value(weight, Some(grp_idx));

        ()
    }

    // TODO add optimization for the case where weight results in no change
    // and mutate in place
    pub fn set_weight(&mut self, element: &T, weight: f64) {
        self.weight_check(weight);
        self.erase(element);
        self.insert(element, weight);

        ()
    }

    pub fn erase(&mut self, element: &T) {
        if self.exists(element) {
            let (grp_idx, in_grp_idx) = match self.pos_map_.get(element) {
                Some(&pos) => pos,
                None => return, // Element not found, nothing to erase
            };

            let grp = &mut self.propensity_group_vec_[grp_idx];
            let w_old = grp[in_grp_idx].1;

            self.sampling_tree_.update_value(-w_old, Some(grp_idx));

            let last_idx = grp.len() - 1;
            // TODO return an error if trying to remove last element?
            if in_grp_idx != last_idx {
                let moved_key = grp[last_idx].0.clone();
                grp.swap(in_grp_idx, last_idx);
                self.pos_map_.insert(moved_key, (grp_idx, in_grp_idx));
            }

            grp.pop();
            self.pos_map_.remove(element);

            ()
        }
    }

    pub fn clear(&mut self) {
        self.sampling_tree_.clear();
        self.pos_map_.clear();
        for grp in &mut self.propensity_group_vec_ {
            grp.clear();
        }

        ()
    }

    fn weight_check(&self, weight: f64) {
        if weight < self.min_weight_ || weight > self.max_weight_ {
            // TODO replace with error
            panic!(
                "Weight {} is out of bounds [{}, {}]",
                weight, self.min_weight_, self.max_weight_
            );
        }
    }

    fn seed(&mut self, seed: u64) {
        self.rng_ = RNGType::seed_from_u64(seed);
    }
}

impl<'a, T> IntoIterator for &'a SamplableSet<T>
where
    T: Clone + Eq + Hash,
{
    type Item = (&'a T, f64);
    type IntoIter = SeqSamplableIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        let mut it = SeqSamplableIter {
            groups: &self.propensity_group_vec_,
            cur_grp: 0,
            cur_idx: 0,
        };
        // It's possible for the group to be empty,
        // so try to find the first non-empty group
        while it.cur_grp < it.groups.len() && it.groups[it.cur_grp].is_empty() {
            it.cur_grp += 1;
        }
        it
    }
}

impl<T> SamplableSet<T>
where
    T: Clone + Eq + Hash,
{
    pub fn into_sampling_iter<'a>(&'a mut self, n: usize) -> SamplingIter<'a, T> {
        SamplingIter {
            set: self,
            remaining: n,
        }
    }

    pub fn into_ext_sampling_iter<'a, R>(
        &'a mut self,
        generator: &'a mut R,
        n: usize,
    ) -> SamplingIterExt<'a, T, R>
    where
        R: Rng + ?Sized,
    {
        SamplingIterExt {
            set: self,
            generator,
            remaining: n,
        }
    }
}

pub struct SeqSamplableIter<'a, T>
where
    T: Clone + Eq + Hash + 'a,
{
    groups: &'a [PropensityGroup<T>],
    cur_grp: usize,
    cur_idx: usize,
}

impl<'a, T> Iterator for SeqSamplableIter<'a, T>
where
    T: Clone + Eq + Hash,
{
    type Item = (&'a T, f64);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.cur_grp >= self.groups.len() {
                return None; // No more groups to iterate over
            }

            let grp = &self.groups[self.cur_grp];
            if self.cur_idx < grp.len() {
                // still elements left in group
                let (ref key, w) = grp[self.cur_idx];
                self.cur_idx += 1;
                return Some((key, w));
            } else {
                // move to next group
                self.cur_grp += 1;
                self.cur_idx = 0;
                while self.cur_grp < self.groups.len() && self.groups[self.cur_grp].is_empty() {
                    self.cur_grp += 1;
                }
            }
        }
    }
}

pub struct SamplingIter<'a, T>
where
    T: Clone + Eq + Hash,
{
    set: &'a mut SamplableSet<T>,
    remaining: usize,
}

impl<'a, T> Iterator for SamplingIter<'a, T>
where
    T: Clone + Eq + Hash,
{
    type Item = (T, f64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        Some(self.set.sample().unwrap())
    }
}

pub struct SamplingIterExt<'a, T, R>
where
    T: Clone + Eq + Hash,
    R: Rng + ?Sized + 'a,
{
    set: &'a mut SamplableSet<T>,
    generator: &'a mut R,
    remaining: usize,
}

impl<'a, T, R> Iterator for SamplingIterExt<'a, T, R>
where
    T: Clone + Eq + Hash,
    R: Rng + ?Sized + 'a,
{
    // type Item = SSetResult<(T, f64)>;
    type Item = (T, f64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        match self.set.sample_ext_rng(self.generator) {
            Some(x) => {
                self.remaining -= 1;
                Some(x)
            }
            None => {
                self.remaining = 0;
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // use rand::SeedableRng;
    // use rand_pcg::Pcg32 as RNGType;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool { (a - b).abs() <= eps }

    #[test]
    fn insert_erase_and_totals() {
        let mut s = SamplableSet::<i32>::new(1.0, 8.0);
        // set fixed internal RNG for this test
        s.rng_ = RNGType::seed_from_u64(42);

        let a = 1;
        let b = 2;
        let c = 3;

        s.insert(&a, 1.0);
        s.insert(&b, 2.0);
        s.insert(&c, 5.0);

        assert_eq!(s.size(), 3);
        assert!(s.exists(&b));
        assert!(approx_eq(s.total_weight(), 8.0, 1e-12));

        s.erase(&b);
        assert_eq!(s.size(), 2);
        assert!(!s.exists(&b));
        assert!(approx_eq(s.total_weight(), 6.0, 1e-12));
    }

    #[test]
    fn get_weight_and_exists() {
        let mut s = SamplableSet::<&'static str>::new(1.0, 8.0);
        s.rng_ = RNGType::seed_from_u64(7);

        s.insert(&"apple", 3.0);
        s.insert(&"banana", 5.0);
        assert!(s.exists(&"apple"));
        assert_eq!(s.get_weight(&"apple"), 3.0);
        assert_eq!(s.get_weight(&"banana"), 5.0);
    }

    #[test]
    fn iterator_walks_all_pairs() {
        let mut s = SamplableSet::<i32>::new(1.0, 8.0);
        s.insert(&10, 2.0);
        s.insert(&11, 3.0);
        s.insert(&12, 1.0);

        let items: Vec<(i32, f64)> = (&s).into_iter().map(|(k, w)| (*k, w)).collect();
        assert_eq!(items.len(), s.size());
        assert!(items.iter().any(|(k, _)| *k == 10));
        assert!(items.iter().any(|(k, _)| *k == 11));
        assert!(items.iter().any(|(k, _)| *k == 12));
    }

    #[test]
    fn sampling_distribution_matches_weights_basic() {
        // weights 1:2:5 -> probabilities 1/8, 2/8, 5/8
        let mut s = SamplableSet::<usize>::new(1.0, 8.0);
        s.rng_ = RNGType::seed_from_u64(123);
        s.insert(&0, 1.0);
        s.insert(&1, 2.0);
        s.insert(&2, 5.0);

        let n = 100_000usize;
        let mut counts = [0usize; 3];
        for _ in 0..n {
            let (k, _) = s.sample().expect("non-empty");
            counts[k] += 1;
        }

        let p = [1.0/8.0, 2.0/8.0, 5.0/8.0];
        for i in 0..3 {
            let freq = counts[i] as f64 / n as f64;
            let sigma = (p[i] * (1.0 - p[i]) / n as f64).sqrt();
            assert!(
                (freq - p[i]).abs() <= 5.0 * sigma,
                // \u{03C3} = sigma
                "bucket {i}: freq={freq:.6}, expected={:.6}, 5\u{03C3}={:.6}",
                p[i], 5.0 * sigma
            );
        }
    }

    #[test]
    fn clear_zeros_but_keeps_groups() {
        let mut s = SamplableSet::<i32>::new(1.0, 8.0);
        s.insert(&1, 1.0);
        s.insert(&2, 2.0);
        assert!(s.total_weight() > 0.0);

        let group_count = s.propensity_group_vec_.len();
        s.clear();

        assert_eq!(s.size(), 0);
        assert!(approx_eq(s.total_weight(), 0.0, 1e-12));
        assert_eq!(s.propensity_group_vec_.len(), group_count);
        assert!(s.propensity_group_vec_.iter().all(|g| g.is_empty()));
    }

    #[test]
    fn sampling_iterator_yields_n_items() {
        let mut s = SamplableSet::<i32>::new(1.0, 8.0);
        s.rng_ = RNGType::seed_from_u64(9);
        s.insert(&1, 3.0);
        s.insert(&2, 5.0);

        let n = 1000;
        let v: Vec<_> = s.into_sampling_iter(n).collect();
        assert_eq!(v.len(), n);
    }
}
