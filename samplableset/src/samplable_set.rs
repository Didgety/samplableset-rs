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

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg32 as RNGType;

#[cfg(feature = "share_rng")]
use std::cell::RefCell;

use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::marker::PhantomData;

use crate::binary_tree::BinaryTree;
use crate::hash_propensity::HashPropensity;

type GroupIndex = usize;
type InGroupIndex = usize;
type SSetPosition = (GroupIndex, InGroupIndex);

type SSetResult<T, K> = Result<T, SSetError<K>>;

type PropensityGroup<T> = Vec<(T, f64)>;

// TODO add option for below
// For consistency with original C++ implementation
// All instances share the same rng stream (per thread)
#[cfg(feature = "share_rng")]
thread_local! {
    static GEN: RefCell<RNGType> = RefCell::new(RNGType::from_seed(rand::random()));
}

/// Private helper function for static GEN
/// If any instance of SamplableSet calls this,
/// RNG will be updated for **all** instances
#[cfg(feature = "share_rng")]
fn seed_static_gen(seed: u64) {
    GEN.with(|g|
    *g.borrow_mut() = RNGType::seed_from_u64(seed)
    );
}

/// Errors that can occur within the sampling set.
#[derive(Debug)]
pub enum SSetError<K> {
    /// The set is empty.
    EmptySet,
    /// Internal state is broken and invalid operations occurred or may occur
    InconsistentState(&'static str),
    /// Key not found in the set.
    KeyNotFound(K),  
    /// Weight is not within the bounds $$[w_{\min}, w_{\max}]$$
    WeightOutOfRange { w: f64, min: f64, max: f64 },
}

impl<K: fmt::Debug> fmt::Display for SSetError<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SSetError::EmptySet => write!(f, "The set is empty."),
            SSetError::InconsistentState(msg) => write!(f, "Inconsistent state: {}", msg),
            SSetError::KeyNotFound(key) => write!(f, "Key not found: {:?}", key),
            SSetError::WeightOutOfRange { w, min, max } => {
                write!(f, "Weight {} is out of range [{}, {}]", w, min, max)
            }
        }
    }
}

/// A weighted set that supports fast sampling *with replacement*
/// with probability proportional to item weights.
///
/// This implements the composition–rejection sampler of
/// St-Onge et al., *Comput. Phys. Commun.* 240 (2019) 30-37
/// (DOI: [10.1016/j.cpc.2019.02.008](https://doi.org/10.1016/j.cpc.2019.02.008)), 
/// specialized with **dyadic (power-of-two) propensity groups**.
///
/// # Model
/// Store pairs $(x_i, w_i)$ with $w_{\min} \le w_i \le w_{\max}$.
/// Items are partitioned by weight scale into groups $G_k$ via `HashPropensity`,
/// approximately:
///
/// - $G_k = \[i \mid 2^k \cdot w_{\min} \le w_i < 2^{k+1} \cdot w_{\min} \]$ for $k = 0,\dots,G-2$,
/// - $G_{G-1}$ covers the top range up to $w_{\max}$.
///
/// For each group $k$, the set maintains:
/// - $S_k = \sum_{i \in G_k} w_i$ — the group’s total, stored in a cumulative
///   binary tree over groups.
/// - $m_k$ — an upper bound on any $w_i$ in $G_k$ (about $2^{k+1} w_{\min}$,
///   while the last group uses $w_{\max}$).
///
/// # Sampling (composition–rejection)
/// 1. **Composition:** choose a group $g$ with probability
///    $P(g) = \dfrac{S_g}{S}$, where $S = \sum_k S_k$, by walking the
///    cumulative tree (logarithmic in the number of groups).
/// 2. **Rejection:** pick an index uniformly within $G_g$ and accept it with
///    probability $\dfrac{w_j}{m_g}$; otherwise retry in the same group.
///    Dyadic grouping keeps $w_j$ close to $m_g$, so the acceptance probability
///    is bounded away from zero (in the ideal dyadic case, $\ge \tfrac{1}{2}$),
///    yielding $\mathcal{O}(1)$ expected retries.
///
/// # Complexity
/// Let $W = \dfrac{w_{\max}}{w_{\min}}$ and $G = \lfloor \log_2 W \rfloor + 1$.
/// - **Sampling:** $\mathcal{O}(\log G) = \mathcal{O}(\log\log W)$ for group selection
///   $+$ $\mathcal{O}(1)$ expected for the rejection step.
/// - **Insert / Erase / set\_weight:** update one group’s total in the tree
///   in $\mathcal{O}(\log G)$ and mutate one bucket in $\mathcal{O}(1)$.
/// If $W$ is bounded, these are effectively $\mathcal{O}(1)$ on average.
///
/// # Edge cases & invariants
/// - When $W \le 2$, there is a **single group** ($G = 1$): the tree has one leaf,
///   and sampling reduces to uniform choice within that group followed by acceptance.
/// - The upper boundary (exact power-of-two span) is handled so the maximum weight
///   does not hash past the last group.
/// - Public methods preserve internal invariants; internal helpers assume them.
///
/// # Examples
/// ```
/// use samplableset_rs::SamplableSet;
/// 
/// let mut s = SamplableSet::<u64>::new(1.0, 8.0).unwrap();
/// s.insert(&1, 3.0).unwrap();
/// s.insert(&2, 5.0).unwrap();
///
/// // Draw one sample (with replacement)
/// let draw = s.sample();
/// assert!(draw.is_ok());
///
/// // Deterministic iteration over stored items
/// for (k, w) in &s {
///     // use k, w
/// }
/// 
/// // Create a sampling iterator and collect the samples
/// let iter = s.into_sampling_iter(10000);
/// let samples: Vec<_> = iter.collect();
/// ```
#[derive(Debug, Clone)]
pub struct SamplableSet<T>
where
    T: Clone + Eq + Hash,
{
    min_weight_: f64,
    max_weight_: f64,

    // Cloning results in identical behavior
    // because the state of RNG is preserved
    // when using #[derive(Clone)]
    // TODO implement Clone manually and reinitialize the RNG
    #[cfg(not(feature = "share_rng"))]
    rng_: RNGType,

    hash_: HashPropensity,
    max_propensity_vec_: Vec<f64>,

    pos_map_: HashMap<T, SSetPosition>,
    sampling_tree_: BinaryTree,

    propensity_group_vec_: Vec<PropensityGroup<T>>,
}

impl<K> SamplableSet<K>
where
    K: Clone + Eq + Hash,
{
    /// Creates a new, empty [SamplableSet] with 
    /// $\lceil \log_2 \dfrac{w_{\max}}{w_{\min}} \rceil + 1$ groups.
    pub fn new(min_weight: f64, max_weight: f64) -> SSetResult<Self, PhantomData<K>> {
        if min_weight <= 0.0 || !max_weight.is_finite() || max_weight <= min_weight {
            return Err(SSetError::WeightOutOfRange {
                w: min_weight,
                // min weight must be > 0
                min: 0.01,
                max: f64::INFINITY,
            });
        }

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
        let propensity_group_vec = vec![Vec::<(K, f64)>::new(); num_groups as usize];

       Ok(SamplableSet {
            min_weight_: min_weight,
            max_weight_: max_weight,
            #[cfg(not(feature = "share_rng"))]
            rng_: RNGType::from_os_rng(),
            hash_: hash,
            max_propensity_vec_: max_propensity_vec,
            pos_map_: HashMap::new(),
            sampling_tree_: sampling_tree,
            propensity_group_vec_: propensity_group_vec,
        })
    }

    /// Returns the number of elements in the set.
    pub fn size(&self) -> usize {
        self.pos_map_.len()
    }

    /// Returns true if the set is empty.
    pub fn empty(&self) -> bool {
        self.pos_map_.is_empty()
    }

    #[inline]
    /// Checks if the element exists in the set.
    pub fn exists(&self, element: &K) -> bool {
        self.pos_map_.contains_key(element)
    }

    /// Returns the total weight of the set.
    /// 
    // The requirement that `cur_node_` be the root
    // is met by the implementation of the BinaryTree.
    pub fn total_weight(&self) -> f64 {
        self.sampling_tree_.get_val()
    }

    /// Returns the weight of the given element, if it exists.
    /// 
    /// You should use [SamplableSet::exists] to check if an element is in the set
    /// before trying to get its weight.
    /// 
    /// Returns [SSetError::KeyNotFound] if the element is not found.
    pub fn get_weight(&self, element: &K) -> SSetResult<f64, K> {
        let &(g, i) = self
            .pos_map_
            .get(element)
            .ok_or(SSetError::KeyNotFound(element.clone()))?;
        Ok(self.propensity_group_vec_[g as usize][i as usize].1)
    }

    /// Inserts an element into the set with the given weight.
    /// 
    /// Returns `true` on success, 
    /// `false` on duplicate key.
    /// 
    /// Returns [SSetError::WeightOutOfRange] if the weight is invalid.
    pub fn insert(&mut self, element: &K, weight: f64) -> SSetResult<bool, K> {
        match self.weight_check(weight) {
            Ok(()) => (),
            Err(e) => return Err(e),
        }

        if self.pos_map_.contains_key(element) {
            return Ok(false);
        }

        let grp_idx: GroupIndex = self.hash_.operator(weight);
        let grp = &mut self.propensity_group_vec_[grp_idx];
        let in_grp_idx: InGroupIndex = grp.len();

        grp.push((element.clone(), weight));
        self.pos_map_.insert(element.clone(), (grp_idx, in_grp_idx));

        self.sampling_tree_.update_value(weight, Some(grp_idx));

        Ok(true)
    }

    // TODO add optimization for the case where weight results in no change
    // and mutate in place
    /// Sets the weight of a node.
    /// If the node does not exist, functionally the same as insert.
    /// 
    /// Returns [SSetError::WeightOutOfRange] if the weight is invalid.
    pub fn set_weight(&mut self, element: &K, weight: f64) -> SSetResult<(), K> {
        match self.weight_check(weight) {
            Ok(()) => {
                let _ = self.erase(element);
                // Weight has already been checked, and we know the 
                // key does not exist since we just removed it, so 
                // we can ignore the return type of insert().
                // However, insert is part of the public API so keeping
                // the return type is a good idea so that users can know if an 
                // insertion failed.
                self.insert(element, weight)?;
            }
            Err(e) => return Err(e),
        }

        Ok(())
    }

    /// Erases an element from the set, if it exists.
    /// 
    /// Returns `true` on element removed, 
    /// `false` if element was not found.
    pub fn erase(&mut self, element: &K) -> bool {
        let (grp_idx, in_grp_idx) = match self.pos_map_.get(element) {
            Some(&pos) => pos,
            // Element does not exist
            None => return false,
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

        return true
    }

    // TODO expose publicly?
    #[cfg(not(feature = "share_rng"))]
    #[allow(dead_code)]
    pub fn seed(&mut self, seed: u64) {
        self.rng_ = RNGType::seed_from_u64(seed);
    }

    /// Draw one `(element, weight)` proportional to weight (with replacement).
    ///
    /// Uses the set’s **internal RNG** and the composition–rejection scheme
    /// described in the type-level docs.
    ///
    /// **Complexity:** $\mathcal{O}(\log\log W)$ expected, where
    /// $W = \dfrac{w_{\max}}{w_{\min}}$.
    ///
    /// For deterministic external RNG, use [SamplableSet::sample_ext_rng] instead.
    /// 
    /// ------
    /// 
    /// Returns `Ok((element, weight))` on success.
    ///
    /// Returns [SSetError::EmptySet] if the set is empty.
    pub fn sample(&mut self) -> SSetResult<(K, f64), PhantomData<K>> {
        // TODO this is the only validity check in the original implementation
        // should the others be changed to debug_assert!() ?
        if self.empty() {
            return Err(SSetError::EmptySet);
        }

        let total = self.total_weight();
        // TODO: this shouldn't be possible, ensure guarantees then remove
        if !total.is_finite() || total <= 0.0 {
            return Err(SSetError::InconsistentState("Invalid total weight"));
        }

        let r: f64 = self.random_range(0.0..1.0);
        let grp_idx: GroupIndex = self.sampling_tree_.get_leaf_idx(Some(r));

        // In valid structure, groups indexed by leaves are non-empty
        let m_k = self.max_propensity_vec_[grp_idx];
        let grp_len = self.propensity_group_vec_[grp_idx].len();
        if grp_len == 0 {
            // If this ever happens, the tree/groups are out of sync.
            return Err(SSetError::InconsistentState("Empty group"));
        }
        loop {
            let u: f64 = self.random_range(0.0..grp_len as f64);
            let in_grp_idx: InGroupIndex = (u as f64).floor() as InGroupIndex;

            let (elem, weight) = {
                let (e, w) = &self.propensity_group_vec_[grp_idx][in_grp_idx];
                (e.clone(), *w)
            };

            let u_acc: f64 = self.random_range(0.0..1.0);
            if u_acc < (weight / m_k) {
                return Ok((elem, weight));
            }
            // Expected O(1) retries
        }
    }

    /// Draw one `(element, weight)` using a **caller-supplied RNG**.
    ///
    /// Algorithm and guarantees are identical to [SamplableSet::sample], but the random draws
    /// come from `generator`. 
    ///
    /// **Complexity:** $\mathcal{O}(\log\log W)$ expected, where
    /// $W = \dfrac{w_{\max}}{w_{\min}}$.
    /// 
    /// ------
    /// 
    /// Returns `Ok((element, weight))` on success.
    /// 
    /// Returns [SSetError::EmptySet] if the set is empty.
    pub fn sample_ext_rng<R>(&mut self, generator: &mut R) -> SSetResult<(K, f64), PhantomData<K>>
    where
        R: Rng + ?Sized,
    {
        if self.empty() {
            return Err(SSetError::EmptySet);
        }

        let total = self.total_weight();
        // TODO: this shouldn't be possible, ensure guarantees then remove
        if !total.is_finite() || total <= 0.0 {
            return Err(SSetError::InconsistentState("Invalid total weight"));
        }

        let r: f64 = generator.random_range(0.0..1.0);
        let grp_idx: GroupIndex = self.sampling_tree_.get_leaf_idx(Some(r));

        let grp = &self.propensity_group_vec_[grp_idx];
        let grp_len = grp.len();
        if grp_len == 0 {
            return Err(SSetError::InconsistentState("Empty group"));
        }

        let m_k = self.max_propensity_vec_[grp_idx];
        loop {
            let u: f64 = generator.random_range(0.0..grp_len as f64);
            let in_grp_idx: InGroupIndex = (u * grp_len as f64).floor() as InGroupIndex;

            let (elem, weight) = {
                let (e, w) = &grp[in_grp_idx];
                (e.clone(), *w)
            };

            let u_acc: f64 = generator.random_range(0.0..1.0);
            if u_acc < (weight / m_k) {
                return Ok((elem, weight));
            }
        }
    }

    /// Clears all elements from the set.
    pub fn clear(&mut self) {
        self.sampling_tree_.clear();
        self.pos_map_.clear();
        for grp in &mut self.propensity_group_vec_ {
            grp.clear();
        }

        ()
    }

    #[doc(hidden)]
    /// Checks that the weight is within the allowed bounds.
    /// 
    /// Returns SSetError::WeightOutOfRange if the weight is invalid.
    fn weight_check(&self, weight: f64) -> SSetResult<(), K> {
        if weight < self.min_weight_ || weight > self.max_weight_ {
            Err(SSetError::WeightOutOfRange {
                w: weight,
                min: self.min_weight_,
                max: self.max_weight_,
            })
        } else {
            Ok(())
        }
    }

    /// Internal helper to get a random `f64` in a range
    /// using either shared or independent RNG depending on 
    /// the feature flag.
    fn random_range(&mut self, range: std::ops::Range<f64>) -> f64 {
        #[cfg(feature = "share_rng")]
        GEN.with(|g| g.borrow_mut().random_range(range));
        #[cfg(not(feature = "share_rng"))]
        self.rng_.random_range(range)
    }
}

impl<'a, T> IntoIterator for &'a SamplableSet<T>
where
    T: Clone + Eq + Hash,
{
    type Item = (&'a T, f64);
    /// The type of the iterator returned by `into_iter`.
    type IntoIter = SeqSamplableIter<'a, T>;

    /// Returns a sequential iterator over the items in the set.
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
    T: Clone + fmt::Debug +  Eq + Hash,
{
    /// Returns an iterator that lazily samples `n` items from the set
    /// using the built in random number generator.
    pub fn into_sampling_iter<'a>(&'a mut self, n: usize) -> SamplingIter<'a, T> {
        SamplingIter {
            set: self,
            remaining: n,
        }
    }

    /// Returns an iterator that lazily samples `n` items from the set
    /// given an external random number generator.
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

/// A sequential iterator over the items in the set
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

    /// Returns the next item sequentially from the set
    /// or None if there are no more items
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

/// A sampling iterator over the items in the set.
/// 
/// This iterator will yield a fixed number of samples from the set.
pub struct SamplingIter<'a, T>
where
    T: Clone + fmt::Debug + Eq + Hash,
{
    set: &'a mut SamplableSet<T>,
    remaining: usize,
}

impl<'a, T> Iterator for SamplingIter<'a, T>
where
    T: Clone + fmt::Debug + Eq + Hash,
{
    type Item = (T, f64);

    /// Returns the next sampled item from the set
    /// using the built in random number generator
    /// or None if there are no samples remaining 
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        Some(self.set.sample().unwrap())
    }
}

/// A sampling iterator over the items in the set that
/// takes an external RNG source.
/// 
/// This iterator will yield a fixed number of samples from the set.
pub struct SamplingIterExt<'a, T, R>
where
    T: Clone + fmt::Debug + Eq + Hash,
    R: Rng + ?Sized + 'a,
{
    set: &'a mut SamplableSet<T>,
    generator: &'a mut R,
    remaining: usize,
}

impl<'a, T, R> Iterator for SamplingIterExt<'a, T, R>
where
    T: Clone + fmt::Debug + Eq + Hash,
    R: Rng + ?Sized + 'a,
{
    type Item = (T, f64);

    /// Returns the next sampled item from the set,
    /// using the provided external random number generator
    /// or None if there are no samples remaining
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        match self.set.sample_ext_rng(self.generator) {
            Ok(x) => {
                self.remaining -= 1;
                Some(x)
            }
            Err(_) => {
                self.remaining = 0;
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool { (a - b).abs() <= eps }

    #[test]
    fn insert_erase_and_totals() {
        let mut s = SamplableSet::<i32>::new(1.0, 8.0).unwrap();
        // set fixed internal RNG for this test
        #[cfg(not(feature = "share_rng"))]
        { s.rng_ = RNGType::seed_from_u64(42); }
        #[cfg(feature = "share_rng")]
        { seed_static_gen(42); }

        let a = 1;
        let b = 2;
        let c = 3;

        let _ = s.insert(&a, 1.0);
        let _ = s.insert(&b, 2.0);
        let _ = s.insert(&c, 5.0);

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
        let mut s = SamplableSet::<&'static str>::new(1.0, 8.0).unwrap();
        #[cfg(not(feature = "share_rng"))]
        { s.rng_ = RNGType::seed_from_u64(7); }
        #[cfg(feature = "share_rng")]
        { seed_static_gen(7); }

        let _ = s.insert(&"apple", 3.0);
        let _ = s.insert(&"banana", 5.0);
        assert!(s.exists(&"apple"));
        assert_eq!(s.get_weight(&"apple").unwrap(), 3.0);
        assert_eq!(s.get_weight(&"banana").unwrap(), 5.0);
    }

    #[test]
    fn iterator_walks_all_pairs() {
        let mut s = SamplableSet::<i32>::new(1.0, 8.0).unwrap();
        let _ = s.insert(&10, 2.0);
        let _ = s.insert(&11, 3.0);
        let _ = s.insert(&12, 1.0);

        let items: Vec<(i32, f64)> = (&s).into_iter().map(|(k, w)| (*k, w)).collect();
        assert_eq!(items.len(), s.size());
        assert!(items.iter().any(|(k, _)| *k == 10));
        assert!(items.iter().any(|(k, _)| *k == 11));
        assert!(items.iter().any(|(k, _)| *k == 12));
    }

    #[test]
    fn sampling_distribution_matches_weights_basic() {
        // weights 1:2:5 -> probabilities 1/8, 2/8, 5/8
        let mut s = SamplableSet::<usize>::new(1.0, 8.0).unwrap();
        #[cfg(not(feature = "share_rng"))]
        { s.rng_ = RNGType::seed_from_u64(123); }
        #[cfg(feature = "share_rng")]
        { seed_static_gen(123); }
        let _ = s.insert(&0, 1.0);
        let _ = s.insert(&1, 2.0);
        let _ = s.insert(&2, 5.0);

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
        let mut s = SamplableSet::<i32>::new(1.0, 8.0).unwrap();
        let _ = s.insert(&1, 1.0);
        let _ = s.insert(&2, 2.0);
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
        let mut s = SamplableSet::<i32>::new(1.0, 8.0).unwrap();
        #[cfg(not(feature = "share_rng"))]
        { s.rng_ = RNGType::seed_from_u64(9); }
        #[cfg(feature = "share_rng")]
        { seed_static_gen(9); }
        let _ = s.insert(&1, 3.0);
        let _ = s.insert(&2, 5.0);

        let n = 1000;
        let v: Vec<_> = s.into_sampling_iter(n).collect();
        assert_eq!(v.len(), n);
    }

    #[test]
    fn single_group_sampling_is_safe() {
        let mut s = SamplableSet::<u64>::new(1.0, 1.5).unwrap(); // R < 2 -> 1 group
        #[cfg(not(feature = "share_rng"))]
        { s.rng_ = RNGType::seed_from_u64(123); }
        #[cfg(feature = "share_rng")]
        { seed_static_gen(123); }

        let _ = s.insert(&10, 1.0);
        let _ = s.insert(&20, 1.2);
        let _ = s.insert(&30, 1.4);

        for _ in 0..50_000 {
            let got = s.sample();
            assert!(got.is_ok(), "sample returned None on non-empty set");
        }
    }

    #[test]
    fn power_of_two_span_is_safe() {
        let mut s = SamplableSet::<u64>::new(1.0, 8.0).unwrap(); // ratio = 8 (power of two)
        #[cfg(not(feature = "share_rng"))]
        { s.rng_ = RNGType::seed_from_u64(7); }
        #[cfg(feature = "share_rng")]
        { seed_static_gen(7); }

        // put something in each group (your insert hashes by weight)
        let _ = s.insert(&1, 1.0);
        let _ = s.insert(&2, 2.0);
        let _ = s.insert(&3, 3.5);
        let _ = s.insert(&4, 7.9);

        for _ in 0..50_000 {
            assert!(s.sample().is_ok());
        }
    }

    #[test]
    fn clear_then_resample_is_safe() {
        let mut s = SamplableSet::<u64>::new(1.0, 8.0).unwrap();
        #[cfg(not(feature = "share_rng"))]
        { s.rng_ = RNGType::seed_from_u64(42); }
        #[cfg(feature = "share_rng")]
        { seed_static_gen(42); }
        let _ = s.insert(&1, 3.0);
        let _ = s.insert(&2, 5.0);

        s.clear();
        assert!(s.empty());

        // Refill and sample heavily
        let _ = s.insert(&10, 1.0);
        let _ = s.insert(&11, 2.0);
        let _ = s.insert(&12, 5.0);
        for _ in 0..20_000 {
            assert!(s.sample().is_ok());
        }
    }

    #[test]
    fn mutate_and_sample_fuzz_is_safe() {
        let mut s = SamplableSet::<u64>::new(0.5, 10.0).unwrap();
        #[cfg(not(feature = "share_rng"))]
        { s.rng_ = RNGType::seed_from_u64(999); }
        #[cfg(feature = "share_rng")]
        { seed_static_gen(999); }

        for k in 0..50 {
            let _ = s.insert(&k, 0.5 + ((k as f64) % 10.0));
        }

        // 64-bit LCG: x_{n+1} = a*x_n + c (mod 2^64)
        let mut r: u64 = 1;
        const A: u64 = 6364136223846793005;
        const C: u64 = 1;

        for _ in 0..10_000 {
            r = r.wrapping_mul(A).wrapping_add(C);

            let which = (r % 3) as u8;
            let key: u64 = (r >> 32) % 60; // use high bits for variety

            match which {
                0 => { s.erase(&key); },
                1 => s.set_weight(&key, 0.5 + ((key as f64) % 10.0)).unwrap(),
                _ => { s.insert(&key, 0.5 + ((key as f64) % 10.0)).unwrap(); },
            }

            if !s.empty() {
                assert!(s.sample().is_ok());
            } else {
                assert!(s.sample().is_ok());
            }
        }
    }

    #[test]
    fn binary_tree_cur_is_always_some() {
        let mut s = SamplableSet::<u64>::new(1.0, 8.0).unwrap();
        let _ = s.insert(&1, 3.0);
        let _ = s.insert(&2, 5.0);
        // get_val() should never panic
        let _ = s.sampling_tree_.get_val();
        // sampling shouldn’t panic either
        for _ in 0..1000 { let _ = s.sample(); }
    }

}
