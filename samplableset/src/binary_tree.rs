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

use std::{
    // cell::RefCell,
    collections::HashMap,
    ops::Deref,
    //rc::{Rc, Weak},
    sync::{Arc, RwLock, Weak},
};

type LeafIdx = usize;

#[doc(hidden)]
/// A strong reference to a `BTreeNode`.
#[derive(Debug, Clone)]
pub(crate) struct NodeRef(Arc<RwLock<BTreeNode>>);

#[doc(hidden)]
/// A weak reference to a `BTreeNode`.
#[derive(Debug)]
pub(crate) struct WeakNodeRef(Weak<RwLock<BTreeNode>>);

// ==============================================

impl NodeRef {
    #[doc(hidden)]
    /// Creates a new `NodeRef`.
    ///
    /// Acts like an `Arc<RwLock<BTreeNode>>`
    pub(crate) fn new(node: BTreeNode) -> Self {
        NodeRef(Arc::new(RwLock::new(node)))
    }

    #[doc(hidden)]
    /// Downgrades the strong reference to a weak reference.
    ///
    /// NodeRef -> WeakNodeRef
    pub(crate) fn downgrade(&self) -> WeakNodeRef {
        WeakNodeRef(Arc::downgrade(self))
    }
}

impl Deref for NodeRef {
    type Target = Arc<RwLock<BTreeNode>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// ==============================================

impl WeakNodeRef {
    #[doc(hidden)]
    /// Creates a new, empty `WeakNodeRef`.
    ///
    /// Acts like a `Weak<RwLock<BTreeNode>>`
    pub(crate) fn new() -> Self {
        WeakNodeRef(Weak::new())
    }

    #[doc(hidden)]
    /// Upgrades the weak reference to a strong reference, if the node is still alive.
    ///
    /// WeakNodeRef -> NodeRef
    pub(crate) fn upgrade(&self) -> Option<NodeRef> {
        self.0.upgrade().map(NodeRef)
    }
}

impl Deref for WeakNodeRef {
    type Target = Weak<RwLock<BTreeNode>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// ==============================================

#[derive(Debug)]
pub(crate) struct BTreeNode {
    val: f64,
    parent: WeakNodeRef,
    left: Option<NodeRef>,
    right: Option<NodeRef>,
}

impl BTreeNode {
    #[doc(hidden)]
    /// Creates a new, empty `BTreeNode`.
    pub(crate) fn new() -> Self {
        Self {
            val: 0.0,
            parent: WeakNodeRef::new(),
            left: None,
            right: None,
        }
    }
}

// ==============================================

#[doc(hidden)]
/// A binary tree.
#[derive(Debug)]
pub(crate) struct BinaryTree {
    root_: Option<NodeRef>,
    cur_node_: Option<NodeRef>,
    leaves_: Vec<NodeRef>,
    leaves_idx_map_: HashMap<usize, LeafIdx>,
}

impl BinaryTree {
    #[doc(hidden)]
    /// Creates a new, empty `BinaryTree`.
    #[allow(dead_code)]
    pub(crate) fn new() -> Self {
        BinaryTree {
            root_: None,
            cur_node_: None,
            leaves_: Vec::new(),
            leaves_idx_map_: HashMap::new(),
        }
    }

    #[doc(hidden)]
    /// Returns true if the current node is the root.
    pub(crate) fn is_root(&self) -> bool {
        self.cur_node_
            .as_ref()
            .map(|n| n.read().unwrap().parent.upgrade().is_none())
            .expect("BinaryTree invariant violated: cannot check root: current_node_ is None")
    }

    #[doc(hidden)]
    /// Returns true if the current node is a leaf.
    pub(crate) fn is_leaf(&self) -> bool {
        self.cur_node_
            .as_ref()
            .map(|n| {
                let nb = n.read().unwrap();
                nb.left.is_none() && nb.right.is_none()
            })
            .expect("BinaryTree invariant violated: cannot check leaf: current node is None")
    }

    #[doc(hidden)]
    /// Returns the value of the current node.
    ///
    /// Guaranteed to always return Something
    pub(crate) fn get_val(&self) -> f64 {
        self.cur_node_
            .as_ref()
            .map(|n| n.read().unwrap().val)
            .expect("BinaryTree invariant violated: cannot get val: current node is None")
    }

    #[doc(hidden)]
    /// Returns the value of the left child.
    ///
    /// This is guaranteed to always return Something
    pub(crate) fn get_left_val(&self) -> f64 {
        let left = {
            let cur = self.cur_node_.as_ref().expect(
                "BinaryTree invariant violated: cannot get left val: current node  is None",
            );
            let c = cur.read().unwrap();
            c.left
                .clone()
                .expect("BinaryTree invariant violated: left child is missing")
        };
        left.read().unwrap().val
    }

    #[doc(hidden)]
    /// Returns the value of the right child.
    ///
    /// Not used in current SamplableSet implementation
    /// But also guaranteed to always return Something
    #[allow(dead_code)]
    pub(crate) fn get_right_val(&self) -> f64 {
        let right = {
            let cur = self.cur_node_.as_ref().expect(
                "BinaryTree invariant violated: cannot get right val: current node is None",
            );
            let c = cur.read().unwrap();
            c.right
                .clone()
                .expect("BinaryTree invariant violated: right child is missing")
        };
        right.read().unwrap().val
    }

    #[doc(hidden)]
    /// Returns the index of the leaf node corresponding to the given value.
    /// Resets the current node to the root after finding the leaf index.
    // TODO have it return an error if not found, should also reset to root
    pub(crate) fn get_leaf_idx(&mut self, r: Option<f64>) -> LeafIdx {
        match r {
            Some(r_val) => {
                let mut cumul: f64 = 0.0;
                let total_val: f64 = self.get_val();
                while !self.is_leaf() {
                    if r_val <= (cumul + self.get_left_val()) / total_val {
                        self.move_down_left();
                    } else {
                        cumul += self.get_left_val();
                        self.move_down_right();
                    }
                }
                let chosen_leaf: LeafIdx = self.get_leaf_idx(None);
                self.reset_cur_node();
                chosen_leaf
            }
            None => {
                let n = self.cur_node_.as_ref().expect(
                    "BinaryTree invariant violated: cannot get leaf index: current_node_ is None",
                );
                // Use the raw pointer as a UID for the node. Safe because there is no dereferencing.
                let key = Arc::as_ptr(n) as usize;
                *self.leaves_idx_map_.get(&key).expect(
                    "BinaryTree invariant violated: current_node_ is not a leaf (no leaf index)",
                )
            }
        }
    }

    #[doc(hidden)]
    /// Resets the current node to the root
    pub(crate) fn reset_cur_node(&mut self) {
        self.cur_node_ = self.root_.clone();
    }

    #[doc(hidden)]
    /// Current node becomes its left child
    pub(crate) fn move_down_left(&mut self) {
        let next = {
            let cur = self.cur_node_.as_ref().expect(
                "BinaryTree invariant violated: cannot move down left: current_node_ is None",
            );
            let guard = cur.read().unwrap();
            guard
                .left
                .clone()
                .expect("BinaryTree invariant violated: cannot move down left: no left child")
        };
        self.cur_node_ = Some(next);
    }

    #[doc(hidden)]
    /// Current node becomes its right child
    pub(crate) fn move_down_right(&mut self) {
        let next = {
            let cur = self.cur_node_.as_ref().expect(
                "BinaryTree invariant violated: cannot move down right: current_node_ is None",
            );
            let guard = cur.read().unwrap();
            guard
                .right
                .clone()
                .expect("BinaryTree invariant violated: cannot move down right: no right child")
        };
        self.cur_node_ = Some(next);
    }

    #[doc(hidden)]
    /// Current node becomes its parent
    pub(crate) fn move_up(&mut self) {
        let next = {
            let cur = self
                .cur_node_
                .as_ref()
                .expect("BinaryTree invariant violated: cannot move up: current_node_ is None");
            let guard = cur.read().unwrap();
            guard
                .parent
                .upgrade()
                .expect("BinaryTree invariant violated: cannot move up: no parent")
        };
        self.cur_node_ = Some(next);
    }

    #[doc(hidden)]
    /// Current node becomes the specified node.
    pub(crate) fn move_to(&mut self, node: NodeRef) {
        self.cur_node_ = Some(node);
    }

    #[doc(hidden)]
    /// Updates the value of a leaf node.
    pub(crate) fn update_value(&mut self, variation: f64, idx: Option<LeafIdx>) {
        match idx {
            Some(leaf_idx) => {
                let node = self.leaves_.get(leaf_idx).cloned().expect(
                    "BinaryTree invariant violated: Cannot update value: invalid leaf index",
                );
                self.cur_node_ = Some(node);
                self._update_helper(variation);
            }
            None => {
                if self.is_leaf() {
                    self._update_helper(variation);
                } else {
                    println!(
                        "BinaryTree invariant violated: Cannot update value: current node is not a leaf"
                    );
                }
            }
        }
    }

    #[doc(hidden)]
    /// Helper function to update the value of the current node.
    fn _update_helper(&mut self, variation: f64) {
        {
            let cur = self
                .cur_node_
                .as_ref()
                .expect("current node is None in _update_helper");
            cur.write().unwrap().val += variation;
        }

        // walk up to the root, adding at each ancestor
        while !self.is_root() {
            self.move_up(); // mutates self.cur_node_
            let cur = self
                .cur_node_
                .as_ref()
                .expect("current node is None after move_up");
            cur.write().unwrap().val += variation;
        }
    }

    #[doc(hidden)]
    /// Zeros the value of the current node.
    pub(crate) fn update_zero(&mut self) {
        if !self.is_leaf() {
            println!("Cannot zero: current node is not a leaf");
        }
        self.cur_node_
            .as_ref()
            .expect("current node is None in update_zero")
            .write()
            .unwrap()
            .val = 0.0;
        while !self.is_root() {
            self.move_up();
            let cur = self
                .cur_node_
                .as_ref()
                .expect("current node is None after move_up in update_zero");
            cur.write().unwrap().val = 0.0;
        }
    }

    #[doc(hidden)]
    /// Zeros all leaves in the tree
    pub(crate) fn clear(&mut self) {
        let leaves = self.leaves_.clone();
        for leaf in leaves {
            self.move_to(leaf);
            self.update_zero();
        }
    }

    #[doc(hidden)]
    /// Creates a new branch in the tree.
    fn branch(&mut self, parent: NodeRef, node_idx: u32, n_nodes: u32) -> Option<NodeRef> {
        if node_idx < n_nodes {
            let child = NodeRef::new(BTreeNode {
                val: 0.0,
                parent: parent.downgrade(),
                left: None,
                right: None,
            });

            let left = self.branch(child.clone(), node_idx * 2 + 1, n_nodes);
            let right = self.branch(child.clone(), node_idx * 2 + 2, n_nodes);

            {
                let mut cb = child.write().unwrap();
                cb.left = left;
                cb.right = right;
            }

            Some(child)
        } else {
            let key = Arc::as_ptr(&parent) as usize;
            if !self.leaves_idx_map_.contains_key(&key) {
                let idx = self.leaves_.len() as LeafIdx;
                self.leaves_idx_map_.insert(key, idx);
                self.leaves_.push(parent.clone());
            }
            None
        }
    }
}

// TODO expand to u64+?
impl<T: Into<u32>> From<T> for BinaryTree {
    fn from(n_leaves: T) -> Self {
        let n_leaves: u32 = n_leaves.into();
        // n_leaves guaranteed to be > 0 by SamplableSet::new()
        debug_assert!(n_leaves >= 1, "BinaryTree must have at least one leaf");

        // let n_leaves = n_leaves * 2 - 1;

        let root = NodeRef::new(BTreeNode::new());

        let mut tree = BinaryTree {
            root_: Some(root.clone()),
            cur_node_: Some(root.clone()),
            leaves_: Vec::with_capacity((n_leaves) as usize),
            leaves_idx_map_: HashMap::new(),
        };

        let n_nodes = n_leaves
            .checked_mul(2)
            .and_then(|v| v.checked_sub(1))
            .expect("Overflow calculating number of nodes");

        // Build subtrees (None when n_leaves == 1)
        let left = tree.branch(root.clone(), 1, n_nodes);
        let right = tree.branch(root.clone(), 2, n_nodes);

        {
            let mut rb = root.write().unwrap();
            rb.left = left;
            rb.right = right;
        }

        tree
    }
}

impl Clone for BinaryTree {
    #[doc(hidden)]
    /// !!! DOES NOT BEHAVE LIKE NORMAL RUST CLONING OF POINTERS !!!
    /// Deep copies the BinaryTree, creating a replica pointing at new data
    fn clone(&self) -> Self {
        let n_leaves = self.leaves_.len();
        debug_assert!(n_leaves > 0, "BinaryTree must have at least one leaf");

        let root = NodeRef::new(BTreeNode::new());

        let mut out = BinaryTree {
            root_: Some(root.clone()),
            cur_node_: Some(root.clone()),
            leaves_: Vec::with_capacity(n_leaves),
            leaves_idx_map_: HashMap::new(),
        };

        let n_nodes = (n_leaves as u32)
            .checked_mul(2)
            .and_then(|v| v.checked_sub(1))
            .expect("Overflow calculating number of nodes");

        let left = BinaryTree::branch(&mut out, root.clone(), 1, n_nodes);
        let right = BinaryTree::branch(&mut out, root.clone(), 2, n_nodes);

        {
            let mut rb = root.write().unwrap();
            rb.left = left;
            rb.right = right;
        }

        for (leaf_idx, src_leaf) in self.leaves_.iter().enumerate() {
            let v = src_leaf.read().unwrap().val;
            BinaryTree::update_value(&mut out, v, Some(leaf_idx as LeafIdx));
        }

        out
    }
}

// ==============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_tree_creation_and_leaves() {
        let tree = BinaryTree::from(4u32);
        assert_eq!(tree.leaves_.len(), 4);
        assert!(tree.root_.is_some());
        assert!(tree.cur_node_.is_some());
    }

    #[test]
    fn test_is_root_and_is_leaf() {
        let mut tree = BinaryTree::from(2u32);
        tree.reset_cur_node();
        assert!(tree.is_root());
        tree.move_down_left();
        assert!(tree.is_leaf());
        tree.reset_cur_node();
        tree.move_down_right();
        assert!(tree.is_leaf());
    }

    #[test]
    fn test_get_val_and_update_value() {
        let mut tree = BinaryTree::from(2u32);
        tree.update_value(5.0, Some(0));
        tree.update_value(3.0, Some(1));
        // After updates, root value should be sum of leaves
        tree.reset_cur_node();
        assert_eq!(tree.get_val(), 8.0);
        // Check leaf values
        tree.move_down_left();
        assert_eq!(tree.get_val(), 5.0);
        tree.reset_cur_node();
        tree.move_down_right();
        assert_eq!(tree.get_val(), 3.0);
    }

    #[test]
    fn test_update_zero_and_clear() {
        let mut tree = BinaryTree::from(2u32);
        tree.update_value(2.0, Some(0));
        tree.update_value(4.0, Some(1));
        tree.clear();
        tree.reset_cur_node();
        assert_eq!(tree.get_val(), 0.0);
        tree.move_down_left();
        assert_eq!(tree.get_val(), 0.0);
        tree.reset_cur_node();
        tree.move_down_right();
        assert_eq!(tree.get_val(), 0.0);
    }

    #[test]
    fn test_clone_tree() {
        let mut tree = BinaryTree::from(2u32);
        tree.update_value(7.0, Some(0));
        tree.update_value(1.0, Some(1));
        let cloned = tree.clone();
        assert_eq!(cloned.leaves_.len(), 2);
        // Check values are copied
        let mut c = cloned;
        c.reset_cur_node();
        assert_eq!(c.get_val(), 8.0);
        c.move_down_left();
        assert_eq!(c.get_val(), 7.0);
        c.reset_cur_node();
        c.move_down_right();
        assert_eq!(c.get_val(), 1.0);
    }

    #[test]
    #[should_panic]
    fn test_binary_tree_from_zero_leaves_panics() {
        let _ = BinaryTree::from(0u32);
    }

    #[test]
    fn test_new_empty_builder() {
        let bt = BinaryTree::new();
        assert!(bt.root_.is_none());
        assert!(bt.cur_node_.is_none());
        assert_eq!(bt.leaves_.len(), 0);
    }

    #[test]
    fn test_navigation_moves() {
        let mut bt = BinaryTree::from(4u32);
        bt.reset_cur_node();
        let root_ptr = Arc::as_ptr(bt.root_.as_ref().unwrap()) as usize;
        bt.move_down_left();
        assert!(!bt.is_root());
        bt.move_up();
        // back at root
        assert_eq!(
            Arc::as_ptr(bt.cur_node_.as_ref().unwrap()) as usize,
            root_ptr
        );
        bt.move_down_right();
        assert!(!bt.is_root());
    }

    #[test]
    fn test_get_left_right_val_after_updates() {
        let mut bt = BinaryTree::from(4u32);
        // Set two leftmost leaves (indices 0,1) and two rightmost (2,3)
        bt.update_value(1.0, Some(0));
        bt.update_value(2.0, Some(1));
        bt.update_value(3.0, Some(2));
        bt.update_value(4.0, Some(3));
        bt.reset_cur_node();
        let left_sum = bt.get_left_val();
        let right_sum = bt.get_right_val();
        assert_eq!(left_sum + right_sum, bt.get_val());
        assert_eq!(bt.get_val(), 10.0);
    }

    #[test]
    fn test_update_value_none_path_on_leaf() {
        let mut bt = BinaryTree::from(2u32);
        // start root -> left leaf
        bt.move_down_left();
        assert!(bt.is_leaf());
        bt.update_value(5.0, None); // should update leaf and propagate
        bt.reset_cur_node();
        assert_eq!(bt.get_val(), 5.0);
    }

    #[test]
    fn test_move_to_specific_leaf() {
        let mut bt = BinaryTree::from(4u32);
        let target_leaf = bt.leaves_[2].clone();
        bt.move_to(target_leaf.clone());
        assert!(bt.is_leaf());
        let cur_ptr = Arc::as_ptr(bt.cur_node_.as_ref().unwrap()) as usize;
        let tgt_ptr = Arc::as_ptr(&target_leaf) as usize;
        assert_eq!(cur_ptr, tgt_ptr);
    }

    #[test]
    fn test_get_leaf_idx_none_returns_correct_index() {
        let mut bt = BinaryTree::from(4u32);
        // Move to leaf index 3 explicitly
        let leaf = bt.leaves_[3].clone();
        bt.move_to(leaf);
        let idx = bt.get_leaf_idx(None);
        assert_eq!(idx, 3);
    }

    #[test]
    fn test_get_leaf_idx_traversal_with_random_r() {
        let mut bt = BinaryTree::from(4u32);
        // Assign weights: leaf0=1, leaf1=2, leaf2=3, leaf3=4 (total=10)
        for (i, w) in [1.0, 2.0, 3.0, 4.0].iter().enumerate() {
            bt.update_value(*w, Some(i));
        }
        // Helper closure to test traversal (get_leaf_idx(Some(r)) now returns the chosen leaf)
        let mut check = |r: f64, expected_leaf: usize| {
            bt.reset_cur_node();
            let leaf_idx = bt.get_leaf_idx(Some(r));
            assert_eq!(
                leaf_idx, expected_leaf,
                "r={} expected leaf {}",
                r, expected_leaf
            );
            // cur_node_ reset to root by get_leaf_idx(Some(_)) implementation
            assert!(bt.is_root());
        };
        // cumulative probs: [0.1, 0.3, 0.6, 1.0]
        check(0.05, 0);
        check(0.25, 1);
        check(0.45, 2);
        check(0.95, 3);
    }

    #[test]
    fn test_move_up_chain_to_root() {
        let mut bt = BinaryTree::from(4u32);
        bt.move_down_right();
        bt.move_down_left(); // navigate two levels
        assert!(!bt.is_root());
        bt.move_up();
        assert!(!bt.is_root());
        bt.move_up();
        assert!(bt.is_root());
    }
}
