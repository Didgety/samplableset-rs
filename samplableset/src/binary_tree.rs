use std::{
    cell::RefCell, collections::HashMap, ops::Deref, rc::{Rc, Weak}
};

type LeafIdx = usize;

#[derive(Debug, Clone)]
pub struct NodeRef(Rc<RefCell<BTreeNode>>);

impl NodeRef {
    pub fn new(node: BTreeNode) -> Self {
        NodeRef(Rc::new(RefCell::new(node)))
    }

    pub fn downgrade(&self) -> WeakNodeRef {
        WeakNodeRef(Rc::downgrade(&self))
    }
}

impl Deref for NodeRef {
    type Target = Rc<RefCell<BTreeNode>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug)]
pub struct WeakNodeRef(Weak<RefCell<BTreeNode>>);

impl WeakNodeRef {
    pub fn new() -> Self {
        WeakNodeRef(Weak::new())
    }

    pub fn upgrade(&self) -> Option<NodeRef> {
        self.0.upgrade().map(NodeRef)
    }
}

impl Deref for WeakNodeRef {
    type Target = Weak<RefCell<BTreeNode>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug)]
pub struct BTreeNode {
    val: f64,
    parent: WeakNodeRef,
    left: Option<NodeRef>,
    right: Option<NodeRef>,
}

impl BTreeNode {
    pub fn new() -> Self {
        Self {
            val: 0.0,
            parent: WeakNodeRef::new(),
            left: None,
            right: None,
        }
    }
}

pub struct BinaryTree {
    root_: Option<NodeRef>,
    cur_node_: Option<NodeRef>,
    leaves_: Vec<NodeRef>,
    leaves_idx_map_: HashMap<usize, LeafIdx>
}

impl BinaryTree {
    pub fn new() -> Self {
        BinaryTree {
            root_: None,
            cur_node_: None,
            leaves_: Vec::new(),
            leaves_idx_map_: HashMap::new()
        }
    }

    pub fn is_root(&self) -> bool {
        self.cur_node_
            .as_ref()
            // default to false if cur_node_ is None
            .map_or(false, |n| 
                n.borrow().parent.upgrade().is_none())
    }

    pub fn is_leaf(&self) -> bool {
        self.cur_node_
            .as_ref()
            // default to false if cur_node_ is None
            .map_or(false, |n| {
                let nb = n.borrow();
                nb.left.is_none() && nb.right.is_none()
            })
    }

    pub fn get_val(&self) -> Option<f64> {       
        self.cur_node_
            .as_ref()
            .map_or(None, |n| Some(n.borrow().val))
    }

    pub fn get_left_val(&self) -> Option<f64> {
        self.cur_node_
            .as_ref()
            .and_then(|n| n.borrow().left.as_ref().map(|l| l.borrow().val))
    }

    pub fn get_right_val(&self) -> Option<f64> {
        self.cur_node_
            .as_ref()
            .and_then(|n| n.borrow().right.as_ref().map(|r| r.borrow().val))
    }

    pub fn get_leaf_idx(&mut self, r: Option<f64>) -> LeafIdx {        
        match r {
            Some(r_val) => {
                let mut cumul: f64 = 0.0;
                let total_val: f64 = self.get_val().unwrap();
                while !self.is_leaf() {
                    if r_val <= (cumul + self.get_left_val().unwrap()) / total_val {
                        self.move_down_left();
                    } else {
                        cumul += self.get_left_val().unwrap();
                        self.move_down_right();
                    }
                }
                0
            }
            None => {
                let n = self.cur_node_.as_ref().expect("current_node_ is None");
                let key = Rc::as_ptr(&n) as *const RefCell<BTreeNode> as usize;
                *self
                    .leaves_idx_map_
                    .get(&key)
                    .expect("current_node_ is not a leaf (no leaf index)")
            }
        }                
    }

    pub fn reset_cur_node(&mut self) {
        self.cur_node_ = self.root_.clone();
    }

    pub fn move_down_left(&mut self) {
        let next = self.cur_node_.as_ref().expect("current node is None")
            .borrow()
            .left
            .clone()
            .expect("cannot move down left: no left child");
        self.cur_node_ = Some(next);
    }

    pub fn move_down_right(&mut self) {
        let next = self.cur_node_.as_ref().expect("current node is None")
            .borrow()
            .right
            .clone()
            .expect("cannot move down right: no right child");
        self.cur_node_ = Some(next);
    }

    pub fn move_up(&mut self) {
        let next = {
        let cur = self.cur_node_.as_ref().expect("current node is None");
        let parent_rc = cur
                .borrow()
                .parent
                .upgrade()
                .expect("cannot move up: no parent");
            parent_rc
        };
        self.cur_node_ = Some(next);
    }

    pub fn move_to(&mut self, node: NodeRef) {
        self.cur_node_ = Some(node);
    }

    pub fn update_value(&mut self, variation: f64, idx: Option<LeafIdx>) {
        match idx {
            Some(leaf_idx) => {
                let node = self
                    .leaves_
                    .get(leaf_idx)
                    .cloned()
                    .expect("invalid leaf index");
                self.cur_node_ = Some(node);
                self._update_helper(variation);
            }
            None => {
                if self.is_leaf() {
                    self._update_helper(variation);
                } else {
                    println!("Cannot update value: current node is not a leaf");
                }
            }
        }
    }

    fn _update_helper(&mut self, variation: f64) {
        {
            let cur = self
                .cur_node_
                .as_ref()
                .expect("current node is None in _update_helper");
            cur.borrow_mut().val += variation;
        }

        // walk up to the root, adding at each ancestor
        while !self.is_root() {
            self.move_up(); // mutates self.cur_node_
            let cur = self
                .cur_node_
                .as_ref()
                .expect("current node is None after move_up");
            cur.borrow_mut().val += variation;
        }
    }

    pub fn update_zero(&mut self) {
        if !self.is_leaf() {
            println!("Cannot zero: current node is not a leaf");
        }
        self.cur_node_
            .as_ref()
            .expect("current node is None in update_zero")
            .borrow_mut()
            .val = 0.0;
        while !self.is_root() {
            self.move_up();
            let cur = self
                .cur_node_
                .as_ref()
                .expect("current node is None after move_up in update_zero");
            cur.borrow_mut().val = 0.0;
        }
    }

    pub fn clear(&mut self) {
        let leaves = self.leaves_.clone();
        for leaf in leaves {
            self.move_to(leaf);
            self.update_zero();
        }
    }

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
                let mut cb = child.borrow_mut();
                cb.left = left;
                cb.right = right;
            }

            Some(child)
        } else {
            let key = Rc::as_ptr(&parent) as *const _ as usize;
            if !self.leaves_idx_map_.contains_key(&key) {
                let idx = self.leaves_.len() as LeafIdx;
                self.leaves_idx_map_.insert(key, idx);
                self.leaves_.push(parent.clone());
            }
            None
        }
    }

    // TODO needed for C++ or python interop?
    fn destroy_tree(node: NodeRef) {
        unimplemented!()
    }
}

// TODO expand to u64+?
impl<T: Into<u32>> From<T> for BinaryTree {
    fn from(n_leaves: T) -> Self {
        let n_leaves: u32 = n_leaves.into();
        if n_leaves < 1 {
            panic!("BinaryTree must have at least one leaf");
        }
        
        let root = NodeRef::new(BTreeNode::new());
        
        let mut tree = BinaryTree {
            root_: Some(root.clone()),
            cur_node_: Some(root.clone()),
            leaves_: Vec::with_capacity(n_leaves as usize),
            leaves_idx_map_: HashMap::new()
        };

        let n_nodes = n_leaves
            .checked_mul(2)
            .and_then(|v| v.checked_sub(1))
            .expect("Overflow calculating number of nodes");

        // this should never fail since we are constructing a new tree
        let left = BinaryTree::branch(&mut tree, root.clone(), 1, n_nodes).unwrap();
        let right = BinaryTree::branch(&mut tree, root.clone(), 2, n_nodes).unwrap();

        root.borrow_mut().left = Some(left.clone());
        root.borrow_mut().right = Some(right.clone());

        tree
    }
}

impl Clone for BinaryTree {
    /// !!! DOES NOT BEHAVE LIKE NORMAL RUST CLONING OF POINTERS !!!
    /// Deep copies the BinaryTree, creating a replica pointing at new nodes
    fn clone(&self) -> Self {
        let n_leaves = self.leaves_.len();
        assert!(n_leaves > 0, "BinaryTree must have at least one leaf");

        let root = NodeRef::new(BTreeNode::new());

        let mut out = BinaryTree {
            root_: Some(root.clone()),
            cur_node_: Some(root.clone()),
            leaves_: Vec::with_capacity(n_leaves),
            leaves_idx_map_: HashMap::new()
        };
        
        let n_nodes = (n_leaves as u32)
            .checked_mul(2)
            .and_then(|v| v.checked_sub(1))
            .expect("Overflow calculating number of nodes");

        let left = BinaryTree::branch(&mut out, root.clone(), 1, n_nodes);
        let right = BinaryTree::branch(&mut out, root.clone(), 2, n_nodes);

        {
            let mut rb = root.borrow_mut();
            rb.left = left;
            rb.right = right;
        }

        for (leaf_idx, src_leaf) in self.leaves_.iter().enumerate() {
            let v = src_leaf.borrow().val;
            BinaryTree::update_value(&mut out, v, Some(leaf_idx as LeafIdx));
        }

        out
    }
}

impl PartialEq for BinaryTree {
    fn eq(&self, other: &Self) -> bool {
        
        false
    }
}

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
        assert_eq!(tree.get_val().unwrap(), 8.0);
        // Check leaf values
        tree.move_down_left();
        assert_eq!(tree.get_val().unwrap(), 5.0);
        tree.reset_cur_node();
        tree.move_down_right();
        assert_eq!(tree.get_val().unwrap(), 3.0);
    }

    #[test]
    fn test_update_zero_and_clear() {
        let mut tree = BinaryTree::from(2u32);
        tree.update_value(2.0, Some(0));
        tree.update_value(4.0, Some(1));
        tree.clear();
        tree.reset_cur_node();
        assert_eq!(tree.get_val().unwrap(), 0.0);
        tree.move_down_left();
        assert_eq!(tree.get_val().unwrap(), 0.0);
        tree.reset_cur_node();
        tree.move_down_right();
        assert_eq!(tree.get_val().unwrap(), 0.0);
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
        assert_eq!(c.get_val().unwrap(), 8.0);
        c.move_down_left();
        assert_eq!(c.get_val().unwrap(), 7.0);
        c.reset_cur_node();
        c.move_down_right();
        assert_eq!(c.get_val().unwrap(), 1.0);
    }

    #[test]
    #[should_panic]
    fn test_binary_tree_from_zero_leaves_panics() {
        let _ = BinaryTree::from(0u32);
    }
}