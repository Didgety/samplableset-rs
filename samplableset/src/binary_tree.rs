use std::{
    cell::RefCell, collections::HashMap, rc::{Rc, Weak}
};

type LeafIdx = u32;

#[derive(Debug, Clone)]
pub struct NodeRef(Rc<RefCell<BTreeNode>>);

impl NodeRef {
    pub fn new(node: BTreeNode) -> Self {
        NodeRef(Rc::new(RefCell::new(node)))
    }

    pub fn downgrade(&self) -> WeakNodeRef {
        WeakNodeRef(Rc::downgrade(&self.0))
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

#[derive(Debug)]
struct BTreeNode {
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
                n.0.borrow().parent.0.upgrade().is_none())
    }

    pub fn is_leaf(&self) -> bool {
        self.cur_node_
            .as_ref()
            // default to false if cur_node_ is None
            .map_or(false, |n| {
                let nb = n.0.borrow();
                nb.left.is_none() && nb.right.is_none()
            })
    }

    pub fn get_val(&self) -> Option<f64> {       
        self.cur_node_
            .as_ref()
            .map_or(None, |n| Some(n.0.borrow().val))
    }

    // should be const
    pub fn get_left_val(&self) -> Option<f64> {
        self.cur_node_
            .as_ref()
            .and_then(|n| n.0.borrow().left.as_ref().map(|l| l.0.borrow().val))
    }

    pub fn get_right_val(&self) -> Option<f64> {
        self.cur_node_
            .as_ref()
            .and_then(|n| n.0.borrow().right.as_ref().map(|r| r.0.borrow().val))
    }

    pub fn get_leaf_idx(&mut self, r: Option<f64>) -> LeafIdx {        
        match r {
            Some(mut r_val) => {
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
                let key = Rc::as_ptr(&n.0) as *const RefCell<BTreeNode> as usize;
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
            .0
            .borrow()
            .left
            .clone()
            .expect("cannot move down left: no left child");
        self.cur_node_ = Some(next);
    }

    pub fn move_down_right(&mut self) {
        let next = self.cur_node_.as_ref().expect("current node is None")
            .0
            .borrow()
            .right
            .clone()
            .expect("cannot move down right: no right child");
        self.cur_node_ = Some(next);
    }

    pub fn move_up(&mut self) {
        let next = self.cur_node_.as_ref().expect("current node is None")
            .0
            .borrow()
            .parent
            .0
            .upgrade()
            .expect("cannot move up: no parent");
        self.cur_node_ = Some(next);
    }

    pub fn move_at(node: BTreeNode) {
        unimplemented!()
    }

    pub fn update_value_at(idx: LeafIdx, variation: f64) {
        unimplemented!()
    }

    pub fn update_value(variation: f64) {
        unimplemented!()
    }

    pub fn update_zero() {
        unimplemented!()
    }

    pub fn clear() {
        unimplemented!()
    }

    fn branch(parent: NodeRef, node_idx: u32, n_nodes: u32) -> NodeRef {
        unimplemented!()
    }

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
        
        let tree = BinaryTree {
            root_: Some(root.clone()),
            cur_node_: Some(root.clone()),
            leaves_: Vec::with_capacity(n_leaves as usize),
            leaves_idx_map_: HashMap::new()
        };

        let n_nodes = n_leaves
            .checked_mul(2)
            .and_then(|v| v.checked_sub(1))
            .expect("Overflow calculating number of nodes");

        let left = BinaryTree::branch(root.clone(), 1, n_nodes);
        let right = BinaryTree::branch(root.clone(), 2, n_nodes);

        root.0.borrow_mut().left = Some(left.clone());
        root.0.borrow_mut().right = Some(right.clone());

        tree
    }
}

impl Clone for BinaryTree {
    fn clone(&self) -> Self {
        let n_leaves = self.leaves_.len();
        assert!(n_leaves > 0, "BinaryTree must have at least one leaf");

        let root = NodeRef::new(BTreeNode::new());

        let out = BinaryTree {
            root_: Some(root.clone()),
            cur_node_: Some(root.clone()),
            leaves_: Vec::with_capacity(n_leaves),
            leaves_idx_map_: HashMap::new()
        };
        
        let n_nodes = (n_leaves as u32)
            .checked_mul(2)
            .and_then(|v| v.checked_sub(1))
            .expect("Overflow calculating number of nodes");

        let left = BinaryTree::branch(root.clone(), 1, n_nodes);
        let right = BinaryTree::branch(root.clone(), 2, n_nodes);

        {
            let mut rb = root.0.borrow_mut();
            rb.left = Some(left);
            rb.right = Some(right);
        }

        for (leaf_idx, src_leaf) in self.leaves_.iter().enumerate() {
            let v = src_leaf.0.borrow().val;
            BinaryTree::update_value_at(leaf_idx as LeafIdx, v);
        }

        out
    }
}

impl PartialEq for BinaryTree {
    fn eq(&self, other: &Self) -> bool {
        
        false
    }
}