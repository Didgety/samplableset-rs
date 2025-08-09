mod binary_tree;
mod hash_propensity;
mod samplable_set;

use binary_tree::BinaryTree;
use hash_propensity::HashPropensity;

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
