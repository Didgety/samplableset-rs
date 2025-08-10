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

#[derive(Debug, Clone)]
pub struct HashPropensity {
    propensity_min_: f64,
    propensity_max_: f64,
    power_of_two_: bool,
    // scale: f64 // precompute 1/(max-min) ?
}

impl HashPropensity {
    pub fn new(propensity_min: f64, propensity_max: f64) -> Self {
        assert!(
            propensity_max.is_finite() && propensity_min > 0.0,
            "Propensity min must be positive and less than infinity"
        );
        assert!(
            propensity_max.is_finite() && propensity_max > propensity_min,
            "Propensity max must be finite and greater than min"
        ); // TODO does it HAVE to be greater than min?

        let is_pow_two = is_pow_two_f64(propensity_max / propensity_min);
        // f64::floor(f64::log2(
        //     propensity_max / propensity_min
        // )) == f64::ceil(f64::log2(
        //     propensity_max / propensity_min
        // ));

        HashPropensity {
            propensity_min_: propensity_min,
            propensity_max_: propensity_max,
            power_of_two_: is_pow_two,
        }
    }

    #[inline]
    pub fn operator(&self, propensity: f64) -> usize {
        // TODO handle case where propensity < self.propensity_min_
        let mut idx: usize = f64::floor(f64::log2(propensity / self.propensity_min_)) as usize;

        // TODO test if epsilon needed
        if self.power_of_two_ && propensity == self.propensity_max_ {
            idx -= 1;
        }
        idx
    }
}

#[inline]
fn is_pow_two_f64(x: f64) -> bool {
    if !(x.is_finite()) || x <= 0.0 {
        return false;
    }
    // IEEE-754: a power of two has zero mantissa bits.
    // let bits = x.to_bits();
    // let exp = (bits >> 52) & 0x7FF;
    // let mant = bits & ((1u64 << 52) - 1);
    // exp != 0 && exp != 0x7ff && mant == 0
    (x.to_bits() & ((1u64 << 52) - 1)) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn test_new_invalid_min_zero() {
        HashPropensity::new(0.0, 8.0);
    }

    #[test]
    #[should_panic]
    fn test_new_invalid_max_le_min() {
        HashPropensity::new(2.0, 1.0);
    }

    #[test]
    fn test_is_pow_two_f64() {
        assert!(is_pow_two_f64(1.0));
        assert!(is_pow_two_f64(2.0));
        assert!(is_pow_two_f64(4.0));
        assert!(is_pow_two_f64(1024.0));
        assert!(!is_pow_two_f64(3.0));
        assert!(!is_pow_two_f64(0.0));
        assert!(!is_pow_two_f64(-2.0));
        assert!(!is_pow_two_f64(f64::NAN));
        assert!(!is_pow_two_f64(f64::INFINITY));
    }

    #[test]
    fn test_new_valid() {
        let hp = HashPropensity::new(1.0, 8.0);
        assert_eq!(hp.propensity_min_, 1.0);
        assert_eq!(hp.propensity_max_, 8.0);
        assert!(hp.power_of_two_);
    }

    #[test]
    fn test_operator_basic() {
        let hp = HashPropensity::new(1.0, 8.0);
        assert_eq!(hp.operator(1.0), 0);
        assert_eq!(hp.operator(2.0), 1);
        assert_eq!(hp.operator(4.0), 2);
        // propensity == max, power_of_two_ == true, so idx -= 1
        assert_eq!(hp.operator(8.0), 2);
        assert_eq!(hp.operator(3193.0), 11);
    }

    #[test]
    fn test_operator_non_pow_two() {
        let hp = HashPropensity::new(1.0, 10.0);
        assert!(!hp.power_of_two_);
        assert_eq!(hp.operator(1.0), 0);
        let idx = hp.operator(5.0);
        assert_eq!(idx, f64::floor(f64::log2(5.0 / 1.0)) as usize);
    }
}
