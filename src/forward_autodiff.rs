use crate::prelude::*;
use crate::{DualNum, DualNumFloat};

#[derive(Copy, Clone, Debug)]
pub struct Dual<T: DualNumFloat> {
    pub re: T,
    pub eps: T,
}

pub type Dual32 = Dual<f32>;
pub type Dual64 = Dual<f64>;

impl<T> DualNum<T> for Dual<T>
where
    T: DualNumFloat,
{
    #[cfg(any(feature = "std", feature = "libm"))]
    fn sin(&self) -> Self {
        let (s, c) = self.re.sin_cos();
        self.chain_rule(s, c)
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    fn cos(&self) -> Self {
        let (s, c) = self.re.sin_cos();
        self.chain_rule(c, -s)
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    fn sin_cos(&self) -> (Self, Self) {
        let (s, c) = self.re.sin_cos();
        (self.chain_rule(s, c), self.chain_rule(c, -s))
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    fn sqrt(&self) -> Self {
        let rec = self.re.recip();
        let half = T::from(0.5).unwrap();
        let f0 = self.re.sqrt();
        let f1 = f0 * rec * half;
        self.chain_rule(f0, f1)
    }
}

impl<T> From<T> for Dual<T>
where
    T: DualNumFloat,
{
    fn from(value: T) -> Self {
        Self::from_re(value)
    }
}

impl<T> Add<T> for Dual<T>
where
    T: DualNumFloat,
{
    type Output = Dual<T>;

    fn add(self, rhs: T) -> Self::Output {
        self + Self::from_re(rhs)
    }
}

impl<T> Sub<T> for Dual<T>
where
    T: DualNumFloat,
{
    type Output = Dual<T>;

    fn sub(self, rhs: T) -> Self::Output {
        self - Self::from_re(rhs)
    }
}

impl<T> Mul<T> for Dual<T>
where
    T: DualNumFloat,
{
    type Output = Dual<T>;

    fn mul(self, rhs: T) -> Self::Output {
        self * Self::from_re(rhs)
    }
}

impl<T> Div<T> for Dual<T>
where
    T: DualNumFloat,
{
    type Output = Dual<T>;

    fn div(self, rhs: T) -> Self::Output {
        self / Self::from_re(rhs)
    }
}

impl<T> Rem<T> for Dual<T>
where
    T: DualNumFloat,
{
    type Output = Dual<T>;

    fn rem(self, rhs: T) -> Self::Output {
        self % Self::from_re(rhs)
    }
}

impl<T: DualNumFloat> Dual<T> {
    #[inline]
    pub fn new(re: T, eps: T) -> Self {
        Self { re, eps }
    }
}

impl<T: DualNumFloat + Zero> Dual<T> {
    /// Create a new dual number from the real part.
    #[inline]
    pub fn from_re(re: T) -> Self {
        Self::new(re, T::zero())
    }
}

impl<T: DualNumFloat + One> Dual<T> {
    /// Set the derivative part to 1.
    /// ```
    /// # use num_dual::{Dual64, DualNum};
    /// let x = Dual64::from_re(5.0).derivative().powi(2);
    /// assert_eq!(x.re, 25.0);
    /// assert_eq!(x.eps, 10.0);
    /// ```
    #[inline]
    pub fn derivative(mut self) -> Self {
        self.eps = T::one();
        self
    }
}

impl<T: DualNumFloat + PartialEq> PartialEq for Dual<T> {
    fn eq(&self, other: &Self) -> bool {
        self.re.eq(&other.re)
    }
}

impl<T: DualNumFloat> PartialOrd for Dual<T> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.re.partial_cmp(&other.re)
    }
}

impl<T: DualNumFloat> Signed for Dual<T> {
    #[inline]
    fn abs(&self) -> Self {
        if self.is_positive() {
            self.clone()
        } else {
            -self.clone()
        }
    }

    #[inline]
    fn abs_sub(&self, other: &Self) -> Self {
        if self.re > other.re {
            self.clone() - other.clone()
        } else {
            Self::zero()
        }
    }

    #[inline]
    fn signum(&self) -> Self {
        if self.is_positive() {
            Self::one()
        } else if self.is_zero() {
            Self::zero()
        } else {
            -Self::one()
        }
    }

    #[inline]
    fn is_positive(&self) -> bool {
        self.re > T::zero()
    }

    #[inline]
    fn is_negative(&self) -> bool {
        self.re < T::zero()
    }
}

impl<T: DualNumFloat> Zero for Dual<T> {
    #[inline]
    fn zero() -> Self {
        Self::from_re(T::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.re.is_zero()
    }
}

impl<T: DualNumFloat> One for Dual<T> {
    #[inline]
    fn one() -> Self {
        Self::from_re(T::one())
    }

    #[inline]
    fn is_one(&self) -> bool {
        self.re.is_one()
    }
}

impl<T: DualNumFloat> Neg for Dual<T> {
    type Output = Dual<T>;

    fn neg(self) -> Self::Output {
        Self::new(-self.re, -self.eps)
    }
}

/* chain rule */
impl<T: DualNumFloat> Dual<T> {
    #[inline]
    fn chain_rule(&self, f0: T, f1: T) -> Self {
        Self::new(f0, self.eps * f1)
    }
}

/* product rule */
impl<T: DualNumFloat> Mul for Dual<T> {
    type Output = Dual<T>;

    #[inline]
    fn mul(self, rhs: Dual<T>) -> Self::Output {
        Dual {
            re: self.re * rhs.re,
            eps: self.eps * rhs.re + self.re * rhs.eps,
        }
    }
}

/* quotient rule */
impl<T: DualNumFloat> Div<Dual<T>> for Dual<T> {
    type Output = Dual<T>;
    #[inline]
    fn div(self, other: Dual<T>) -> Dual<T> {
        let inv = other.re.recip();
        Dual::new(
            self.re * inv,
            (self.eps * other.re - other.eps * self.re) * inv * inv,
        )
    }
}

// impl<T: DualNumFloat> Dual<T> {
//     #[cfg(any(feature = "std", feature = "libm"))]
//     #[inline]
//     pub fn sin(&self) -> Self {
//         let (s, c) = self.re.sin_cos();
//         self.chain_rule(s, c)
//     }

//     #[cfg(any(feature = "std", feature = "libm"))]
//     #[inline]
//     pub fn cos(&self) -> Self {
//         let (s, c) = self.re.sin_cos();
//         self.chain_rule(c, -s)
//     }
// }

impl<T: DualNumFloat> fmt::Display for Dual<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} + {}ε", self.re, self.eps)
    }
}

impl<T: DualNumFloat> Add for Dual<T> {
    type Output = Dual<T>;

    fn add(self, rhs: Self) -> Self::Output {
        Dual {
            re: self.re + rhs.re,
            eps: self.eps + rhs.eps,
        }
    }
}

impl<T: DualNumFloat> Sub for Dual<T> {
    type Output = Dual<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        Dual {
            re: self.re - rhs.re,
            eps: self.eps - rhs.eps,
        }
    }
}

impl<T: DualNumFloat> Rem for Dual<T> {
    type Output = Dual<T>;

    fn rem(self, _rhs: Self) -> Self::Output {
        unimplemented!()
    }
}

impl<T: DualNumFloat> MulAssign for Dual<T> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl<T: DualNumFloat> DivAssign for Dual<T> {
    fn div_assign(&mut self, rhs: Self) {
        *self = self.clone() / rhs;
    }
}

impl<T: DualNumFloat> AddAssign for Dual<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.re = self.re + rhs.re;
        self.eps = self.eps + rhs.eps;
    }
}

impl<T: DualNumFloat> SubAssign for Dual<T> {
    fn sub_assign(&mut self, rhs: Self) {
        self.re = self.re - rhs.re;
        self.eps = self.eps - rhs.eps;
    }
}

impl<T: DualNumFloat> RemAssign for Dual<T> {
    fn rem_assign(&mut self, _rhs: Self) {
        unimplemented!()
    }
}

impl<T: DualNumFloat> Num for Dual<T> {
    type FromStrRadixErr = ();

    fn from_str_radix(_s: &str, _radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        unimplemented!()
    }
}

impl<T> SimdValue for Dual<T>
where
    T: DualNumFloat,
{
    type Element = Dual<T::Element>;
    type SimdBool = T::SimdBool;
    const LANES: usize = T::LANES;

    #[inline]
    fn splat(val: Self::Element) -> Self {
        // Need to make lanes copies of each of:
        // - the real part
        // - each of the N epsilon parts
        let re = T::splat(val.re);
        let eps = T::splat(val.eps);
        Self::new(re, eps)
    }

    #[inline]
    fn extract(&self, i: usize) -> Self::Element {
        let re = self.re.extract(i);
        let eps = self.eps.extract(i);
        Self::Element { re, eps }
    }

    #[inline]
    unsafe fn extract_unchecked(&self, i: usize) -> Self::Element {
        let re = unsafe { self.re.extract_unchecked(i) };
        let eps = unsafe { self.eps.extract_unchecked(i) };
        Self::Element { re, eps }
    }

    #[inline]
    fn replace(&mut self, i: usize, val: Self::Element) {
        self.re.replace(i, val.re);
        self.eps.replace(i, val.eps);
    }

    #[inline]
    unsafe fn replace_unchecked(&mut self, i: usize, val: Self::Element) {
        unsafe { self.re.replace_unchecked(i, val.re) };
        unsafe { self.eps.replace_unchecked(i, val.eps) };
    }

    #[inline]
    fn select(self, cond: Self::SimdBool, other: Self) -> Self {
        let re = self.re.select(cond, other.re);
        let eps = self.eps.select(cond, other.eps);
        Self::new(re, eps)
    }
}

impl<T> nalgebra::Field for Dual<T> where T: DualNumFloat {}

impl<T> FromPrimitive for Dual<T>
where
    T: DualNumFloat,
{
    fn from_i64(n: i64) -> Option<Self> {
        Some(Self::from_re(T::from_i64(n)?))
    }

    fn from_u64(n: u64) -> Option<Self> {
        Some(Self::from_re(T::from_u64(n)?))
    }

    fn from_f32(n: f32) -> Option<Self> {
        Some(Self::from_re(T::from_f32(n)?))
    }

    fn from_f64(n: f64) -> Option<Self> {
        Some(Self::from_re(T::from_f64(n)?))
    }
}

impl<T> approx::AbsDiffEq for Dual<T>
where
    T: DualNumFloat + approx::AbsDiffEq<Epsilon = T>,
{
    type Epsilon = Self;

    fn default_epsilon() -> Self::Epsilon {
        Self::from_re(T::default_epsilon())
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.re.abs_diff_eq(&other.re, epsilon.re)
    }
}

impl<T> approx::RelativeEq for Dual<T>
where
    T: DualNumFloat + approx::AbsDiffEq<Epsilon = T>,
{
    fn default_max_relative() -> Self::Epsilon {
        todo!()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        todo!()
    }
}

impl<T> approx::UlpsEq for Dual<T>
where
    T: DualNumFloat + approx::UlpsEq + approx::AbsDiffEq<Epsilon = T>,
{
    fn default_max_ulps() -> u32 {
        todo!()
    }

    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        todo!()
    }
}

impl<T> simba::scalar::SupersetOf<f32> for Dual<T>
where
    T: DualNumFloat + simba::scalar::SupersetOf<f32>,
{
    #[inline(always)]
    fn is_in_subset(&self) -> bool {
        self.re.is_in_subset()
    }

    #[inline(always)]
    fn to_subset_unchecked(&self) -> f32 {
        self.re.to_subset_unchecked()
    }

    #[inline(always)]
    fn from_subset(element: &f32) -> Self {
        let re = T::from_subset(element);
        let eps = T::zero();
        Self::new(re, eps)
    }
}

impl<T> simba::scalar::SupersetOf<f64> for Dual<T>
where
    T: DualNumFloat + simba::scalar::SupersetOf<f64>,
{
    #[inline(always)]
    fn is_in_subset(&self) -> bool {
        self.re.is_in_subset()
    }

    #[inline(always)]
    fn to_subset_unchecked(&self) -> f64 {
        self.re.to_subset_unchecked()
    }

    #[inline(always)]
    fn from_subset(element: &f64) -> Self {
        let re = T::from_subset(element);
        let eps = T::zero();
        Self::new(re, eps)
    }
}

impl<FSuper, F> simba::scalar::SubsetOf<Dual<FSuper>> for Dual<F>
where
    FSuper: DualNumFloat + simba::scalar::SupersetOf<F>,
    F: DualNumFloat,
{
    #[inline(always)]
    fn to_superset(&self) -> Dual<FSuper> {
        let re = FSuper::from_subset(&self.re);
        let eps = FSuper::from_subset(&self.eps);
        Dual { re, eps }
    }

    #[inline(always)]
    fn from_superset_unchecked(element: &Dual<FSuper>) -> Self {
        let re = FSuper::to_subset_unchecked(&element.re);
        let eps = FSuper::to_subset_unchecked(&element.eps);
        Self::new(re, eps)
    }

    #[inline(always)]
    fn is_in_subset(element: &Dual<FSuper>) -> bool {
        FSuper::is_in_subset(&element.re) && FSuper::is_in_subset(&element.eps)
    }
}

impl<T> ComplexField for Dual<T>
where
    T: DualNumFloat,
    // T: simba::scalar::SubsetOf<Dual<T>>,
    T: simba::scalar::SupersetOf<T::Element>,
    T: simba::scalar::SupersetOf<T>,
    T: simba::scalar::SupersetOf<f32>,
    T: simba::scalar::SupersetOf<f64>,
    T: approx::RelativeEq + approx::UlpsEq + approx::AbsDiffEq<Epsilon = T>,
{
    type RealField = Self;

    #[doc = r" Builds a pure-real complex number from the given value."]
    fn from_real(re: Self::RealField) -> Self {
        re
    }

    #[doc = r" The real part of this complex number."]
    fn real(self) -> Self::RealField {
        self
    }

    #[doc = r" The imaginary part of this complex number."]
    fn imaginary(self) -> Self::RealField {
        Self::zero()
    }

    #[doc = r" The modulus of this complex number."]
    fn modulus(self) -> Self::RealField {
        self.abs()
    }

    #[doc = r" The squared modulus of this complex number."]
    fn modulus_squared(self) -> Self::RealField {
        self * self
    }

    #[doc = r" The argument of this complex number."]
    fn argument(self) -> Self::RealField {
        todo!()
    }

    #[doc = r" The sum of the absolute value of this complex number's real and imaginary part."]
    fn norm1(self) -> Self::RealField {
        self.abs()
    }

    #[doc = r" Multiplies this complex number by `factor`."]
    fn scale(self, factor: Self::RealField) -> Self {
        self * factor
    }

    #[doc = r" Divides this complex number by `factor`."]
    fn unscale(self, factor: Self::RealField) -> Self {
        self / factor
    }

    fn floor(self) -> Self {
        panic!("called floor() on a dual number")
    }

    fn ceil(self) -> Self {
        panic!("called ceil() on a dual number")
    }

    fn round(self) -> Self {
        panic!("called round() on a dual number")
    }

    fn trunc(self) -> Self {
        panic!("called trunc() on a dual number")
    }

    fn fract(self) -> Self {
        panic!("called fract() on a dual number")
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        todo!("mul_add() not yet implemented for Dual numbers");
    }

    #[doc = r" The absolute value of this complex number: `self / self.signum()`."]
    #[doc = r""]
    #[doc = r" This is equivalent to `self.modulus()`."]
    fn abs(self) -> Self::RealField {
        Signed::abs(&self)
    }

    #[doc = r" Computes (self.conjugate() * self + other.conjugate() * other).sqrt()"]
    fn hypot(self, other: Self) -> Self::RealField {
        todo!("hypot() not yet implemented for Dual numbers");
    }

    fn recip(self) -> Self {
        todo!("recip() not yet implemented for Dual numbers");
    }

    fn conjugate(self) -> Self {
        self
    }

    fn sin(self) -> Self {
        #[cfg(not(any(feature = "std", feature = "libm")))]
        panic!("sin() not available because neither the 'std' nor the 'libm' feature is enabled");

        #[cfg(any(feature = "std", feature = "libm"))]
        DualNum::sin(&self)
    }

    fn cos(self) -> Self {
        #[cfg(not(any(feature = "std", feature = "libm")))]
        panic!("cos() not available because neither the 'std' nor the 'libm' feature is enabled");

        #[cfg(any(feature = "std", feature = "libm"))]
        DualNum::cos(&self)
    }

    fn sin_cos(self) -> (Self, Self) {
        #[cfg(not(any(feature = "std", feature = "libm")))]
        panic!(
            "sin_cos() not available because neither the 'std' nor the 'libm' feature is enabled"
        );

        #[cfg(any(feature = "std", feature = "libm"))]
        DualNum::sin_cos(&self)
    }

    fn tan(self) -> Self {
        todo!("tan() not yet implemented for Dual numbers");
    }

    fn asin(self) -> Self {
        todo!("asin() not yet implemented for Dual numbers");
    }

    fn acos(self) -> Self {
        todo!("acos() not yet implemented for Dual numbers");
    }

    fn atan(self) -> Self {
        todo!("atan() not yet implemented for Dual numbers");
    }

    fn sinh(self) -> Self {
        todo!("sinh() not yet implemented for Dual numbers");
    }

    fn cosh(self) -> Self {
        todo!("cosh() not yet implemented for Dual numbers");
    }

    fn tanh(self) -> Self {
        todo!("tanh() not yet implemented for Dual numbers");
    }

    fn asinh(self) -> Self {
        todo!("asinh() not yet implemented for Dual numbers");
    }

    fn acosh(self) -> Self {
        todo!("acosh() not yet implemented for Dual numbers");
    }

    fn atanh(self) -> Self {
        todo!("atanh() not yet implemented for Dual numbers");
    }

    fn log(self, base: Self::RealField) -> Self {
        todo!("log() not yet implemented for Dual numbers");
    }

    fn log2(self) -> Self {
        todo!("log2() not yet implemented for Dual numbers");
    }

    fn log10(self) -> Self {
        todo!("log10() not yet implemented for Dual numbers");
    }

    fn ln(self) -> Self {
        todo!("ln() not yet implemented for Dual numbers");
    }

    fn ln_1p(self) -> Self {
        todo!("ln_1p() not yet implemented for Dual numbers");
    }

    fn sqrt(self) -> Self {
        #[cfg(not(any(feature = "std", feature = "libm")))]
        panic!("sqrt() not available because neither the 'std' nor the 'libm' feature is enabled");

        #[cfg(any(feature = "std", feature = "libm"))]
        DualNum::sqrt(&self)
    }

    fn exp(self) -> Self {
        todo!("exp() not yet implemented for Dual numbers");
    }

    fn exp2(self) -> Self {
        todo!("exp2() not yet implemented for Dual numbers");
    }

    fn exp_m1(self) -> Self {
        todo!("exp_m1() not yet implemented for Dual numbers");
    }

    fn powi(self, n: i32) -> Self {
        todo!("powi() not yet implemented for Dual numbers");
    }

    fn powf(self, n: Self::RealField) -> Self {
        todo!("powf() not yet implemented for Dual numbers");
    }

    fn powc(self, n: Self) -> Self {
        todo!("powc() not yet implemented for Dual numbers");
    }

    fn cbrt(self) -> Self {
        todo!("cbrt() not yet implemented for Dual numbers");
    }

    fn is_finite(&self) -> bool {
        todo!("is_finite() not yet implemented for Dual numbers");
    }

    fn try_sqrt(self) -> Option<Self> {
        todo!("try_sqrt() not yet implemented for Dual numbers");
    }
}

impl<T> RealField for Dual<T>
where
    T: DualNumFloat,
    // T: simba::scalar::SubsetOf<Dual<T>>,
    T: simba::scalar::SupersetOf<T::Element>,
    T: simba::scalar::SupersetOf<T>,
    T: simba::scalar::SupersetOf<f32>,
    T: simba::scalar::SupersetOf<f64>,
    T: approx::RelativeEq + approx::UlpsEq + approx::AbsDiffEq<Epsilon = T>,
{
    #[inline]
    fn copysign(self, sign: Self) -> Self {
        if sign.re.is_sign_positive() {
            self.abs()
        } else {
            -self.abs()
        }
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        todo!()
    }

    #[inline]
    fn pi() -> Self {
        Self::from_re(<T as FloatConst>::PI())
    }

    #[inline]
    fn two_pi() -> Self {
        Self::from_re(<T as FloatConst>::TAU())
    }

    #[inline]
    fn frac_pi_2() -> Self {
        Self::from_re(<T as FloatConst>::FRAC_PI_4())
    }

    #[inline]
    fn frac_pi_3() -> Self {
        Self::from_re(<T as FloatConst>::FRAC_PI_3())
    }

    #[inline]
    fn frac_pi_4() -> Self {
        Self::from_re(<T as FloatConst>::FRAC_PI_4())
    }

    #[inline]
    fn frac_pi_6() -> Self {
        Self::from_re(<T as FloatConst>::FRAC_PI_6())
    }

    #[inline]
    fn frac_pi_8() -> Self {
        Self::from_re(<T as FloatConst>::FRAC_PI_8())
    }

    #[inline]
    fn frac_1_pi() -> Self {
        Self::from_re(<T as FloatConst>::FRAC_1_PI())
    }

    #[inline]
    fn frac_2_pi() -> Self {
        Self::from_re(<T as FloatConst>::FRAC_2_PI())
    }

    #[inline]
    fn frac_2_sqrt_pi() -> Self {
        Self::from_re(<T as FloatConst>::FRAC_2_SQRT_PI())
    }

    #[inline]
    fn e() -> Self {
        Self::from_re(<T as FloatConst>::E())
    }

    #[inline]
    fn log2_e() -> Self {
        Self::from_re(<T as FloatConst>::LOG2_E())
    }

    #[inline]
    fn log10_e() -> Self {
        Self::from_re(<T as FloatConst>::LOG10_E())
    }

    #[inline]
    fn ln_2() -> Self {
        Self::from_re(<T as FloatConst>::LN_2())
    }

    #[inline]
    fn ln_10() -> Self {
        Self::from_re(<T as FloatConst>::LN_10())
    }

    #[inline]
    fn is_sign_positive(&self) -> bool {
        self.re.is_sign_positive()
    }

    #[inline]
    fn is_sign_negative(&self) -> bool {
        self.re.is_sign_negative()
    }

    /// Got to be careful using this, because it throws away the derivatives of the one not chosen
    #[inline]
    fn max(self, other: Self) -> Self {
        if other > self { other } else { self }
    }

    /// Got to be careful using this, because it throws away the derivatives of the one not chosen
    #[inline]
    fn min(self, other: Self) -> Self {
        if other < self { other } else { self }
    }

    /// If the min/max values are constants and the clamping has an effect, you lose your gradients.
    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        if self < min {
            min
        } else if self > max {
            max
        } else {
            self
        }
    }

    #[inline]
    fn min_value() -> Option<Self> {
        Some(Self::from_re(T::min_value()))
    }

    #[inline]
    fn max_value() -> Option<Self> {
        Some(Self::from_re(T::max_value()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_sqrt() {
        let x = Dual64::from_re(4.0).derivative();
        let y = x.sqrt();
        assert_eq!(y.re, 2.0);
        assert_eq!(y.eps, 0.25);
    }
}
