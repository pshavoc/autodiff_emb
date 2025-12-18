use core::{
    fmt,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign},
};

use nalgebra::{Scalar, SimdValue};
#[cfg(feature = "std")]
use num_traits::Float;

#[cfg(not(feature = "std"))]
use num_traits::float::FloatCore as Float;

use num_traits::{FloatConst, Num, NumOps, One, Signed, Zero};

pub trait DualNum<F>: NumOps<F> + Signed + 'static {
    /// Highest derivative that can be calculated with this struct
    const NDERIV: usize;
}

pub trait DualNumFloat:
    Float + FloatConst + core::fmt::Display + core::fmt::Debug + Sync + Send + 'static
{
    #[cfg(any(feature = "std", feature = "libm"))]
    fn sin_cos(&self) -> (Self, Self);
}

#[derive(Copy, Clone, Debug)]
pub struct Dual<T: DualNumFloat> {
    pub re: T,
    pub eps: T,
}

pub type Dual32 = Dual<f32>;
pub type Dual64 = Dual<f64>;

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

impl<T: DualNumFloat> Dual<T> {
    #[cfg(any(feature = "std", feature = "libm"))]
    #[inline]
    pub fn sin(&self) -> Self {
        let (s, c) = self.re.sin_cos();
        self.chain_rule(s, c)
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[inline]
    pub fn cos(&self) -> Self {
        let (s, c) = self.re.sin_cos();
        self.chain_rule(c, -s)
    }
}

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
    T: DualNumFloat + SimdValue,
    T::Element: DualNumFloat,
{
    type Element = Dual<T::Element>;
    type SimdBool = T::SimdBool;

    fn lanes() -> usize {
        T::lanes()
    }

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

impl DualNumFloat for f32 {
    #[cfg(feature = "libm")]
    fn sin_cos(&self) -> (Self, Self) {
        use libm::sincosf;
        sincosf(*self)
    }
}

// impl<T> nalgebra::Field for Dual<T> where T: DualNumFloat + SimdValue {}

impl nalgebra::Field for Dual32 {}
