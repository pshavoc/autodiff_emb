use crate::prelude::*;
use crate::{DualNum, DualNumFloat};

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

impl<T: DualNumFloat> PartialOrd for Dual<T> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.re.partial_cmp(&other.re)
    }
}

impl<T: DualNumFloat> Signed for Dual<T>
{
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

impl<T> nalgebra::Field for Dual<T>
    where T: DualNumFloat
{}


macro_rules! impl_from_primitive_for_scalar(
    ($t:ty) => {
        impl num_traits::FromPrimitive for Dual<$t> {
            fn from_i64(n: i64) -> Option<Self> {
                Some(Self::from_re(n as $t))
            }

            fn from_u64(n: u64) -> Option<Self> {
                Some(Self::from_re(n as $t))
            }

            fn from_f32(n: f32) -> Option<Self> {
                Some(Self::from_re(n as $t))
            }

            fn from_f64(n: f64) -> Option<Self> {
                Some(Self::from_re(n as $t))
            }
        }
    }
);

impl_from_primitive_for_scalar!(f32);
impl_from_primitive_for_scalar!(f64);

macro_rules! impl_complex_field_for_scalar(
    ($t:ty) => {
        impl nalgebra::ComplexField for Dual<$t> {
            type RealField = Self;

            #[inline]
            fn from_real(re: Self::RealField) -> Self {
                re
            }

            #[inline]
            fn real(self) -> Self::RealField {
                self
            }

            #[inline]
            fn imaginary(self) -> Self::RealField {
                Self::zero()
            }

            #[inline]
            fn norm1(self) -> Self::RealField {
                self.abs()
            }

            #[inline]
            fn modulus(self) -> Self::RealField {
                self.abs()
            }

            #[inline]
            fn modulus_squared(self) -> Self::RealField {
                self * self
            }

            #[inline]
            fn argument(self) -> Self::RealField {
                if self >= Self::zero() {
                    Self::zero()
                } else {
                    Self::pi()
                }
            }

            #[inline]
            fn to_exp(self) -> (Self, Self) {
                todo!()
            }

            #[inline]
            fn recip(self) -> Self {
                todo!()
            }

            #[inline]
            fn conjugate(self) -> Self {
                self
            }

            #[inline]
            fn scale(self, factor: Self::RealField) -> Self {
                self * factor
            }

            #[inline]
            fn unscale(self, factor: Self::RealField) -> Self {
                self / factor
            }

            #[inline]
            fn floor(self) -> Self {
                panic!("called floor() on a dual number")
            }

            #[inline]
            fn ceil(self) -> Self {
                panic!("called ceil() on a dual number")
            }

            #[inline]
            fn round(self) -> Self {
                panic!("called round() on a dual number")
            }

            #[inline]
            fn trunc(self) -> Self {
                panic!("called trunc() on a dual number")
            }

            #[inline]
            fn fract(self) -> Self {
                panic!("called fract() on a dual number")
            }

            #[inline]
            fn abs(self) -> Self::RealField {
                Signed::abs(&self)
            }

            #[inline]
            fn signum(self) -> Self {
                todo!()
            }

            #[inline]
            fn mul_add(self, a: Self, b: Self) -> Self {
                todo!()
            }

            #[inline]
            fn powi(self, n: i32) -> Self {
                todo!()
            }
        }
    }
);

impl_complex_field_for_scalar!(f32);
impl_complex_field_for_scalar!(f64);




/*
impl<T> RealField for Dual<T>
where
    T: DualNumFloat
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
        DualNum::atan2(&self, other)
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

*/