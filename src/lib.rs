// #![cfg_attr(
//     all(not(feature = "std"), not(test)),
//     no_std
// )]

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(all(feature = "alloc", not(feature = "std")))]
#[cfg_attr(test, macro_use)]
extern crate alloc;

pub mod prelude {

    pub(crate) use core::{
        any::Any,
        fmt,
        ops::{
            Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
        },
    };

    #[cfg(any(feature = "std", feature = "libm"))]
    pub(crate) use num_traits::Float;

    #[cfg(all(not(feature = "std"), not(feature = "libm")))]
    pub(crate) use num_traits::float::FloatCore as Float;

    pub(crate) use num_traits::{FloatConst, FromPrimitive, Num, NumOps, One, Signed, Zero};

    pub(crate) use nalgebra::{
        ComplexField, Field, RealField, SMatrix, SVector, SimdValue, VectorView, SimdPartialOrd
    };
}

mod forward_autodiff;
pub use forward_autodiff::*;

use prelude::*;

pub trait DualNumFloat: Float + FloatConst + SimdValue<Element = Self, SimdBool = bool> + SimdPartialOrd + FromPrimitive + Clone + Copy + Send + Sync + fmt::Debug + fmt::Display + 'static {}

impl DualNumFloat for f32 {}
impl DualNumFloat for f64 {}

pub trait DualNum<T>
where
    Self: Field + FromPrimitive + From<T> + NumOps<T> + Clone + Copy + Send + Sync + Any + fmt::Debug + fmt::Display + 'static,
    T: DualNumFloat,
{

    #[cfg(any(feature = "std", feature = "libm"))]
    fn sin(&self) -> Self;

}

#[cfg(any(feature = "std", feature = "libm"))]
pub trait RealDualNum<T>: DualNum<T> + RealField
where T: DualNumFloat {}


pub fn jacobian<G, F, const N: usize, const M: usize>(
    g: G,
    x: VectorView<'_, Dual<F>, nalgebra::Const<N>>,
) -> SMatrix<F, N, M>
where
    G: Fn(VectorView<'_, Dual<F>, nalgebra::Const<N>>) -> SVector<Dual<F>, { M }>,
    F: DualNumFloat,
{
    let mut jac = SMatrix::<F, N, M>::zeros();
    let mut x_p = SVector::<Dual<F>, N>::zeros();

    for i in 0..N {
        for j in 0..N {
            if i == j {
                x_p[j] = x[j].derivative();
            } else {
                x_p[j] = Dual::from_re(x[j].re);
            }
        }

        let y = g(x_p.as_view());

        for k in 0..M {
            jac[(i, k)] = y[k].eps;
        }
    }

    jac
}

#[cfg(test)]
mod tests {

    use super::*;
    #[test]
    fn test_derivative() {
        let f = |x: Dual32, y: Dual32| -> Dual32 { x * x + y };

        let x = f(Dual32::from_re(3.0).derivative(), Dual32::from_re(3.0));

        assert_eq!(x.re, 12.0);
        assert_eq!(x.eps, 6.0);
    }

    #[test]
    fn test_jacobian() {
        use nalgebra::{OVector, SVector, U1, U2, VectorView};

        fn my_fn(x: VectorView<'_, Dual32, U2>) -> OVector<Dual32, U1> {
            let y = x[0] * x[0] + Dual32::from_re(2.0) * x[1];
            OVector::<Dual32, U1>::from_row_slice(&[y])
        }

        let x = SVector::<Dual32, 2>::from_row_slice(&[Dual32::from_re(3.0), Dual32::from_re(5.0)]);
        // let x_view: nalgebra::VectorView<'_, f32, nalgebra::U2> = x.as_view();

        let jac = jacobian(my_fn, x.as_view());
        assert_eq!(jac[(0, 0)], 6.0);
        assert_eq!(jac[(1, 0)], 2.0);
    }
}
