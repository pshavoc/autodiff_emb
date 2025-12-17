// #![cfg_attr(
//     all(not(feature = "std"), not(test)),
//     no_std
// )]

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(all(feature = "alloc", not(feature = "std")))]
#[cfg_attr(test, macro_use)]
extern crate alloc;

mod forward_autodiff;
pub use forward_autodiff::*;

pub fn jacobian<'a, G, T, F, const N: usize, const M: usize, O>(
    g: G,
    x: nalgebra::VectorView<'a, Dual<T>, nalgebra::Const<N>>,
) -> nalgebra::SMatrix<T, N, N>
where
    G: FnOnce(
        nalgebra::VectorView<'a, Dual<T>, nalgebra::Const<N>>,
    ) -> nalgebra::OVector<Dual<T>, nalgebra::Const<M>>,
    T: DualNumFloat,
{
    todo!()
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
}
