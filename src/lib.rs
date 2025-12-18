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
use nalgebra::SVector;

pub fn jacobian<G, T, const N: usize, const M: usize>(
    g: G,
    x: nalgebra::VectorView<'_, Dual<T>, nalgebra::Const<N>>,
) -> nalgebra::SMatrix<T, N, M>
where
    G: Fn(
        nalgebra::VectorView<'_, Dual<T>, nalgebra::Const<N>>,
    ) -> nalgebra::SVector<Dual<T>, { M }>,
    T: DualNumFloat,
{
    let mut jac = nalgebra::SMatrix::<T, N, M>::zeros();
    let mut x_p = SVector::<Dual<T>, N>::zeros();

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

        use nalgebra::{SVector, OVector, VectorView, U1, U2};

        fn my_fn(x: VectorView<'_, Dual32, U2>) -> OVector<Dual32, U1> {
            OVector::<Dual32, U1>::from_row_slice(&[x[0] * x[0] + x[1]])
        }

        let x = SVector::<Dual32, 2>::from_row_slice(&[Dual32::from_re(3.0), Dual32::from_re(5.0)]);
        // let x_view: nalgebra::VectorView<'_, f32, nalgebra::U2> = x.as_view();
        

        let jac = jacobian(my_fn, x.as_view());
        assert_eq!(jac[(0, 0)], 6.0);

    }
}
