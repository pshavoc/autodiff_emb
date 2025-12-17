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

