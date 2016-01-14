// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::nonzero::NonZero;
use std::ops::{Add, Sub};
use std::u32;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct NodeIndex {
    index: NonZero<u32>
}

impl NodeIndex {
    pub fn new(value: usize) -> NodeIndex {
        assert!(value < (u32::MAX as usize));
        unsafe {
            NodeIndex { index: NonZero::new((value as u32) + 1) }
        }
    }

    pub fn get(self) -> usize {
        (*self.index - 1) as usize
    }
}

impl Add for NodeIndex {
    type Output = NodeIndex;
    fn add(self, rhs: NodeIndex) -> NodeIndex {
        let underlying = self.get() + rhs.get();
        assert!(underlying > self.get());
        assert!(underlying > rhs.get());
        NodeIndex::new(underlying)
    }
}
impl Sub for NodeIndex {
    type Output = NodeIndex;
    fn sub(self, rhs: NodeIndex) -> NodeIndex {
        assert!(self.get() >= rhs.get());
        NodeIndex::new(self.get() - rhs.get())
    }
}

