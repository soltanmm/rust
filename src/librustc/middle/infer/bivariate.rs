// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Applies the "bivariance relationship" to two types and/or regions.
//! If (A,B) are bivariant then either A <: B or B <: A. It occurs
//! when type/lifetime parameters are unconstrained. Usually this is
//! an error, but we permit it in the specific case where a type
//! parameter is constrained in a where-clause via an associated type.
//!
//! There are several ways one could implement bivariance. You could
//! just do nothing at all, for example, or you could fully verify
//! that one of the two subtyping relationships hold. We choose to
//! thread a middle line: we relate types up to regions, but ignore
//! all region relationships.
//!
//! At one point, handling bivariance in this fashion was necessary
//! for inference, but I'm actually not sure if that is true anymore.
//! In particular, it might be enough to say (A,B) are bivariant for
//! all (A,B).

use super::combine::{self, CombineFields};
use super::type_variable::{BiTo};

use middle::traits::{Normalized, PredicateObligation};
use middle::ty::{self, Ty};
use middle::ty::TyVar;
use middle::ty::relate::{Relate, RelateResult, TypeRelation};

pub struct Bivariate<'a, 'o, 'tcx: 'a + 'o> {
    fields: CombineFields<'a, 'o, 'tcx>
}

impl<'a, 'o, 'tcx> Bivariate<'a, 'o, 'tcx> {
    pub fn new(fields: CombineFields<'a, 'o, 'tcx>) -> Bivariate<'a, 'o, 'tcx> {
        Bivariate { fields: fields }
    }
}

impl<'a, 'o, 'tcx> TypeRelation<'a, 'tcx> for Bivariate<'a, 'o, 'tcx> {
    fn tag(&self) -> &'static str { "Bivariate" }

    fn tcx(&self) -> &'a ty::ctxt<'tcx> { self.fields.tcx() }

    fn obligations(&self) -> &Vec<PredicateObligation<'tcx>> { self.fields.obligations }

    fn a_is_expected(&self) -> bool { self.fields.a_is_expected }

    fn relate_with_variance<T:Relate<'a,'tcx>>(&mut self,
                                               variance: ty::Variance,
                                               a: &T,
                                               b: &T)
                                               -> RelateResult<'tcx, T>
    {
        match variance {
            // If we have Foo<A> and Foo is invariant w/r/t A,
            // and we want to assert that
            //
            //     Foo<A> <: Foo<B> ||
            //     Foo<B> <: Foo<A>
            //
            // then still A must equal B.
            ty::Invariant => self.relate(a, b),

            ty::Covariant => self.relate(a, b),
            ty::Bivariant => self.relate(a, b),
            ty::Contravariant => self.relate(a, b),
        }
    }

    fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        debug!("{}.tys({:?}, {:?})", self.tag(),
               a, b);
        if a == b { return Ok(a); }

        let infcx = self.fields.infcx;
        let a = infcx.type_variables.borrow().replace_if_possible(a);
        let b = infcx.type_variables.borrow().replace_if_possible(b);
        let Normalized { value: a, obligations: a_norm_obligations } =
            infcx.normalize_if_possible(a);
        self.fields.obligations.extend(a_norm_obligations);
        let Normalized { value: b, obligations: b_norm_obligations } =
            infcx.normalize_if_possible(b);
        self.fields.obligations.extend(b_norm_obligations);
        match (&a.sty, &b.sty) {
            (&ty::TyInfer(TyVar(a_id)), &ty::TyInfer(TyVar(b_id))) => {
                infcx.type_variables.borrow_mut().relate_vars(a_id, BiTo, b_id);
                Ok(a)
            }

            (&ty::TyInfer(TyVar(a_id)), _) => {
                try!(self.fields.instantiate(b, BiTo, a_id));
                Ok(a)
            }

            (_, &ty::TyInfer(TyVar(b_id))) => {
                try!(self.fields.instantiate(a, BiTo, b_id));
                Ok(a)
            }

            _ => {
                combine::super_combine_tys(self.fields.infcx, self, a, b)
            }
        }
    }

    fn regions(&mut self, a: ty::Region, _: ty::Region) -> RelateResult<'tcx, ty::Region> {
        Ok(a)
    }

    fn binders<T>(&mut self, a: &ty::Binder<T>, b: &ty::Binder<T>)
                  -> RelateResult<'tcx, ty::Binder<T>>
        where T: Relate<'a,'tcx>
    {
        let a1 = self.tcx().erase_late_bound_regions(a);
        let b1 = self.tcx().erase_late_bound_regions(b);
        let c = try!(self.relate(&a1, &b1));
        Ok(ty::Binder(c))
    }
}
