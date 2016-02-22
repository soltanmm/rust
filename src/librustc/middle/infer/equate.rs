// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::combine::{self, CombineFields};
use super::higher_ranked::HigherRankedRelations;
use super::{Subtype};
use super::type_variable::{EqTo};

use middle::traits::Normalized;
use middle::ty::{self, Ty};
use middle::ty::TyVar;
use middle::ty::relate::{Relate, RelateOk, RelateResult, RelateResultTrait, TypeRelation};

/// Ensures `a` is made equal to `b`. Returns `a` on success.
pub struct Equate<'a, 'tcx: 'a> {
    fields: CombineFields<'a, 'tcx>
}

impl<'a, 'tcx> Equate<'a, 'tcx> {
    pub fn new(fields: CombineFields<'a, 'tcx>) -> Equate<'a, 'tcx> {
        Equate { fields: fields }
    }
}

impl<'a, 'tcx> TypeRelation<'a,'tcx> for Equate<'a, 'tcx> {
    fn tag(&self) -> &'static str { "Equate" }

    fn tcx(&self) -> &'a ty::ctxt<'tcx> { self.fields.tcx() }

    fn a_is_expected(&self) -> bool { self.fields.a_is_expected }

    fn relate_with_variance<T:Relate<'a,'tcx>>(&mut self,
                                               _: ty::Variance,
                                               a: &T,
                                               b: &T)
                                               -> RelateResult<'tcx, T>
    {
        self.relate(a, b)
    }

    fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        debug!("{}.tys({:?}, {:?})", self.tag(),
               a, b);
        if a == b { return Ok(RelateOk::from(a)); }

        let infcx = self.fields.infcx;
        let a = infcx.type_variables.borrow().replace_if_possible(a);
        let b = infcx.type_variables.borrow().replace_if_possible(b);
        // Normalize the types
        let Normalized { value: a, obligations: a_norm_obligations } =
            infcx.normalize_if_possible(a);
        let Normalized { value: b, obligations: b_norm_obligations } =
            infcx.normalize_if_possible(b);
        match (&a.sty, &b.sty) {
            (&ty::TyInfer(TyVar(a_id)), &ty::TyInfer(TyVar(b_id))) => {
                infcx.type_variables.borrow_mut().relate_vars(a_id, EqTo, b_id);
                Ok(RelateOk::from(a))
                    .with_obligations(a_norm_obligations)
                    .with_obligations(b_norm_obligations)
            }

            (&ty::TyInfer(TyVar(a_id)), _) => {
                self.fields.instantiate(b, EqTo, a_id).map_value(|_| a)
                    .with_obligations(a_norm_obligations)
                    .with_obligations(b_norm_obligations)
            }

            (_, &ty::TyInfer(TyVar(b_id))) => {
                self.fields.instantiate(a, EqTo, b_id).map_value(|_| a)
                    .with_obligations(a_norm_obligations)
                    .with_obligations(b_norm_obligations)
            }

            _ => {
                combine::super_combine_tys(self.fields.infcx, self, a, b).map_value(|_| a)
                    .with_obligations(a_norm_obligations)
                    .with_obligations(b_norm_obligations)
            }
        }
    }

    fn regions(&mut self, a: ty::Region, b: ty::Region) -> RelateResult<'tcx, ty::Region> {
        debug!("{}.regions({:?}, {:?})",
               self.tag(),
               a,
               b);
        let origin = Subtype(self.fields.trace.clone());
        self.fields.infcx.region_vars.make_eqregion(origin, a, b);
        Ok(RelateOk::from(a))
    }

    fn binders<T>(&mut self, a: &ty::Binder<T>, b: &ty::Binder<T>)
                  -> RelateResult<'tcx, ty::Binder<T>>
        where T: Relate<'a, 'tcx>
    {
        self.fields.higher_ranked_sub(a, b)
            .and_then_with(|_| self.fields.higher_ranked_sub(b, a))
    }
}
