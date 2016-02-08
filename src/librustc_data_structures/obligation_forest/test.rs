// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::iter::FromIterator;

use super::{ObligationForest, Outcome, Error};

#[test]
fn push_pop_ref_cell() {
    let forest = RefCell::new(ObligationForest::new());
    forest.borrow_mut().push_root("A");
    forest.borrow_mut().push_root("B");
    forest.borrow_mut().push_root("C");

    // first round, B errors out, A has subtasks, and C completes, creating this:
    //      A |-> A.1
    //        |-> A.2
    //        |-> A.3
    let Outcome { completed: ok, errors: err, .. } =
        ObligationForest::process_obligations_external(&forest, |obligation, _| {
            match *obligation {
                "A" => Ok(Some(vec!["A.1", "A.2", "A.3"])),
                "B" => Err("B is for broken"),
                "C" => Ok(Some(vec![])),
                _ => panic!("unexpected obligation {:?}", obligation),
            }
        });
    assert_eq!(ok, vec!["C"]);
    assert_eq!(err, vec![Error {error: "B is for broken",
                                backtrace: vec!["B"]}]);

    // second round: two delays, one success, creating an uneven set of subtasks:
    //      A |-> A.1
    //        |-> A.2
    //        |-> A.3 |-> A.3.i
    //      D |-> D.1
    //        |-> D.2
    forest.borrow_mut().push_root("D");
    let Outcome { completed: ok, errors: err, .. }: Outcome<&'static str, ()> =
        ObligationForest::process_obligations_external(&forest, |obligation, _| {
            match *obligation {
                "A.1" => Ok(None),
                "A.2" => Ok(None),
                "A.3" => Ok(Some(vec!["A.3.i"])),
                "D" => Ok(Some(vec!["D.1", "D.2"])),
                _ => panic!("unexpected obligation {:?}", obligation),
            }
        });
    assert_eq!(ok, Vec::<&'static str>::new());
    assert_eq!(err, Vec::new());


    // third round: ok in A.1 but trigger an error in A.2. Check that it
    // propagates to A.3.i, but not D.1 or D.2.
    //      D |-> D.1 |-> D.1.i
    //        |-> D.2 |-> D.2.i
    let Outcome { completed: ok, errors: err, .. } =
        ObligationForest::process_obligations_external(&forest, |obligation, _| {
            match *obligation {
                "A.1" => Ok(Some(vec![])),
                "A.2" => Err("A is for apple"),
                "D.1" => Ok(Some(vec!["D.1.i"])),
                "D.2" => Ok(Some(vec!["D.2.i"])),
                _ => panic!("unexpected obligation {:?}", obligation),
            }
        });
    assert_eq!(ok, vec!["A.1"]);
    assert_eq!(err, vec![Error { error: "A is for apple",
                                 backtrace: vec!["A.2", "A"] }]);

    // fourth round: error in D.1.i that should propagate to D.2.i
    let Outcome { completed: ok, errors: err, .. } =
        ObligationForest::process_obligations_external(&forest, |obligation, _| {
            match *obligation {
                "D.1.i" => Err("D is for dumb"),
                _ => panic!("unexpected obligation {:?}", obligation),
            }
        });
    assert_eq!(ok, Vec::<&'static str>::new());
    assert_eq!(err, vec![Error { error: "D is for dumb",
                                 backtrace: vec!["D.1.i", "D.1", "D"] }]);
}


#[test]
fn push_pop() {
    let mut forest = ObligationForest::new();
    forest.push_root("A");
    forest.push_root("B");
    forest.push_root("C");

    // first round, B errors out, A has subtasks, and C completes, creating this:
    //      A |-> A.1
    //        |-> A.2
    //        |-> A.3
    let Outcome { completed: ok, errors: err, .. } = forest.process_obligations(|obligation, _| {
        match *obligation {
            "A" => Ok(Some(vec!["A.1", "A.2", "A.3"])),
            "B" => Err("B is for broken"),
            "C" => Ok(Some(vec![])),
            _ => unreachable!(),
        }
    });
    assert_eq!(ok, vec!["C"]);
    assert_eq!(err, vec![Error {error: "B is for broken",
                                backtrace: vec!["B"]}]);

    // second round: two delays, one success, creating an uneven set of subtasks:
    //      A |-> A.1
    //        |-> A.2
    //        |-> A.3 |-> A.3.i
    //      D |-> D.1
    //        |-> D.2
    forest.push_root("D");
    let Outcome { completed: ok, errors: err, .. }: Outcome<&'static str, ()> =
        forest.process_obligations(|obligation, _| {
            match *obligation {
                "A.1" => Ok(None),
                "A.2" => Ok(None),
                "A.3" => Ok(Some(vec!["A.3.i"])),
                "D" => Ok(Some(vec!["D.1", "D.2"])),
                _ => unreachable!(),
            }
        });
    assert_eq!(ok, Vec::<&'static str>::new());
    assert_eq!(err, Vec::new());


    // third round: ok in A.1 but trigger an error in A.2. Check that it
    // propagates to A.3.i, but not D.1 or D.2.
    //      D |-> D.1 |-> D.1.i
    //        |-> D.2 |-> D.2.i
    let Outcome { completed: ok, errors: err, .. } = forest.process_obligations(|obligation, _| {
        match *obligation {
            "A.1" => Ok(Some(vec![])),
            "A.2" => Err("A is for apple"),
            "D.1" => Ok(Some(vec!["D.1.i"])),
            "D.2" => Ok(Some(vec!["D.2.i"])),
            _ => unreachable!(),
        }
    });
    assert_eq!(ok, vec!["A.1"]);
    assert_eq!(err, vec![Error { error: "A is for apple",
                                 backtrace: vec!["A.2", "A"] }]);

    // fourth round: error in D.1.i that should propagate to D.2.i
    let Outcome { completed: ok, errors: err, .. } = forest.process_obligations(|obligation, _| {
        match *obligation {
            "D.1.i" => Err("D is for dumb"),
            _ => panic!("unexpected obligation {:?}", obligation),
        }
    });
    assert_eq!(ok, Vec::<&'static str>::new());
    assert_eq!(err, vec![Error { error: "D is for dumb",
                                 backtrace: vec!["D.1.i", "D.1", "D"] }]);
}

// Test that if a tree with grandchildren succeeds, everything is
// reported as expected:
// A
//   A.1
//   A.2
//      A.2.i
//      A.2.ii
//   A.3
#[test]
fn success_in_grandchildren() {
    let mut forest = ObligationForest::new();
    forest.push_root("A");

    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations::<(),_>(|obligation, _| {
            match *obligation {
                "A" => Ok(Some(vec!["A.1", "A.2", "A.3"])),
                _ => unreachable!(),
            }
        });
    assert!(ok.is_empty());
    assert!(err.is_empty());

    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations::<(),_>(|obligation, _| {
            match *obligation {
                "A.1" => Ok(Some(vec![])),
                "A.2" => Ok(Some(vec!["A.2.i", "A.2.ii"])),
                "A.3" => Ok(Some(vec![])),
                _ => unreachable!(),
            }
        });
    assert_eq!(BTreeSet::from_iter(ok), BTreeSet::from_iter(vec!["A.3", "A.1"]));
    assert!(err.is_empty());

    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations::<(),_>(|obligation, _| {
            match *obligation {
                "A.2.i" => Ok(Some(vec!["A.2.i.a"])),
                "A.2.ii" => Ok(Some(vec![])),
                _ => unreachable!(),
            }
        });
    assert_eq!(ok, vec!["A.2.ii"]);
    assert!(err.is_empty());

    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations::<(),_>(|obligation, _| {
            match *obligation {
                "A.2.i.a" => Ok(Some(vec![])),
                _ => unreachable!(),
            }
        });
    assert_eq!(ok, vec!["A.2.i.a", "A.2.i", "A.2", "A"]);
    assert!(err.is_empty());

    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations::<(),_>(|_, _| unreachable!());
    assert!(ok.is_empty());
    assert!(err.is_empty());
}

#[test]
fn to_errors_no_throw() {
    // check that converting multiple children with common parent (A)
    // only yields one of them (and does not panic, in particular).
    let mut forest = ObligationForest::new();
    forest.push_root("A");
    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations::<(),_>(|obligation, _| {
            match *obligation {
                "A" => Ok(Some(vec!["A.1", "A.2", "A.3"])),
                _ => unreachable!(),
            }
        });
    assert_eq!(ok.len(), 0);
    assert_eq!(err.len(), 0);
    let errors = forest.to_errors(());
    assert_eq!(errors.len(), 1);
}

#[test]
fn backtrace() {
    // check that converting multiple children with common parent (A)
    // only yields one of them (and does not panic, in particular).
    let mut forest: ObligationForest<&'static str> = ObligationForest::new();
    forest.push_root("A");
    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations::<(),_>(|obligation, mut backtrace| {
            assert!(backtrace.next().is_none());
            match *obligation {
                "A" => Ok(Some(vec!["A.1"])),
                _ => unreachable!(),
            }
        });
    assert!(ok.is_empty());
    assert!(err.is_empty());
    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations::<(),_>(|obligation, mut backtrace| {
            assert!(backtrace.next().unwrap() == &"A");
            assert!(backtrace.next().is_none());
            match *obligation {
                "A.1" => Ok(Some(vec!["A.1.i"])),
                _ => unreachable!(),
            }
        });
    assert!(ok.is_empty());
    assert!(err.is_empty());
    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations::<(),_>(|obligation, mut backtrace| {
            assert!(backtrace.next().unwrap() == &"A.1");
            assert!(backtrace.next().unwrap() == &"A");
            assert!(backtrace.next().is_none());
            match *obligation {
                "A.1.i" => Ok(None),
                _ => unreachable!(),
            }
        });
    assert_eq!(ok.len(), 0);
    assert!(err.is_empty());
}

#[test]
fn snapshots_0() {
    let mut forest: ObligationForest<&'static str> = ObligationForest::new();
    forest.push_root("A");
    let snap = forest.start_snapshot();
    let Outcome { completed: ok_snap, errors: err_snap, .. } =
        forest.process_obligations::<(),_>(|obligation, _| {
            match *obligation {
                "A" => Ok(Some(vec![])),
                _ => unreachable!()
            }
        });
    assert_eq!(ok_snap, vec!["A"]);
    assert_eq!(err_snap, vec![]);

    forest.rollback_snapshot(snap);
    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations::<(),_>(|obligation, _| {
            match *obligation {
                "A" => Ok(Some(vec![])),
                _ => unreachable!()
            }
        });
    assert_eq!(ok, vec!["A"]);
    assert_eq!(err, vec![]);
}

// Like push_pop, but with snapshots taken in various places without rollbacks until the very end.
#[test]
fn snapshots_1() {
    let mut forest = ObligationForest::new();
    let mut snapshots = Vec::new();
    forest.push_root("A");
    forest.push_root("B");
    snapshots.push(forest.start_snapshot());
    forest.push_root("C");

    // first round, B errors out, A has subtasks, and C completes, creating this:
    //      A |-> A.1
    //        |-> A.2
    //        |-> A.3
    let Outcome { completed: ok, errors: err, .. } = forest.process_obligations(|obligation, _| {
        match *obligation {
            "A" => Ok(Some(vec!["A.1", "A.2", "A.3"])),
            "B" => Err("B is for broken"),
            "C" => Ok(Some(vec![])),
            _ => unreachable!(),
        }
    });
    assert_eq!(ok, vec!["C"]);
    assert_eq!(err, vec![Error {error: "B is for broken",
                                backtrace: vec!["B"]}]);
    snapshots.push(forest.start_snapshot());

    // second round: two delays, one success, creating an uneven set of subtasks:
    //      A |-> A.1
    //        |-> A.2
    //        |-> A.3 |-> A.3.i
    //      D |-> D.1
    //        |-> D.2
    forest.push_root("D");
    snapshots.push(forest.start_snapshot());
    let Outcome { completed: ok, errors: err, .. }: Outcome<&'static str, ()> =
        forest.process_obligations(|obligation, _| {
            match *obligation {
                "A.1" => Ok(None),
                "A.2" => Ok(None),
                "A.3" => Ok(Some(vec!["A.3.i"])),
                "D" => Ok(Some(vec!["D.1", "D.2"])),
                _ => unreachable!(),
            }
        });
    assert_eq!(ok, Vec::<&'static str>::new());
    assert_eq!(err, Vec::new());


    // third round: ok in A.1 but trigger an error in A.2. Check that it
    // propagates to A.3.i, but not D.1 or D.2.
    //      D |-> D.1 |-> D.1.i
    //        |-> D.2 |-> D.2.i
    let Outcome { completed: ok, errors: err, .. } = forest.process_obligations(|obligation, _| {
        match *obligation {
            "A.1" => Ok(Some(vec![])),
            "A.2" => Err("A is for apple"),
            "D.1" => Ok(Some(vec!["D.1.i"])),
            "D.2" => Ok(Some(vec!["D.2.i"])),
            _ => unreachable!(),
        }
    });
    assert_eq!(ok, vec!["A.1"]);
    assert_eq!(err, vec![Error { error: "A is for apple",
                                 backtrace: vec!["A.2", "A"] }]);
    snapshots.push(forest.start_snapshot());

    // fourth round: error in D.1.i that should propagate to D.2.i
    let Outcome { completed: ok, errors: err, .. } = forest.process_obligations(|obligation, _| {
        match *obligation {
            "D.1.i" => Err("D is for dumb"),
            _ => panic!("unexpected obligation {:?}", obligation),
        }
    });
    assert_eq!(ok, Vec::<&'static str>::new());
    assert_eq!(err, vec![Error { error: "D is for dumb",
                                 backtrace: vec!["D.1.i", "D.1", "D"] }]);
    while let Some(snapshot) = snapshots.pop() {
        forest.rollback_snapshot(snapshot);
    }
    // All we should be left with are the two roots "A" and "B" as Pending.
    let Outcome { completed: ok, errors: err, .. } = forest.process_obligations(|obligation, _| {
        match *obligation {
            "A" => Ok(Some(vec![])),
            "B" => Err("B is for broken"),
            _ => unreachable!(),
        }
    });
    assert_eq!(ok, vec!["A"]);
    assert_eq!(err, vec![Error { error: "B is for broken",
                                 backtrace: vec!["B"] }]);
}

// Like push_pop, but with snapshots, rollbacks, and commits interspersed throughout.
#[test]
fn snapshots_2() {
    let mut forest: ObligationForest<&'static str> = ObligationForest::new();
    let mut snapshots = Vec::new();
    // Snapshots: |
    forest.push_root("A");
    forest.push_root("B");
    // Snapshots: | -> |
    snapshots.push(forest.start_snapshot());
    forest.push_root("C");
    // Snapshots: |
    forest.rollback_snapshot(snapshots.pop().unwrap());

    // Snapshots: | -> |
    snapshots.push(forest.start_snapshot());

    // first round, B errors out, A has subtasks, and C completes, creating this:
    //      A |-> A.1
    //        |-> A.2
    //        |-> A.3
    let Outcome { completed: ok, errors: err, .. } = forest.process_obligations(|obligation, _| {
        match *obligation {
            "A" => Ok(Some(vec!["A.1", "A.2", "A.3"])),
            "B" => Err("B is for broken"),
            "C" => panic!("C shouldn't exist after being rolled back"),
            _ => unreachable!(),
        }
    });
    assert!(ok.is_empty());
    assert_eq!(err, vec![Error {error: "B is for broken",
                                backtrace: vec!["B"]}]);

    forest.push_root("C");

    // Snapshots: | -> | -> |
    snapshots.push(forest.start_snapshot());

    // second round: two delays, one success, creating an uneven set of subtasks:
    //      A |-> A.1
    //        |-> A.2
    //        |-> A.3 |-> A.3.i
    //      D |-> D.1
    //        |-> D.2
    forest.push_root("D");
    // Snapshots: | -> | -> | -> |
    snapshots.push(forest.start_snapshot());
    let Outcome { completed: ok, errors: err, .. }: Outcome<&'static str, ()> =
        forest.process_obligations(|obligation, _| {
            match *obligation {
                "A.1" => Ok(None),
                "A.2" => Ok(None),
                "A.3" => Ok(Some(vec!["A.3.i"])),
                "C" => Ok(Some(vec![])),
                "D" => Ok(Some(vec!["D.1", "D.2"])),
                _ => unreachable!(),
            }
        });
    assert_eq!(ok, vec!["C"]);
    assert_eq!(err, Vec::new());

    // Undo the entire previous action with uneven subtasks, then do it again
    // Snapshots: | -> | -> |
    forest.rollback_snapshot(snapshots.pop().unwrap());

    // Snapshots: | -> | -> | -> |
    snapshots.push(forest.start_snapshot());
    let Outcome { completed: ok, errors: err, .. }: Outcome<&'static str, ()> =
        forest.process_obligations(|obligation, _| {
            match *obligation {
                "A.1" => Ok(None),
                "A.2" => Ok(None),
                "A.3" => Ok(Some(vec!["A.3.i"])),
                "C" => Ok(Some(vec![])),
                "D" => Ok(Some(vec!["D.1", "D.2"])),
                _ => unreachable!(),
            }
        });
    assert_eq!(ok, vec!["C"]);
    assert_eq!(err, Vec::new());

    // Snapshots: | -> | -> |
    forest.commit_snapshot(snapshots.pop().unwrap());

    // third round: ok in A.1 but trigger an error in A.2. Check that it
    // propagates to A.3.i, but not D.1 or D.2.
    //      D |-> D.1 |-> D.1.i
    //        |-> D.2 |-> D.2.i
    let Outcome { completed: ok, errors: err, .. } = forest.process_obligations(|obligation, _| {
        match *obligation {
            "A.1" => Ok(Some(vec![])),
            "A.2" => Err("A is for apple"),
            "D.1" => Ok(Some(vec!["D.1.i"])),
            "D.2" => Ok(Some(vec!["D.2.i"])),
            _ => unreachable!(),
        }
    });
    assert_eq!(ok, vec!["A.1"]);
    assert_eq!(err, vec![Error { error: "A is for apple",
                                 backtrace: vec!["A.2", "A"] }]);

    // fourth round: error in D.1.i that should propagate to D.2.i
    let Outcome { completed: ok, errors: err, .. } = forest.process_obligations(|obligation, _| {
        match *obligation {
            "D.1.i" => Err("D is for dumb"),
            _ => panic!("unexpected obligation {:?}", obligation),
        }
    });
    assert_eq!(ok, Vec::<&'static str>::new());
    assert_eq!(err, vec![Error { error: "D is for dumb",
                                 backtrace: vec!["D.1.i", "D.1", "D"] }]);

    forest.rollback_snapshot(snapshots.pop().unwrap());
    forest.rollback_snapshot(snapshots.pop().unwrap());

    // All we should be left with are the two roots "A" and "B" as Pending.
    let Outcome { completed: ok, errors: err, .. } = forest.process_obligations(|obligation, _| {
        match *obligation {
            "A" => Ok(Some(vec![])),
            "B" => Err("B is for broken"),
            _ => panic!("unexpected obligation {:?}", obligation),
        }
    });
    assert_eq!(ok, vec!["A"]);
    assert_eq!(err, vec![Error { error: "B is for broken",
                                 backtrace: vec!["B"] }]);
}

// Like to_errors_no_throw, but with snapshots taken and rolled back.
#[test]
fn snapshots_3() {
    // check that converting multiple children with common parent (A)
    // only yields one of them (and does not panic, in particular).
    let mut forest = ObligationForest::new();
    let mut snapshots = Vec::new();
    forest.push_root("A");
    snapshots.push(forest.start_snapshot());
    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations::<(),_>(|obligation, _| {
            match *obligation {
                "A" => Ok(Some(vec!["A.1", "A.2", "A.3"])),
                _ => unreachable!(),
            }
        });
    assert_eq!(ok.len(), 0);
    assert_eq!(err.len(), 0);
    let errors = forest.to_errors(());
    assert_eq!(errors.len(), 1);
    forest.rollback_snapshot(snapshots.pop().unwrap());
    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations::<(),_>(|obligation, _| {
            match *obligation {
                "A" => Ok(Some(vec!["A.1", "A.2", "A.3"])),
                _ => unreachable!(),
            }
        });
    assert_eq!(ok.len(), 0);
    assert_eq!(err.len(), 0);
    let errors = forest.to_errors(());
    assert_eq!(errors.len(), 1);
}

// Like backtrace, but with snapshots taken and rolled back.
#[test]
fn snapshots_4() {
    // check that converting multiple children with common parent (A)
    // only yields one of them (and does not panic, in particular).
    let mut forest: ObligationForest<&'static str> = ObligationForest::new();
    let mut snapshots = Vec::new();
    forest.push_root("A");
    snapshots.push(forest.start_snapshot());
    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations::<(),_>(|obligation, mut backtrace| {
            assert!(backtrace.next().is_none());
            match *obligation {
                "A" => Ok(Some(vec!["A.1"])),
                _ => unreachable!(),
            }
        });
    assert!(ok.is_empty());
    assert!(err.is_empty());
    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations::<(),_>(|obligation, mut backtrace| {
            assert!(backtrace.next().unwrap() == &"A");
            assert!(backtrace.next().is_none());
            match *obligation {
                "A.1" => Ok(Some(vec!["A.1.i"])),
                _ => unreachable!(),
            }
        });
    assert!(ok.is_empty());
    assert!(err.is_empty());
    snapshots.push(forest.start_snapshot());
    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations::<(),_>(|obligation, mut backtrace| {
            assert!(backtrace.next().unwrap() == &"A.1");
            assert!(backtrace.next().unwrap() == &"A");
            assert!(backtrace.next().is_none());
            match *obligation {
                "A.1.i" => Ok(None),
                _ => unreachable!(),
            }
        });
    assert_eq!(ok.len(), 0);
    assert!(err.is_empty());
    forest.commit_snapshot(snapshots.pop().unwrap());
    forest.rollback_snapshot(snapshots.pop().unwrap());
    // We should be back to just having "A" as the only pending root.
    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations::<(),_>(|obligation, mut backtrace| {
            assert!(backtrace.next().is_none());
            match *obligation {
                "A" => Ok(Some(vec!["A.1"])),
                _ => unreachable!(),
            }
        });
    assert!(ok.is_empty());
    assert!(err.is_empty());
    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations::<(),_>(|obligation, mut backtrace| {
            assert!(backtrace.next().unwrap() == &"A");
            assert!(backtrace.next().is_none());
            match *obligation {
                "A.1" => Ok(Some(vec!["A.1.i"])),
                _ => unreachable!(),
            }
        });
    assert!(ok.is_empty());
    assert!(err.is_empty());
    let Outcome { completed: ok, errors: err, .. } =
        forest.process_obligations::<(),_>(|obligation, mut backtrace| {
            assert!(backtrace.next().unwrap() == &"A.1");
            assert!(backtrace.next().unwrap() == &"A");
            assert!(backtrace.next().is_none());
            match *obligation {
                "A.1.i" => Ok(None),
                _ => unreachable!(),
            }
        });
    assert_eq!(ok.len(), 0);
    assert!(err.is_empty());
}

