// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The `ObligationForest` is a utility data structure used in trait
//! matching to track the set of outstanding obligations (those not
//! yet resolved to success or error). It also tracks the "backtrace"
//! of each pending obligation (why we are trying to figure this out
//! in the first place). See README.md for a general overview of how
//! to use this class.

use std::fmt::Debug;
use std::mem;

mod node_index;

#[cfg(test)]
mod test;

#[derive(Debug)]
pub struct ObligationForest<O> {
    /// The list of obligations. In between calls to
    /// `process_obligations`, this list only contains nodes in the
    /// `Pending` or `Success` state (with a non-zero number of
    /// incomplete children). During processing, some of those nodes
    /// may be changed to the error state, or we may find that they
    /// are completed (That is, `num_incomplete_children` drops to 0).
    /// At the end of processing, those nodes will be removed by a
    /// call to `compress`.
    ///
    /// At all times we maintain the invariant that every node appears
    /// at a higher index than its parent. This is needed by the
    /// backtrace iterator (which uses `split_at`).
    nodes: Vec<Node<O>>,
    snapshots: Vec<usize>,

    /// List of inversion actions that may be performed to undo mutations
    /// while in snapshots.
    undo_log: Vec<UndoAction<O>>,
}

/// A single inversion of an action on a node in an ObligationForest.
#[derive(Debug)]
enum UndoAction<O> {
    // FIXME potential optimization: store push undos as a count in the snapshots vec (because we
    // never remove when in a snapshot, and because snapshots are sequenced in a specific way
    // w.r.t. what's in the ObligationForest's undo list, it's possible to just keep track of push
    // counts instead of sequencing them in the undo list with node state modifications)
    UndoPush,
    UndoModify {
        at: NodeIndex,
        undoer: UndoModify<O>,
    },
}

/// A single inversion of a node modification action in an ObligationForest.
#[derive(Debug)]
enum UndoModify<O> {
    /// Undoes a transition of a node from pending to error.
    UndoPendingIntoError {
        obligation: O,
    },
    /// Undoes a transition of a node from successful to error.
    UndoSuccessIntoError {
        obligation: O,
        num_incomplete_children: usize,
    },
    /// Undoes a pending node's success; also handles the corresponding num_incomplete_children
    /// fields' decrements on parents.
    UndoPendingIntoSuccess,
}

#[derive(Debug)]
pub struct Snapshot {
    /// Length of the 'undo' vector at the time we took the snapshot.
    len: usize,
}

pub use self::node_index::NodeIndex;

#[derive(Debug)]
struct Node<O> {
    state: NodeState<O>,
    parent: Option<NodeIndex>,
    root: NodeIndex, // points to the root, which may be the current node
}

/// The state of one node in some tree within the forest. This
/// represents the current state of processing for the obligation (of
/// type `O`) associated with this node.
#[derive(Debug)]
enum NodeState<O> {
    /// Obligation not yet resolved to success or error.
    Pending { obligation: O },

    /// Obligation resolved to success; `num_incomplete_children`
    /// indicates the number of children still in an "incomplete"
    /// state. Incomplete means that either the child is still
    /// pending, or it has children which are incomplete. (Basically,
    /// there is pending work somewhere in the subtree of the child.)
    ///
    /// Once all children have completed, success nodes are removed
    /// from the vector by the compression step.
    ///
    /// Successes with `num_incomplete_children == 0` are always
    /// immediately reported.
    Success { obligation: O, num_incomplete_children: usize },

    /// This obligation was resolved to an error. Error nodes are
    /// removed from the vector by the compression step. Errors
    /// are always immediately reported.
    Error,
}

#[derive(Debug)]
pub struct Outcome<O,E> {
    /// Obligations that were completely evaluated, including all
    /// (transitive) subobligations.
    pub completed: Vec<O>,

    /// Backtrace of obligations that were found to be in error.
    pub errors: Vec<Error<O,E>>,

    /// If true, then we saw no successful obligations, which means
    /// there is no point in further iteration. This is based on the
    /// assumption that when trait matching returns `Err` or
    /// `Ok(None)`, those results do not affect environmental
    /// inference state. (Note that if we invoke `process_obligations`
    /// with no pending obligations, stalled will be true.)
    pub stalled: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Error<O,E> {
    pub error: E,
    pub backtrace: Vec<O>,
}

impl<O: Clone + Debug> ObligationForest<O> {
    pub fn new() -> ObligationForest<O> {
        ObligationForest {
            nodes: vec![],
            snapshots: vec![],
            undo_log: vec![],
        }
    }

    /// Return the total number of nodes in the forest that have not
    /// yet been fully resolved.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn start_snapshot(&mut self) -> Snapshot {
        self.snapshots.push(self.undo_log.len());
        Snapshot { len: self.snapshots.len() }
    }

    pub fn commit_snapshot(&mut self, snapshot: Snapshot) {
        // Check that we are obeying stack discipline.
        assert_eq!(snapshot.len, self.snapshots.len());
        self.snapshots.pop().unwrap();
        if !self.in_snapshot() {
            self.undo_log.clear();
        }
    }

    pub fn rollback_snapshot(&mut self, snapshot: Snapshot) {
        // Check that we are obeying stack discipline.
        assert_eq!(snapshot.len, self.snapshots.len());
        let undo_len = self.snapshots.pop().unwrap();

        let undoers: Vec<_> = (undo_len..self.undo_log.len())
            .map(|_| self.undo_log.pop().unwrap())
            .collect();
        for undoer in undoers {
            undoer.undo(self);
        }
    }

    pub fn in_snapshot(&self) -> bool {
        !self.snapshots.is_empty()
    }

    /// Adds a new tree to the forest.
    ///
    /// This CAN be done during a snapshot.
    pub fn push_root(&mut self, obligation: O) {
        let index = NodeIndex::new(self.nodes.len());
        self.nodes.push(Node::new(index, None, obligation));
        if self.in_snapshot() {
            self.undo_log.push(UndoAction::UndoPush);
        }
    }

    /// Convert all remaining obligations to the given error.
    pub fn to_errors<E:Clone>(&mut self, error: E) -> Vec<Error<O,E>> {
        let mut errors = vec![];
        for index in 0..self.nodes.len() {
            debug_assert!(!self.nodes[index].is_popped());
            self.inherit_error(index);
            if let NodeState::Pending { .. } = self.nodes[index].state {
                let backtrace = self.backtrace(index);
                errors.push(Error { error: error.clone(), backtrace: backtrace });
            }
        }
        if !self.in_snapshot() {
            self.compress();
        }
        errors
    }

    /// Returns the set of obligations that are in a pending state.
    pub fn pending_obligations(&self) -> Vec<O> where O: Clone {
        self.nodes.iter()
                  .filter_map(|n| match n.state {
                      NodeState::Pending { ref obligation } => Some(obligation),
                      _ => None,
                  })
                  .cloned()
                  .collect()
    }

    /// Process the obligations.
    pub fn process_obligations<E,F>(&mut self, mut action: F) -> Outcome<O,E>
        where E: Debug, F: FnMut(&mut O, Backtrace<O>) -> Result<Option<Vec<O>>, E>
    {
        debug!("process_obligations(len={})", self.nodes.len());

        let mut errors = vec![];
        let mut stalled = true;

        // We maintain the invariant that the list is in pre-order, so
        // parents occur before their children. Also, whenever an
        // error occurs, we propagate it from the child all the way to
        // the root of the tree. Together, these two facts mean that
        // when we visit a node, we can check if its root is in error,
        // and we will find out if any prior node within this forest
        // encountered an error.

        let mut successful_obligations = Vec::new();
        for index in 0..self.nodes.len() {
            if self.in_snapshot() {
                if self.nodes[index].is_popped() {
                    continue;
                }
            } else {
                debug_assert!(!self.nodes[index].is_popped());
            }
            self.inherit_error(index);

            debug!("process_obligations: node {} == {:?}",
                   index, self.nodes[index].state);

            let result = {
                let parent = self.nodes[index].parent;
                let (prefix, suffix) = self.nodes.split_at_mut(index);
                let backtrace = Backtrace::new(prefix, parent);
                match suffix[0].state {
                    NodeState::Error |
                    NodeState::Success { .. } =>
                        continue,
                    NodeState::Pending { ref mut obligation } =>
                        action(obligation, backtrace),
                }
            };

            debug!("process_obligations: node {} got result {:?}", index, result);

            match result {
                Ok(None) => {
                    // no change in state
                }
                Ok(Some(children)) => {
                    // if we saw a Some(_) result, we are not (yet) stalled
                    stalled = false;
                    self.success(index, children, &mut successful_obligations);
                }
                Err(err) => {
                    let backtrace = self.backtrace(index);
                    errors.push(Error { error: err, backtrace: backtrace });
                }
            }
        }

        // Now we compress the result if we're not in a snapshot
        if !self.in_snapshot() {
            self.compress();
        }

        debug!("process_obligations: complete");

        Outcome {
            completed: successful_obligations,
            errors: errors,
            stalled: stalled,
        }
    }

    /// Indicates that node `index` has been processed successfully,
    /// yielding `children` as the derivative work. If children is an
    /// empty vector, this will update the ref count on the parent of
    /// `index` to indicate that a child has completed
    /// successfully. Otherwise, adds new nodes to represent the child
    /// work.
    fn success(&mut self, index: usize, children: Vec<O>, successful_obligations: &mut Vec<O>) {
        debug!("success(index={}, children={:?})", index, children);

        let num_incomplete_children = children.len();

        // change state from `Pending` to `Success`, temporarily swapping in `Error`
        let state = mem::replace(&mut self.nodes[index].state, NodeState::Error);
        self.nodes[index].state = match state {
            NodeState::Pending { obligation } =>
                NodeState::Success { obligation: obligation,
                                     num_incomplete_children: num_incomplete_children },
            NodeState::Success { .. } |
            NodeState::Error =>
                unreachable!()
        };

        // Potentially add new child work.
        if num_incomplete_children == 0 {
            // if there is no work left to be done, decrement parent's ref count
            self.update_parent(index, successful_obligations);
        } else {
            // create child work
            let root_index = self.nodes[index].root;
            let node_index = NodeIndex::new(index);
            self.nodes.extend(
                children.into_iter()
                        .map(|o| Node::new(root_index, Some(node_index), o)));
        }

        if self.in_snapshot() {
            self.undo_log.extend((0..num_incomplete_children).map(|_| UndoAction::UndoPush));
            self.undo_log.push(UndoAction::UndoModify {
                at: NodeIndex::new(index),
                undoer: UndoModify::UndoPendingIntoSuccess,
            });
        }
    }

    /// Decrements the ref count on the parent of `child`; if the
    /// parent's ref count then reaches zero, proceeds recursively.
    fn update_parent(&mut self, child: usize, successful_obligations: &mut Vec<O>) {
        debug!("update_parent(child={})", child);
        match self.nodes[child].state {
            NodeState::Success { ref obligation, .. } => {
                successful_obligations.push(obligation.clone());
            },
            _ => unreachable!(),
        }
        if let Some(parent) = self.nodes[child].parent {
            let parent = parent.get();
            match self.nodes[parent].state {
                NodeState::Success { ref mut num_incomplete_children, .. } => {
                    *num_incomplete_children -= 1;
                    if *num_incomplete_children > 0 {
                        return;
                    }
                }
                _ => unreachable!(),
            }
            self.update_parent(parent, successful_obligations);
        }
    }

    /// If the root of `child` is in an error state, places `child`
    /// into an error state. This is used during processing so that we
    /// skip the remaining obligations from a tree once some other
    /// node in the tree is found to be in error.
    fn inherit_error(&mut self, child: usize) {
        let root = self.nodes[child].root.get();
        if let NodeState::Error = self.nodes[root].state {
            let old_state = mem::replace(&mut self.nodes[child].state, NodeState::Error);
            if self.in_snapshot() {
                self.undo_log.push(UndoAction::UndoModify {
                    at: NodeIndex::new(child),
                    undoer: match old_state {
                        NodeState::Pending { obligation } =>
                            UndoModify::UndoPendingIntoError { obligation: obligation },
                        NodeState::Success { obligation, num_incomplete_children } =>
                            UndoModify::UndoSuccessIntoError {
                                obligation: obligation,
                                num_incomplete_children: num_incomplete_children,
                            },
                        _ => unreachable!()
                    }
                });
            }
        }
    }

    /// Returns a vector of obligations for `p` and all of its
    /// ancestors, putting them into the error state in the process.
    /// The fact that the root is now marked as an error is used by
    /// `inherit_error` above to propagate the error state to the
    /// remainder of the tree.
    fn backtrace(&mut self, mut p: usize) -> Vec<O> {
        let mut trace = vec![];
        loop {
            let state = mem::replace(&mut self.nodes[p].state, NodeState::Error);
            let obligation = match state {
                NodeState::Pending { obligation } => {
                    if self.in_snapshot() {
                        self.undo_log.push(UndoAction::UndoModify {
                            at: NodeIndex::new(p),
                            undoer: UndoModify::UndoPendingIntoError {
                                obligation: obligation.clone()
                            }
                        });
                    }
                    obligation
                },
                NodeState::Success { obligation, num_incomplete_children } => {
                    if self.in_snapshot() {
                        self.undo_log.push(UndoAction::UndoModify {
                            at: NodeIndex::new(p),
                            undoer: UndoModify::UndoSuccessIntoError {
                                obligation: obligation.clone(),
                                num_incomplete_children: num_incomplete_children,
                            }
                        });
                    }
                    obligation
                },
                NodeState::Error => {
                    // we should not encounter an error, because if
                    // there was an error in the ancestors, it should
                    // have been propagated down and we should never
                    // have tried to process this obligation
                    panic!("encountered error in node {:?} when collecting stack trace", p);
                },
            };
            trace.push(obligation);

            // loop to the parent
            match self.nodes[p].parent {
                Some(q) => { p = q.get(); }
                None => { return trace; }
            }
        }
    }

    /// Compresses the vector, removing all popped nodes. This adjusts
    /// the indices and hence invalidates any outstanding
    /// indices. Cannot be used during a transaction.
    fn compress(&mut self) {
        assert!(!self.in_snapshot()); // didn't write code to unroll this action
        let mut rewrites: Vec<_> = (0..self.nodes.len()).collect();

        // Finish propagating error state. Note that in this case we
        // only have to check immediate parents, rather than all
        // ancestors, because all errors have already occurred that
        // are going to occur.
        let nodes_len = self.nodes.len();
        for i in 0..nodes_len {
            if !self.nodes[i].is_popped() {
                self.inherit_error(i);
            }
        }

        // Now go through and move all nodes that are either
        // successful or which have an error over into to the end of
        // the list, preserving the relative order of the survivors
        // (which is important for the `inherit_error` logic).
        let mut dead = 0;
        for i in 0..nodes_len {
            if self.nodes[i].is_popped() {
                dead += 1;
            } else if dead > 0 {
                self.nodes.swap(i, i - dead);
                rewrites[i] -= dead;
            }
        }

        // Pop off all the nodes we killed.
        for _ in 0..dead {
            match self.nodes.pop().unwrap().state {
                NodeState::Error => { },
                NodeState::Pending { .. } => unreachable!(),
                NodeState::Success { num_incomplete_children, .. } => {
                    assert_eq!(num_incomplete_children, 0);
                }
            }
        }

        // Adjust the parent indices, since we compressed things.
        for node in &mut self.nodes {
            if let Some(ref mut index) = node.parent {
                let new_index = rewrites[index.get()];
                debug_assert!(new_index < (nodes_len - dead));
                *index = NodeIndex::new(new_index);
            }

            node.root = NodeIndex::new(rewrites[node.root.get()]);
        }
    }
}

impl<O> Node<O> {
    fn new(root: NodeIndex, parent: Option<NodeIndex>, obligation: O) -> Node<O> {
        Node {
            parent: parent,
            state: NodeState::Pending { obligation: obligation },
            root: root
        }
    }

    fn is_popped(&self) -> bool {
        match self.state {
            NodeState::Pending { .. } => false,
            NodeState::Success { num_incomplete_children, .. } => num_incomplete_children == 0,
            NodeState::Error => true,
        }
    }
}

#[derive(Clone)]
pub struct Backtrace<'b, O: 'b> {
    nodes: &'b [Node<O>],
    pointer: Option<NodeIndex>,
}

impl<'b, O> Backtrace<'b, O> {
    fn new(nodes: &'b [Node<O>], pointer: Option<NodeIndex>) -> Backtrace<'b, O> {
        Backtrace { nodes: nodes, pointer: pointer }
    }
}

impl<'b, O> Iterator for Backtrace<'b, O> {
    type Item = &'b O;

    fn next(&mut self) -> Option<&'b O> {
        debug!("Backtrace: self.pointer = {:?}", self.pointer);
        if let Some(p) = self.pointer {
            self.pointer = self.nodes[p.get()].parent;
            match self.nodes[p.get()].state {
                NodeState::Pending { ref obligation } |
                NodeState::Success { ref obligation, .. } => {
                    Some(obligation)
                }
                NodeState::Error => {
                    panic!("Backtrace encountered an error.");
                }
            }
        } else {
            None
        }
    }
}

impl<O> UndoAction<O> {
    fn undo(self, forest: &mut ObligationForest<O>) {
        match self {
            UndoAction::UndoPush => {
                forest.nodes.pop().unwrap();
            },
            UndoAction::UndoModify { at, undoer } => undoer.undo(forest, at),
        }
    }
}

impl<O> UndoModify<O> {
    fn undo(self, forest: &mut ObligationForest<O>, at: NodeIndex) {
        match self {
            UndoModify::UndoPendingIntoError { obligation } => {
                mem::replace(
                    &mut forest.nodes[at.get()].state,
                    NodeState::Pending { obligation: obligation });
            },
            UndoModify::UndoSuccessIntoError { obligation, num_incomplete_children } => {
                mem::replace(
                    &mut forest.nodes[at.get()].state,
                    NodeState::Success { obligation: obligation,
                                         num_incomplete_children: num_incomplete_children });
            },
            UndoModify::UndoPendingIntoSuccess => {
                // Restore old state
                let obligation =
                    match mem::replace(&mut forest.nodes[at.get()].state, NodeState::Error) {
                        NodeState::Success { obligation, .. } => {
                            obligation
                        },
                        _ => unreachable!()
                    };
                mem::replace(&mut forest.nodes[at.get()].state,
                             NodeState::Pending { obligation: obligation });
                // Update parents
                let mut current_node = at;
                while let Some(next_node) = forest.nodes[current_node.get()].parent {
                    match &mut forest.nodes[next_node.get()].state {
                        &mut NodeState::Success { ref mut num_incomplete_children, .. } => {
                            *num_incomplete_children += 1;
                            // If `next_node` would not have recursively updated its parent, stop
                            // recursively un-updating parents.
                            if *num_incomplete_children > 1 {
                                break;
                            }
                        },
                        _ => unreachable!()
                    }
                    current_node = next_node;
                }
            },
        }
    }
}
