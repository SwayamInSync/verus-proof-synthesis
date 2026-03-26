use vstd::prelude::*;
use verus_state_machines_macros::*;

fn main() {}
pub type ReqId = nat;
pub type LogIdx = nat;

verus!{

// File: spec/simple_log.rs
pub ghost enum ReadReq<R> {
    /// a new read request that has entered the system
    Init { op: R },
    /// a request that has been dispatched at a specific version
    Req { version: LogIdx, op: R },
}

pub ghost struct UpdateResp(pub LogIdx);

state_machine! {
    SimpleLog<DT: Dispatch> {

    fields {
        /// a sequence of update operations,
        pub log: Seq<DT::WriteOperation>,
        /// the completion tail current index into the log
        pub version: LogIdx,
        /// in flight read requests
        pub readonly_reqs: Map<ReqId, ReadReq<DT::ReadOperation>>,
        /// inflight update requests
        pub update_reqs: Map<ReqId, DT::WriteOperation>,
        /// responses to update requests that haven't been returned
        pub update_resps: Map<ReqId, UpdateResp>,
    }
    }
}


// File: lib.rs
#[verus::trusted]
#[is_variant]
pub enum InputOperation<DT: Dispatch> {
    Read(DT::ReadOperation),
    Write(DT::WriteOperation),
}

#[verus::trusted]
#[is_variant]
pub enum OutputOperation<DT: Dispatch> {
    Read(DT::Response),
    Write(DT::Response),
}

#[verus::trusted]
#[is_variant]
pub enum AsyncLabel<DT: Dispatch> {
    Internal,
    Start(ReqId, InputOperation<DT>),
    End(ReqId, OutputOperation<DT>),
}

state_machine!{ AsynchronousSingleton<DT: Dispatch> {           // $line_count$Trusted$

    fields {                                                    // $line_count$Trusted$
        pub state: DT::View,                                    // $line_count$Trusted$
        pub reqs: Map<ReqId, InputOperation<DT>>,               // $line_count$Trusted$
        pub resps: Map<ReqId, OutputOperation<DT>>,             // $line_count$Trusted$
    }                                                           // $line_count$Trusted$
                                                                
}
}

#[verus::trusted]
#[is_variant]
pub enum SimpleLogBehavior<DT: Dispatch> {
    Stepped(SimpleLog::State<DT>, AsyncLabel<DT>, Box<SimpleLogBehavior<DT>>),
    Inited(SimpleLog::State<DT>),
}

#[verus::trusted]
#[is_variant]
pub enum AsynchronousSingletonBehavior<DT: Dispatch> {
    Stepped(
        AsynchronousSingleton::State<DT>,
        AsyncLabel<DT>,
        Box<AsynchronousSingletonBehavior<DT>>,
    ),
    Inited(AsynchronousSingleton::State<DT>),
}

#[verus::trusted]
pub open spec fn behavior_equiv<DT: Dispatch>(
    a: SimpleLogBehavior<DT>,
    b: AsynchronousSingletonBehavior<DT>,
) -> bool
    decreases a, b,
{
    arbitrary() // TODO: fill in the spec body
}

#[verus::trusted]
pub trait Dispatch: Sized {
    /// Type of a read-only operation. Operations of this type do not mutate the data structure.
    type ReadOperation: Sized;

    /// Type of a write operation. Operations of this type may mutate the data structure.
    /// Write operations are sent between replicas.
    type WriteOperation: Sized + Send;

    /// Type of the response of either a read or write operation.
    type Response: Sized;

    /// Type of the view of the data structure for specs and proofs.
    type View;
}



// File: spec/linearization.rs
proof fn trick_equiv<DT: Dispatch>(
    a: SimpleLogBehavior<DT>,
    a2: SimpleLogBehavior<DT>,
    b: AsynchronousSingletonBehavior<DT>,
)
    requires
        behavior_equiv(a, b),
        a.is_Stepped(),
        a2.is_Stepped(),
        a.get_Stepped_2() == a2.get_Stepped_2(),
        a.get_Stepped_1().is_Internal(),
        a2.get_Stepped_1().is_Internal(),
    ensures
        behavior_equiv(a2, b),
    decreases b,
{
    if !b.is_Inited() && behavior_equiv(a, *b.get_Stepped_2()) {
        trick_equiv(a, a2, *b.get_Stepped_2());
    }
}


}
