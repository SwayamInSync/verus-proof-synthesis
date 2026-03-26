use vstd::prelude::*;

fn main() {}

verus!{

// File: src/properties.rs
pub trait SpecCombinator {

    type Type;

    open spec fn wf(&self, v: Self::Type) -> bool {
        true
    }

    open spec fn requires(&self) -> bool {
        true
    }

    spec fn spec_parse(&self, s: Seq<u8>) -> Option<(int, Self::Type)>;

    spec fn spec_serialize(&self, v: Self::Type) -> Seq<u8>;

}

pub trait SecureSpecCombinator: SpecCombinator {
    arbitrary() // TODO: fill in the spec body
}




}
