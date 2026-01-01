/// Operation types for the autodiff DAG
pub const Op = enum {
    none, // Leaf node (input/parameter)
    add,
    sub,
    mul,
    div,
    pow,
    neg,
    exp,
    log,
    tanh,
    relu,
};
