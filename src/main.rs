mod compact_vec;
mod constraints;
mod dot;
mod node;
mod builder;
mod symbols;
mod types;

#[cfg(test)]
mod tests;

fn main() {
    let dot = build_example_graph();
    println!("{dot}");
}

/// Build a moderately complex IR graph and return its DOT representation.
///
/// Pipe the output into Graphviz, e.g.:
///
///     cargo run 2>/dev/null | dot -Tpng -o /tmp/graph.png && xdg-open /tmp/graph.png
///     cargo run 2>/dev/null | dot -Tx11
///     cargo run 2>/dev/null | dot -Tsvg -o /tmp/graph.svg && xdg-open /tmp/graph.svg
fn build_example_graph() -> String {
    use builder::IRBuilder;
    use node::{Node, NodeId};
    use types::Type;

    let mut builder = IRBuilder::new();

    // Variables for SSA
    let x_var = builder.create_variable("x");
    let y_var = builder.create_variable("y");

    // If-else diamond
    let cond = Node::const_bool(true);
    let if_node = builder.create_if(NodeId(1), &cond);
    let then_ctrl = builder.create_then(if_node);
    let else_ctrl = builder.create_else(if_node);

    // ── Then branch: x = Param(0) + 5, y = (x) * 2 ──
    builder.set_control(then_ctrl);
    let param = builder.create_param(0, Type::I32);
    let five = Node::const_int(5);
    let add = builder.create_add(&param, &five).unwrap();
    let add_id = builder.get_node_id(&add);
    builder.write_variable(x_var, add_id);

    let two = Node::const_int(2);
    let mul = builder.create_mul(&add, &two).unwrap();
    let mul_id = builder.get_node_id(&mul);
    builder.write_variable(y_var, mul_id);

    // ── Else branch: x = Param(0) * 3, y = x + 10 ──
    builder.set_control(else_ctrl);
    let three = Node::const_int(3);
    let mul2 = builder.create_mul(&param, &three).unwrap();
    let mul2_id = builder.get_node_id(&mul2);
    builder.write_variable(x_var, mul2_id);

    let ten = Node::const_int(10);
    let add2 = builder.create_add(&mul2, &ten).unwrap();
    let add2_id = builder.get_node_id(&add2);
    builder.write_variable(y_var, add2_id);

    // Merge control at Region, read both variables (creates Phis)
    let region = builder.create_region(&[then_ctrl, else_ctrl]);
    builder.set_control(region);
    let _x_phi = builder.read_variable(x_var);
    let _y_phi = builder.read_variable(y_var);

    // ── Memory chain: allocate a record and store x into it ──
    builder.set_control(then_ctrl);
    let obj_type = Type::make_record(vec![("x", Type::I32)]);
    let ptr = builder.create_new(obj_type);
    let _store = builder.create_store(ptr, add_id);

    builder.to_dot()
}