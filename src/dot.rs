use std::collections::HashMap;

use crate::constraints::*;
use crate::node::*;
use crate::symbols::symbol_name;
use crate::types::{Type, TypeConstraint};

// ── Edge role for visual distinction ──

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EdgeRole {
    Control,
    Memory,
    Data,
}

fn get_edge_role(kind: NodeKind, index: usize) -> EdgeRole {
    match (kind, index) {
        (NodeKind::Entry | NodeKind::Then | NodeKind::Else | NodeKind::Region | NodeKind::Loop, _) => {
            EdgeRole::Control
        }
        (_, 0) => EdgeRole::Control,
        (NodeKind::New | NodeKind::Load | NodeKind::Store, 1) => EdgeRole::Memory,
        _ => EdgeRole::Data,
    }
}

fn get_input_port_name(kind: NodeKind, index: usize) -> &'static str {
    if kind == NodeKind::Phi && index == 0 {
        return "region";
    }
    match (kind, index) {
        (NodeKind::Entry, _) => "",
        (NodeKind::Then | NodeKind::Else, 0) => "ctrl",
        (NodeKind::Region | NodeKind::Loop, _) => "ctrl",
        (_, 0) => "ctrl",
        (NodeKind::New, 1) => "mem",
        (NodeKind::Load, 1) => "mem",
        (NodeKind::Load, 2) => "ptr",
        (NodeKind::Store, 1) => "mem",
        (NodeKind::Store, 2) => "ptr",
        (NodeKind::Store, 3) => "val",
        (NodeKind::If, 1) => "pred",
        (NodeKind::Add | NodeKind::Sub | NodeKind::Mul | NodeKind::Div, 1) => "lhs",
        (NodeKind::Add | NodeKind::Sub | NodeKind::Mul | NodeKind::Div, 2) => "rhs",
        (NodeKind::Neg | NodeKind::Not | NodeKind::StaticCast | NodeKind::DynamicCast, 1) => "val",
        (NodeKind::Phi, i) => match i {
            1 => "v0", 2 => "v1", 3 => "v2", 4 => "v3", 5 => "v4", _ => "v",
        },
        _ => "",
    }
}

fn node_color(kind: NodeKind) -> &'static str {
    match kind {
        NodeKind::Const | NodeKind::Param | NodeKind::Identity => "#1b5e20",
        NodeKind::Entry | NodeKind::If | NodeKind::Then | NodeKind::Else
        | NodeKind::Region | NodeKind::Loop => "#0d47a1",
        NodeKind::Add | NodeKind::Sub | NodeKind::Mul | NodeKind::Div
        | NodeKind::Neg | NodeKind::Not => "#f57f17",
        NodeKind::Memory | NodeKind::New | NodeKind::Load | NodeKind::Store => "#880e4f",
        NodeKind::StaticCast | NodeKind::DynamicCast => "#00695c",
        NodeKind::Phi => "#b71c1c",
        NodeKind::Unreachable => "#424242",
    }
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn format_type(t: &Type) -> String {
    match t {
        Type::Any => "Any".into(),
        Type::Never => "Never".into(),
        Type::Control => "Ctrl".into(),
        Type::Memory => "Mem".into(),
        Type::Unit => "Unit".into(),
        Type::Bool(BoolConstraint::Any) => "Bool".into(),
        Type::Bool(BoolConstraint::Const(v)) => format!("Bool({})", v),
        Type::Int(p, c) if c.is_const() => format!("{:?}={}", p, c.get_const_value().unwrap()),
        Type::Int(p, c) => format!("{:?}[{},{}]", p, c.min, c.max),
        Type::UInt(p, c) if c.is_const() => format!("{:?}={}", p, c.get_const_value().unwrap()),
        Type::UInt(p, c) => format!("{:?}[{},{}]", p, c.min, c.max),
        Type::Float(FloatConstraint::Any(p)) => format!("{:?}", p),
        Type::Float(FloatConstraint::Const(v)) => format!("F64={}", v.0),
        Type::Type(TypeConstraint::Any) => "Type".into(),
        Type::Type(TypeConstraint::Const(_)) => "Type(...)".into(),
        Type::Data(tag, _) => {
            let name = tag.and_then(|s| symbol_name(s)).unwrap_or_default();
            if name.is_empty() { "Data".into() } else { format!("Data({})", name) }
        }
        Type::Fun(_) => "Fun(...)".into(),
        Type::Union(types, _) => {
            let inner: Vec<_> = types.iter().map(format_type).collect();
            format!("Union[{}]", inner.join(", "))
        }
        Type::Error(inner, _) => format!("Error({})", format_type(inner)),
    }
}

// ── Float-rank computation ──
//
// Each edge contributes a weight based on its role:
//   control: 1.0  (strong vertical pull — clean control skeleton)
//   memory:  0.4  (medium — memory chains add mild ordering)
//   data:    0.2  (light — pure nodes cascade slowly)
//
// For node i: rank_i = max over inputs j of (rank_j + weight_{j→i})
// Roots (no inputs) stay at 0. The graph is processed in topological
// order (inputs have lower IDs), so convergence takes a few iterations.

fn edge_weight(kind: NodeKind, index: usize) -> f64 {
    match get_edge_role(kind, index) {
        EdgeRole::Control => 1.0,
        EdgeRole::Memory => 0.4,
        EdgeRole::Data => 0.2,
    }
}

fn compute_float_ranks(nodes: &[Node]) -> Vec<f64> {
    let n = nodes.len();
    let mut rank: Vec<f64> = vec![0.0; n];
    loop {
        let mut changed = false;
        for i in 1..n {
            let node = &nodes[i];
            let mut new_rank = 0.0_f64;
            for p in 0..node.inputs_len() {
                let input = node.get_input(p);
                if !input.is_valid() { continue; }
                let in_idx = input.as_usize();
                if in_idx >= n { continue; }
                if input == NodeId(i as u32) { continue; }
                let candidate = rank[in_idx] + edge_weight(node.kind, p);
                if candidate > new_rank { new_rank = candidate; }
            }
            if (new_rank - rank[i]).abs() > 1e-9 {
                rank[i] = new_rank;
                changed = true;
            }
        }
        if !changed { break; }
    }
    rank
}

/// Convert float ranks to usize groups for `rank=same`.
/// Scale ×2 so each 0.5 step → 1 integer step, round, normalize to 0.
fn float_ranks_to_groups(float_ranks: &[f64]) -> Vec<usize> {
    let n = float_ranks.len();
    if n == 0 { return vec![]; }
    let scaled: Vec<usize> = float_ranks.iter().map(|&r| {
        let s = (r * 2.0).round() as isize;
        if s < 0 { 0 } else { s as usize }
    }).collect();
    let min = *scaled.iter().min().unwrap_or(&0);
    scaled.into_iter().map(|s| s - min).collect()
}

// ── Main public entry point ──

/// Serialize the Sea of Nodes graph to Graphviz DOT format.
pub fn graph_to_dot(
    nodes: &[Node],
    _outputs: &[OutputsVec],
    var_labels: &HashMap<NodeId, String>,
) -> String {
    let n = nodes.len();
    if n == 0 {
        return String::new();
    }

    let float_ranks = compute_float_ranks(nodes);
    let ranks = float_ranks_to_groups(&float_ranks);

    let mut dot = String::new();
    dot.push_str("digraph SeaOfNodes {\n");
    dot.push_str("  rankdir=TB;\n");
    dot.push_str("  graph [fontname=\"monospace\", fontsize=10, bgcolor=\"#1e1e1e\", fontcolor=\"#eeeeee\", splines=polyline, overlap=false, sep=\"+15\"];\n");
    dot.push_str("  node [shape=none, fontname=\"monospace\", fontsize=10, style=filled, fillcolor=\"#2d2d2d\", fontcolor=\"#eeeeee\", margin=0];\n");
    dot.push_str("  edge [fontname=\"monospace\", fontsize=8, fontcolor=\"#eeeeee\"];\n\n");

    // ── Rank groups ──

    let max_rank = *ranks.iter().max().unwrap_or(&0);
    let mut rank_groups: Vec<Vec<u32>> = vec![Vec::new(); max_rank + 1];
    for i in 1..n {
        if nodes[i].kind == NodeKind::Unreachable { continue; }
        rank_groups[ranks[i]].push(i as u32);
    }
    for (r, group) in rank_groups.iter().enumerate() {
        if group.is_empty() { continue; }
        dot.push_str(&format!("  {{ rank=same; rank={};", r));
        for &id in group {
            dot.push_str(&format!(" \"{}\";", id));
        }
        dot.push_str(" }\n");
    }
    dot.push('\n');

    // ── Node definitions ──

    for i in 1..n {
        let node = &nodes[i];
        let kind = node.kind;
        if kind == NodeKind::Unreachable { continue; }

        let color = node_color(kind);
        let type_str = format_type(&node.t);
        let kind_str = format!("{:?}", kind);
        let var_anno = var_labels
            .get(&NodeId(i as u32))
            .map(|v| format!("{}: ", v))
            .unwrap_or_default();

        let num_ports = node.inputs_len();
        let colspan = if num_ports > 0 { num_ports } else { 1 };

        let mut label = String::from(
            "<<TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" CELLPADDING=\"4\">",
        );
        for part in &[i.to_string(), format!("{}{}", var_anno, kind_str), type_str] {
            let escaped = html_escape(part);
            label.push_str(&format!("<TR><TD COLSPAN=\"{}\">{}</TD></TR>", colspan, escaped));
        }
        if num_ports > 0 {
            label.push_str("<TR>");
            for p in 0..num_ports {
                let name = get_input_port_name(kind, p);
                let display = if name.is_empty() { p.to_string() } else { name.to_string() };
                label.push_str(&format!("<TD PORT=\"i{}\">{}</TD>", p, html_escape(&display)));
            }
            label.push_str("</TR>");
        }
        label.push_str("</TABLE>>");

        dot.push_str(&format!(
            "  \"{}\" [label={}, fillcolor=\"{}\"];\n", i, label, color
        ));
    }

    dot.push('\n');

    // ── Edges ──

    for i in 1..n {
        let node = &nodes[i];
        if node.kind == NodeKind::Unreachable { continue; }
        for p in 0..node.inputs_len() {
            let input = node.get_input(p);
            if !input.is_valid() || input == NodeId(i as u32) { continue; }
            let in_idx = input.as_usize();
            if in_idx >= n { continue; }

            let role = get_edge_role(node.kind, p);
            let (color, style) = match role {
                EdgeRole::Control => ("#64b5f6", "solid"),
                EdgeRole::Memory => ("#ef5350", "dashed"),
                EdgeRole::Data => ("#eeeeee", "solid"),
            };

            let port_name = get_input_port_name(node.kind, p);
            let tail_port = if port_name.is_empty() { String::new() } else { format!(":i{}", p) };
            let label = get_input_port_name(node.kind, p);

            dot.push_str(&format!(
                "  \"{}\" -> \"{}\"{} [color=\"{}\", style={}, label=\"{}\"];\n",
                in_idx, i, tail_port, color, style, label
            ));
        }
    }

    dot.push_str("}\n");
    dot
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::IRBuilder;
    use crate::types::Type;

    #[test]
    fn test_dot_simple_arithmetic() {
        let mut builder = IRBuilder::new();
        let x = builder.create_param(0, Type::I32);
        let five = Node::const_int(5);
        let add = builder.create_add(&x, &five).unwrap();
        let _add_id = builder.get_node_id(&add);

        let dot = builder.to_dot();
        assert!(dot.contains("Entry"));
        assert!(dot.contains("Param"));
        assert!(dot.contains("Add"));
        assert!(dot.contains("digraph SeaOfNodes"));
        assert!(dot.contains("<TABLE"));
        assert!(dot.contains("</TABLE>"));
    }

    #[test]
    fn test_dot_if_else_phi() {
        let mut builder = IRBuilder::new();
        let v = builder.create_variable("x");
        let five = Node::const_int(5);
        let ten = Node::const_int(10);
        let five_id = builder.intern_node(&five);
        let ten_id = builder.intern_node(&ten);

        let cond_true = Node::const_bool(true);
        let if_node = builder.create_if(NodeId(1), &cond_true);
        let then_ctrl = builder.create_then(if_node);
        let else_ctrl = builder.create_else(if_node);

        builder.set_control(then_ctrl);
        builder.write_variable(v, five_id);
        builder.set_control(else_ctrl);
        builder.write_variable(v, ten_id);

        let region = builder.create_region(&[then_ctrl, else_ctrl]);
        builder.set_control(region);
        let _result = builder.read_variable(v);

        let dot = builder.to_dot();
        assert!(dot.contains("Phi"));
        assert!(dot.contains("x:"));
        assert!(dot.contains("color=\"#64b5f6\""));
        assert!(dot.contains("color=\"#eeeeee\""));
    }

    #[test]
    fn test_dot_memory_chain() {
        let mut builder = IRBuilder::new();
        let mem = builder.get_current_memory();
        let obj_type = Type::make_record(vec![("x", Type::I32)]);
        let ptr = builder.create_new(mem, obj_type);
        let val = Node::const_int(42);
        let val_id = builder.intern_node(&val);
        let store_before = builder.get_current_memory();
        let _store = builder.create_store(store_before, ptr, val_id);

        let dot = builder.to_dot();
        assert!(dot.contains("Memory"));
        assert!(dot.contains("New"));
        assert!(dot.contains("Store"));
        assert!(dot.contains("color=\"#ef5350\""));
        assert!(dot.contains("style=dashed"));
    }

    #[test]
    fn test_dot_outputs_smoke() {
        let builder = IRBuilder::new();
        let dot = builder.to_dot();
        assert!(dot.starts_with("digraph SeaOfNodes {"));
        assert!(dot.ends_with("}\n"));
    }

    // Print float ranks for the integration scenario — for manual inspection
    #[test]
    fn test_float_ranks_display() {
        let mut builder = IRBuilder::new();
        let x_var = builder.create_variable("x");
        let y_var = builder.create_variable("y");

        let cond = Node::const_bool(true);
        let if_node = builder.create_if(NodeId(1), &cond);
        let then_ctrl = builder.create_then(if_node);
        let else_ctrl = builder.create_else(if_node);

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

        builder.set_control(else_ctrl);
        let three = Node::const_int(3);
        let mul2 = builder.create_mul(&param, &three).unwrap();
        let mul2_id = builder.get_node_id(&mul2);
        builder.write_variable(x_var, mul2_id);

        let ten = Node::const_int(10);
        let add2 = builder.create_add(&mul2, &ten).unwrap();
        let add2_id = builder.get_node_id(&add2);
        builder.write_variable(y_var, add2_id);

        let region = builder.create_region(&[then_ctrl, else_ctrl]);
        builder.set_control(region);
        let _x_phi = builder.read_variable(x_var);
        let _y_phi = builder.read_variable(y_var);

        builder.set_control(then_ctrl);
        let mem = builder.get_current_memory();
        let obj_type = Type::make_record(vec![("x", Type::I32)]);
        let ptr = builder.create_new(mem, obj_type);
        let store_mem = builder.get_current_memory();
        let _store = builder.create_store(store_mem, ptr, add_id);

        let nodes = builder.nodes();
        let float_ranks = compute_float_ranks(nodes);
        let groups = float_ranks_to_groups(&float_ranks);

        println!("=== Float Ranks ===");
        let mut pairs: Vec<(usize, f64, usize)> = (1..nodes.len())
            .map(|i| (i, float_ranks[i], groups[i]))
            .collect();
        pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));

        for (i, fr, gr) in &pairs {
            let kind = format!("{:?}", nodes[*i].kind);
            println!("  n{:>2} {:>15}  float={:5.2}  group={}", i, kind, fr, gr);
        }

        // Also emit DOT and let the user render it
        _ = builder.to_dot();
    }
}