import pathlib
import numpy as np
import onnx
import onnx.checker
from onnx import TensorProto, helper

# 1. Use a fixed seed
np.random.seed(42)

# Generate fixed random weights and biases
w1_data = np.random.randn(4, 8).astype(np.float32).flatten()
b1_data = np.random.randn(8).astype(np.float32).flatten()
w2_data = np.random.randn(8, 2).astype(np.float32).flatten()
b2_data = np.random.randn(2).astype(np.float32).flatten()

# 2. Graph Inputs and Outputs
X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2])

# 3. Initializers (Weights and Biases constants)
w1 = helper.make_tensor('W1', TensorProto.FLOAT, [4, 8], w1_data)
b1 = helper.make_tensor('B1', TensorProto.FLOAT, [8], b1_data)
w2 = helper.make_tensor('W2', TensorProto.FLOAT, [8, 2], w2_data)
b2 = helper.make_tensor('B2', TensorProto.FLOAT, [2], b2_data)

# 4. Explicitly using ONLY MatMul, Add, and Relu (Opset 13)
nodes = [
    # Layer 1
    helper.make_node('MatMul', ['input', 'W1'], ['matmul1']),
    helper.make_node('Add', ['matmul1', 'B1'], ['add1']),
    helper.make_node('Relu', ['add1'], ['relu1']),
    # Layer 2
    helper.make_node('MatMul', ['relu1', 'W2'], ['matmul2']),
    helper.make_node('Add', ['matmul2', 'B2'], ['add2']),
    helper.make_node('Relu', ['add2'], ['output'])
]

# Assemble the Graph
graph = helper.make_graph(
    nodes, 'mlp_4_8_2', [X], [Y],
    initializer=[w1, b1, w2, b2]
)

# 5. Pin opset 13 
model = helper.make_model(
    graph,
    producer_name='onnxcc_generate_test_models',
    opset_imports=[helper.make_opsetid('', 13)]
)

# 6. Validate 
onnx.checker.check_model(model)

# 7. Save 
out_dir = pathlib.Path(__file__).resolve().parent.parent / 'tests' / 'fixtures'
out_dir.mkdir(parents=True, exist_ok=True)
model_path = out_dir / 'mlp_4_8_2.onnx'
onnx.save(model, model_path)

# 8. Generate and save the matching (1,4) input as raw 16 bytes
input_data = np.random.randn(1, 4).astype(np.float32)
input_path = out_dir / 'mlp_4_8_2_input.bin'
input_data.tofile(input_path)

# 9. Report 
node_ops = [n.op_type for n in graph.node]
print(f'Wrote model: {model_path}')
print(f'Wrote input: {input_path} ({input_path.stat().st_size} bytes)')
print(f'Node op types ({len(node_ops)}): {node_ops}')
print(f'Initializer count: {len(graph.initializer)}')