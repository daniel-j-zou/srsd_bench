import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from symbolicregression.model import SymbolicTransformerRegressor
import sympy as sp
import re

# Load the model state_dict
model_path = "resource/ckpt/model_original.pt"
model = torch.load(model_path, map_location="cpu")

# Initialize the SymbolicTransformerRegressor with the model
estimator = SymbolicTransformerRegressor(
    model=model,
    max_input_points=10000,  # Example parameter; adjust based on your use case
    n_trees_to_refine=10,
    rescale=True
)

# Check if GPU is available and move model to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
estimator.model.to(device)

# Function to convert equation string to sympy expression and simplify
def e2e_w_transformer2sympy(eq_str, threshold=1e-2):
    eq_str_w_normalized_vars = re.sub(r'\bx_([0-9]*[0-9])\b', r'x\1', eq_str)
    sympy_expr = sp.parse_expr(eq_str_w_normalized_vars)
    max_iterations = 10  # Limit to avoid infinite loop
    for _ in range(max_iterations):
        updated_expr = sympy_expr.xreplace({c: 0 for c in sympy_expr.atoms(sp.Number) if abs(float(c)) < threshold})
        updated_expr = sp.simplify(updated_expr)
        if sympy_expr == updated_expr:
            break
        sympy_expr = updated_expr

    return updated_expr

# Function to train the model and generate a symbolic expression
def train(estimator, train_samples, train_targets):
    estimator.fit(train_samples, train_targets)
    replace_ops = {'add': '+', 'mul': '*', 'sub': '-', 'pow': '**', 'inv': '1/', 'arctan': 'atan'}
    eq_str = estimator.retrieve_tree(with_infos=True)['relabed_predicted_tree'].infix()
    for op, replace_op in replace_ops.items():
        eq_str = eq_str.replace(op, replace_op)
    sympy_eq = e2e_w_transformer2sympy(eq_str)
    return sympy_eq

# Sample Training Data
# x1 (1000 x p), x2 (100 x p), y (100 x p)
x1 = torch.randn(1000, 5)  # Replace with actual data
y1 = torch.randn(1000, 1)  # Replace with actual data
x2 = torch.randn(100, 5)   # Replace with actual data
y2 = torch.randn(100, 1)   # Replace with actual data

# Move inputs to the appropriate device
x1, y1, x2, y2 = x1.to(device), y1.to(device), x2.to(device), y2.to(device)

# Set model to training mode
estimator.model.train()

# Define optimizer
optimizer = optim.Adam(estimator.model.parameters(), lr=1e-5)

# Training Loop
n_epochs = 10
criterion = nn.MSELoss()

for epoch in range(n_epochs):
    # Zero the parameter gradients
    optimizer.zero_grad()

    # Step 1: Generate a symbolic expression from x1 using the model
    sympy_expression = train(estimator, x1, y1)  # Generate the symbolic expression from x1 and y1

    # Step 2: Evaluate the symbolic expression on x2 to get y_hat
    # Create a function from the sympy expression
    variables = sp.symbols('x0:%d' % x2.shape[1])
    sympy_function = sp.lambdify(variables, sympy_expression, 'numpy')

    # Convert x2 to numpy to evaluate it using the sympy function
    x2_numpy = x2.cpu().detach().numpy()
    y_hat_numpy = sympy_function(*x2_numpy.T)
    y_hat = torch.tensor(y_hat_numpy, dtype=torch.float32).to(device)

    # Step 3: Compute the loss between y_hat and y2 using MSE Loss
    loss = criterion(y_hat, y2)

    # Step 4: Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Print loss for each epoch
    print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item()}")


