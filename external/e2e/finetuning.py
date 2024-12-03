import torch
import torch.optim as optim
import torch.nn as nn
import sympy as sp
import re
import time
from symbolicregression.model import SymbolicTransformerRegressor

# Load the model state_dict
model_path = "resource/ckpt/model_original.pt"
print("Loading model from:", model_path)
model = torch.load(model_path, map_location="cpu")

# Initialize the SymbolicTransformerRegressor with the model
estimator = SymbolicTransformerRegressor(
    model=model,
    max_input_points=200,  # Example parameter; adjust based on your use case
    n_trees_to_refine=10,
    rescale=True
)

# Move the model to GPU
estimator.model.to("cuda")


# Function to convert equation string to sympy expression and simplify
def e2e_w_transformer2sympy(eq_str, threshold=1e-2):
    eq_str_w_normalized_vars = re.sub(r'\bx_([0-9]*[0-9])\b', r'x\1', eq_str)
    sympy_expr = sp.parse_expr(eq_str_w_normalized_vars)
    return sympy_expr


# Function to train the model and generate a symbolic expression
def train(estimator, train_samples, train_targets):
    # Ensure data is on GPU
    train_samples = train_samples.to("cuda")
    train_targets = train_targets.to("cuda")

    # Fit the estimator (train_samples and train_targets are on GPU)
    estimator.fit(train_samples, train_targets)

    # Retrieve symbolic expression (still in CPU-compatible format)
    replace_ops = {'add': '+', 'mul': '*', 'sub': '-', 'pow': '**', 'inv': '1/', 'arctan': 'atan'}
    eq_str = estimator.retrieve_tree(with_infos=True)['relabed_predicted_tree'].infix()
    for op, replace_op in replace_ops.items():
        eq_str = eq_str.replace(op, replace_op)

    # Convert to sympy expression
    sympy_eq = e2e_w_transformer2sympy(eq_str)
    return sympy_eq


# Sample Training Data (stay on CPU initially)
x1 = torch.randn(200, 5)  # Replace with actual data
y1 = torch.randn(200, 1)  # Replace with actual data
x2 = torch.randn(200, 5)  # Replace with actual data
y2 = torch.randn(200, 1)  # Replace with actual data

print("Data loaded.")

# Set model to training mode
estimator.model.train()

# Define optimizer
optimizer = optim.Adam(estimator.model.parameters(), lr=1e-5)

# Training Loop
n_epochs = 10
criterion = nn.MSELoss()
clip_value = 1.0

for epoch in range(n_epochs):
    print(f"Epoch {epoch + 1}/{n_epochs}")
    start = time.perf_counter()

    # Zero the parameter gradients
    optimizer.zero_grad()

    # Step 1: Generate a symbolic expression from x1 using the model
    sympy_expression = train(estimator, x1, y1)  # x1, y1 remain on CPU initially
    step1 = time.perf_counter()
    print("Step 1 Time: ", step1 - start)

    # Step 2: Evaluate the symbolic expression on x2 to get y_hat
    # Convert x2 to CPU numpy for sympy evaluation
    x2_numpy = x2.cpu().detach().numpy()
    variables = sp.symbols('x0:%d' % x2.shape[1])
    sympy_function = sp.lambdify(variables, sympy_expression, 'numpy')

    # Evaluate symbolic expression
    y_hat_numpy = sympy_function(*x2_numpy.T)

    # Convert y_hat back to GPU tensor
    y_hat = torch.tensor(y_hat_numpy, dtype=torch.float32, requires_grad=True).to("cuda")
    y_hat = y_hat.view(-1, 1)  # Reshape to match y2 size
    step2 = time.perf_counter()
    print("Step 2 Time: ", step2 - step1)

    # Step 3: Compute the loss between y_hat and y2 using MSE Loss
    loss = criterion(y_hat, y2.to("cuda"))
    step3 = time.perf_counter()
    print("Step 3 Time: ", step3 - step2)

    # Step 4: Backward pass and optimization
    loss.backward()
    torch.nn.utils.clip_grad_norm_(estimator.model.parameters(), clip_value)
    optimizer.step()
    step4 = time.perf_counter()
    print("Step 4 Time: ", step4 - step3)

    # Print loss for each epoch
    print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item()}")
