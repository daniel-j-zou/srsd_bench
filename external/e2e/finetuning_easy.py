import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import sympy as sp
import re
import time
from symbolicregression.model import SymbolicTransformerRegressor
import subprocess

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
    # Ensure data is on CPU
    train_samples_cpu = train_samples.to("cpu")
    train_targets_cpu = train_targets.to("cpu")

    # Fit the estimator (train_samples and train_targets are on CPU)
    estimator.fit(train_samples_cpu, train_targets_cpu)

    # Retrieve symbolic expression (still in CPU-compatible format)
    replace_ops = {'add': '+', 'mul': '*', 'sub': '-', 'pow': '**', 'inv': '1/', 'arctan': 'atan'}
    eq_str = estimator.retrieve_tree(with_infos=True)['relabed_predicted_tree'].infix()
    for op, replace_op in replace_ops.items():
        eq_str = eq_str.replace(op, replace_op)

    # Convert to sympy expression
    sympy_eq = e2e_w_transformer2sympy(eq_str)
    return sympy_eq


# Function to load dataset
def load_dataset(dataset_file_path, delimiter=' '):
    tabular_dataset = np.loadtxt(dataset_file_path, delimiter=delimiter)
    return tabular_dataset[:, :-1], tabular_dataset[:, -1]


# Function to generate training data
def generate_training_data():
    command = [
        "python",
        "dataset_generator.py",
        "--config",
        "configs/datasets/feynman/easy_set_updated.yaml",
        "--train",
        "1.0",
        "--val",
        "0.0",
        "--test",
        "1.0",
    ]
    working_dir = "../../"  # Directory where the script should be executed

    print("Generating training data...")
    result = subprocess.run(command, cwd=working_dir, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Data generation failed: {result.stderr}")

    print("Data generation complete.")


# Set model to training mode
estimator.model.train()

# Define optimizer
optimizer = optim.Adam(estimator.model.parameters(), lr=1e-5)

# Training Loop
n_epochs = 10
criterion = nn.MSELoss()
clip_value = 1.0

train_dir = "resource/datasets/train"
test_dir = "resource/datasets/test"

for epoch in range(n_epochs):
    print(f"Epoch {epoch + 1}/{n_epochs}")
    epoch_start = time.perf_counter()

    # Step 0: Generate training data
    generate_training_data()

    # Load and iterate over all 30 training examples
    total_loss = 0
    for filename in os.listdir(train_dir):
        if filename.endswith(".txt"):
            example_start = time.perf_counter()

            train_path = os.path.join(train_dir, filename)
            test_path = os.path.join(test_dir, filename)

            # Load training and test datasets
            x1, y1 = load_dataset(train_path)
            x2, y2 = load_dataset(test_path)

            # Convert to PyTorch tensors
            x1 = torch.tensor(x1, dtype=torch.float32).to("cuda")
            y1 = torch.tensor(y1, dtype=torch.float32).view(-1, 1).to("cuda")
            x2 = torch.tensor(x2, dtype=torch.float32).to("cuda")
            y2 = torch.tensor(y2, dtype=torch.float32).view(-1, 1).to("cuda")

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Step 1: Generate a symbolic expression from x1 using the model
            sympy_expression = train(estimator, x1, y1)

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

            # Step 3: Compute the loss between y_hat and y2 using MSE Loss
            loss = criterion(y_hat, y2)

            # Step 4: Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(estimator.model.parameters(), clip_value)
            optimizer.step()

            total_loss += loss.item()

            # Print time for each example
            example_end = time.perf_counter()
            print(f"Example {filename} Time: {example_end - example_start:.4f} seconds")

    # Print loss for each epoch
    avg_loss = total_loss / 30
    print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss}")

    # Print time for each epoch
    epoch_end = time.perf_counter()
    print(f"Epoch {epoch + 1} Time: {epoch_end - epoch_start:.4f} seconds")

    # Save the entire model every 5 epochs
    if (epoch + 1) % 5 == 0:
        save_path = f"resource/ckpt/model_test_epoch{epoch + 1}.pt"
        torch.save(estimator.model, save_path)
        print(f"Model saved to {save_path}")
