import json
import numpy as np

notebook_path = '/home/zhukowych/Projects/ucu/mllab/mllab-numerical-optimization/notebooks/quadratic_optimization.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

cells = nb['cells']
insert_index = -1

# Find the index of the cell with the specific markdown content
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'markdown':
        source = "".join(cell['source'])
        if "## Quadratic optimization problem under exact line search" in source:
            insert_index = i
            break

if insert_index == -1:
    print("Target cell not found, appending to the end")
    insert_index = len(cells)

new_markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "Different matrix sizes"
    ]
}

new_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "matrix_sizes = [5, 10, 50, 100, 500]\n",
        "cond_num = 10\n",
        "max_iters = 1000\n",
        "tol = 1e-6\n",
        "\n",
        "plt.figure(figsize=(12, 8))\n",
        "\n",
        "for n in matrix_sizes:\n",
        "    eig_min = 1.0 \n",
        "    eig_max = eig_min * cond_num\n",
        "    \n",
        "    x0 = 10 * np.ones(n)\n",
        "\n",
        "    A, b = generate_random_quadratic_function(\n",
        "        n=n,\n",
        "        eig_min=eig_min,\n",
        "        eig_max=eig_max,\n",
        "        null_space_size=0,\n",
        "        seed=42\n",
        "    )\n",
        "\n",
        "    lr = 2 / (eig_max + eig_min)\n",
        "\n",
        "    optimizer, values = gradient_descent_quadratic(A, b, x0=x0, max_iters=max_iters, lr=lr, tol=tol)\n",
        "\n",
        "    x_star = -np.linalg.inv(A) @ b\n",
        "    f_star = quadratic(A, b, x_star)\n",
        "    error = np.array(values).flatten() - f_star.item()\n",
        "\n",
        "    plt.plot(error, label=f'Size: {n}')\n",
        "\n",
        "plt.yscale('log')\n",
        "plt.xlabel('Iteration')\n",
        "plt.ylabel('$f(x_k) - f(x^*)$')\n",
        "plt.title('Error to Minimal Value vs. Iterations for Different Matrix Sizes')\n",
        "plt.grid(True)\n",
        "plt.legend()\n",
        "plt.show()"
    ]
}

# Insert the new cells
cells.insert(insert_index, new_code_cell)
cells.insert(insert_index, new_markdown_cell)

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Inserted cells at index {insert_index}")
