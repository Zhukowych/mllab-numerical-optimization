import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import jax

    import numpy as np
    import jax.numpy as jnp

    from numpy.typing import NDArray

    import polars as pl
    import plotly.express as px

    from scipy.stats import ortho_group

    import polars as pl
    return NDArray, jax, jnp, np, ortho_group, pl, px


@app.cell
def _(mo):
    mo.md(r"""
    # Random quadratic form generation
    """)
    return


@app.cell
def _(NDArray, NDarray, np, ortho_group):
    def gen_eigenvalues(
            min_ev: float = 0.0, max_ev: float = 1e9, dim: int = 3, null_space_dim: int = 0
        ) -> NDArray[np.float64]:
        non_zero_evs = np.random.uniform(low=min_ev, high=max_ev, size=dim - null_space_dim)
        return np.hstack((np.zeros(null_space_dim), non_zero_evs))


    def gen_eigenvectors(dim: int = 3) -> NDarray[np.float64]:
        return ortho_group.rvs(dim=dim)
    return gen_eigenvalues, gen_eigenvectors


@app.cell
def _(NDArray, gen_eigenvalues, gen_eigenvectors, jnp, np):
    class QuadraticForm:
        def __init__(self, min_ev: float = 0.0, max_ev: float = 1e9, dim: int = 3, null_space_dim: int = 0) -> None:
            self.eigen_vals = np.sort(gen_eigenvalues(min_ev=min_ev, max_ev=max_ev, dim=dim))
            jordan_form = np.diag(self.eigen_vals)

            eigen_vecs = gen_eigenvectors(dim=dim)
            inv_eigen_vec = np.linalg.inv(eigen_vecs)

            self.b = np.random.rand(dim)

            self.A = eigen_vecs @ jordan_form @ inv_eigen_vec

        def convert_to_jax(self) -> None:
            self.A = jnp.array(self.A)
            self.b = jnp.array(self.b)

        def calculate_gradient(self, x: NDArray | jnp.ndarray) -> NDArray | jnp.ndarray:
            return self.A @ x - self.b

        def calculate_function(self, x: NDArray | jnp.ndarray) -> NDArray | jnp.ndarray:
            return 0.5 * x.T @ self.A @ x - self.b @ x
    return (QuadraticForm,)


@app.cell
def _(mo):
    min_ev = mo.ui.slider(start=0, stop=1000, step=0.1)
    max_ev = mo.ui.slider(start=0, stop=1000, step=0.1)
    dim = mo.ui.slider(start=0, stop=100, step=1)
    null_space_dim = mo.ui.slider(start=0, stop=100, step=1)
    return dim, max_ev, min_ev, null_space_dim


@app.cell
def _(dim, max_ev, min_ev, mo, null_space_dim):
    mo.vstack(
        [
            mo.hstack([mo.md("Minimal eigenvalue"), min_ev]),
            mo.hstack([mo.md("Maximum eigenvalue"), max_ev]),
            mo.hstack([mo.md("Quadratic form dimension"), dim]),
            mo.hstack([mo.md("Null space dimension of matrix $A$"), null_space_dim]),
        ]
    )
    return


@app.cell
def _(QuadraticForm, dim, max_ev, min_ev, null_space_dim):
    f1 = QuadraticForm(min_ev.value, max_ev.value, dim.value, null_space_dim.value)
    return (f1,)


@app.cell
def _(f1):
    f1.A, f1.b, f1.eigen_vals
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Convergence test
    For step $\frac{\nabla f(x) ^ \top \nabla f(x)}{\nabla f(x) ^ \top A\nabla f(x)}$, experimentaly test whether it convegece to $\frac{1}{L}$
    """)
    return


@app.cell
def _(mo):
    lower_dim = mo.ui.slider(start=0, stop=100, step=1)
    upper_dim = mo.ui.slider(start=0, stop=100, step=1)
    initial_point_samples_num = mo.ui.slider(start=1, stop=100, step=1)
    gd_num_iterations = mo.ui.slider(start = 100, stop = 3000, step = 1)
    return gd_num_iterations, initial_point_samples_num, lower_dim, upper_dim


@app.cell
def _(
    gd_num_iterations,
    initial_point_samples_num,
    lower_dim,
    max_ev,
    mo,
    upper_dim,
):
    mo.vstack(
        [
            mo.hstack([mo.md("Maximum eigenvalue"), max_ev]),
            mo.hstack([mo.md("Starting dimension of quadratic form"), lower_dim]),
            mo.hstack([mo.md("Maximum dimension of quadratic form"), upper_dim]),
            mo.hstack([mo.md("Initial point samples per quadratic form"), initial_point_samples_num]),
            mo.hstack([mo.md("G.D. number of iterations"), gd_num_iterations]),
        ]
    )
    return


@app.cell
def _():
    device = "cuda"
    return (device,)


@app.cell
def _():
    results = {
        "dimension": [],
        "kernel_size": [],
        "initial_point_index": [],
        "iteration": [],
        "function_value": [],
        "gradient_norm": [],
        "learning_rate": [], 
        "max_eigenvalue": [],
        "min_eigenvalue": [],
        "max_eigenvalue_inverse": [],
    }
    return (results,)


@app.cell
def _(
    QuadraticForm,
    device,
    gd_num_iterations,
    initial_point_samples_num,
    jax,
    jnp,
    lower_dim,
    max_ev,
    mo,
    results,
    upper_dim,
):
    with jax.default_device(jax.devices(device)[0]):
        for curr_dim in mo.status.progress_bar(range(lower_dim.value, upper_dim.value + 1), title="Dimension of function", show_eta=True, show_rate=True):
            min_ev_random, max_ev_random = jnp.sort(jax.random.uniform(
                key = jax.random.PRNGKey(curr_dim),
                shape=(2,),
                minval=0,
                maxval=max_ev.value,
            ))
            for null_space_dim_curr in (0, 5):
                f = QuadraticForm(min_ev_random, max_ev_random, curr_dim, null_space_dim_curr)
                f.convert_to_jax()



                for init_point_idx in mo.status.progress_bar(range(initial_point_samples_num.value), title="Initial point", show_eta=True, show_rate=True):
                    x = jax.random.uniform(
                        key = jax.random.PRNGKey(curr_dim),
                        shape=(curr_dim,),
                        minval=-100,
                        maxval=100,
                    )

                    with mo.status.spinner(subtitle="Descending...") as _spinner:
                        for i in range(gd_num_iterations.value):
                            grad_i = f.calculate_gradient(x)
                            lambda_i = (grad_i.T @ grad_i) / (grad_i.T @ f.A @grad_i)

                            if (i % 100 == 0):

                                results["dimension"].append(curr_dim)
                                results["kernel_size"].append(null_space_dim_curr)
                                results["max_eigenvalue"].append(f.eigen_vals[-1])
                                results["min_eigenvalue"].append(f.eigen_vals[null_space_dim_curr])
                                results["max_eigenvalue_inverse"].append(1 / f.eigen_vals[-1])
                                results["initial_point_index"].append(init_point_idx)
                                results["iteration"].append(i)
                                results["function_value"].append(f.calculate_function(x))
                                results["gradient_norm"].append(jnp.linalg.norm(grad_i))
                                results["learning_rate"].append(lambda_i)

                                _spinner.update(f"max_ev: {f.eigen_vals[-1]}, min_ev: {f.eigen_vals[null_space_dim_curr]}, max_ev_inv: {1 / f.eigen_vals[-1]}, lr: {lambda_i}") 

                            x -= lambda_i * grad_i
    return


@app.cell
def _(pl, results):
    results_df = pl.DataFrame(results)
    return (results_df,)


@app.cell
def _(results_df):
    results_df.write_parquet("quadratic_forms_gd_results_3000_iters.parquet")
    return


@app.cell
def _(pl):
    results_df_loaded = pl.read_parquet("quadratic_forms_gd_results_3000_iters.parquet")
    return (results_df_loaded,)


@app.cell
def _(mo, results_df_loaded):
    mo.ui.dataframe(results_df_loaded)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Hypothesis whether learning rate converges to $\frac{2}{\lambda_{\min} + \lambda_{\max}}$
    """)
    return


@app.cell
def _(pl, results_df_loaded):
    results_lr_hyp = results_df_loaded.with_columns(
    pl.lit(2).truediv(pl.col("max_eigenvalue").add(pl.col("min_eigenvalue"))).alias("best_lr")
    ).with_columns(
        pl.col("learning_rate").sub(pl.col("best_lr")).abs().alias("lr_error")
    ).group_by("dimension", "kernel_size", "initial_point_index").agg("iteration", "gradient_norm", "learning_rate", "best_lr", "lr_error").explode("iteration", "gradient_norm", "learning_rate", "best_lr", "lr_error")
    return (results_lr_hyp,)


@app.cell
def _(mo, results_lr_hyp):
    mo.ui.dataframe(results_lr_hyp)
    return


@app.cell
def _(mo, results_lr_hyp):
    dim_to_plot = mo.ui.dropdown(options=results_lr_hyp["dimension"].unique())
    ker_to_plot = mo.ui.dropdown(options=results_lr_hyp["kernel_size"].unique())
    init_point_to_plot = mo.ui.dropdown(options=results_lr_hyp["initial_point_index"].unique())
    what_to_plot = mo.ui.dropdown(options=["gradient_norm", "learning_rate", "best_lr", "lr_error"])
    return dim_to_plot, init_point_to_plot, ker_to_plot, what_to_plot


@app.cell
def _(dim_to_plot, init_point_to_plot, ker_to_plot, mo, what_to_plot):
    mo.vstack(
        [
            mo.hstack([mo.md("Dim of function"), dim_to_plot]),
            mo.hstack([mo.md("Kernel size of function"), ker_to_plot]),
            mo.hstack([mo.md("Initial point index"), init_point_to_plot]),
            mo.hstack([mo.md("What to plot on y-axis?"), what_to_plot]),
        ]
    )
    return


@app.cell
def _(
    dim_to_plot,
    init_point_to_plot,
    ker_to_plot,
    pl,
    px,
    results_lr_hyp,
    what_to_plot,
):
    chosen_data = results_lr_hyp.filter(pl.col("dimension").eq(dim_to_plot.value) 
                                        & pl.col("kernel_size").eq(ker_to_plot.value)
                                        & pl.col("initial_point_index").eq(init_point_to_plot.value) 
                                        & pl.col("iteration").ge(500)
                                       )
    y_axis = what_to_plot.value
    y_min, y_max = chosen_data[y_axis].min(), chosen_data[y_axis].max()

    px.scatter(
        chosen_data, x="iteration", y=y_axis, range_y=[y_min, y_max]
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
