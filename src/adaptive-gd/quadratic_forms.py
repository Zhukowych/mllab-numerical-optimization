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
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    from scipy.stats import ortho_group

    from pathlib import Path
    return NDArray, Path, go, jax, jnp, make_subplots, np, ortho_group, pl


@app.cell
def _(jax):
    jax.config.update("jax_enable_x64", True)
    return


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
    min_ev = mo.ui.slider(start=0, stop=10000, step=0.1)
    max_ev = mo.ui.slider(start=0, stop=10000, step=0.1)
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
    For step $\frac{\nabla f(x) ^ \top \nabla f(x)}{\nabla f(x) ^ \top A\nabla f(x)}$, experimentaly test whether it convegece to $\frac{2}{\lambda_{\min} + \lambda_{\max}}$
    """)
    return


@app.cell
def _(mo):
    lower_dim = mo.ui.slider(start=0, stop=500, step=1)
    upper_dim = mo.ui.slider(start=0, stop=500, step=1)
    initial_point_samples_num = mo.ui.slider(start=1, stop=100, step=1)
    gd_num_iterations = mo.ui.slider(start = 100, stop = 3000, step = 1)
    exp_name = mo.ui.text()
    return (
        exp_name,
        gd_num_iterations,
        initial_point_samples_num,
        lower_dim,
        upper_dim,
    )


@app.cell
def _(Path, QuadraticForm, jax, jnp, mo, pl):
    def run_experiment(min_ev: float, max_ev: float, lower_dim: int, upper_dim: int, gd_num_iterations: int, initial_point_samples_num: int, experiment_name: str, null_space_dims: tuple = (0, 5), device_num: int = 1) -> None:
        with jax.default_device(jax.devices("cuda")[1]):
            for curr_dim in mo.status.progress_bar(range(lower_dim, upper_dim + 1), title="Dimension of function", show_eta=True, show_rate=True):


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

                for null_space_dim_curr in null_space_dims:
                    f = QuadraticForm(min_ev, max_ev, curr_dim, null_space_dim_curr)
                    f.convert_to_jax()



                    for init_point_idx in mo.status.progress_bar(range(initial_point_samples_num), title="Initial point", show_eta=True, show_rate=True):
                        x = jax.random.uniform(
                            key = jax.random.PRNGKey(curr_dim),
                            shape=(curr_dim,),
                        )

                        x /= jnp.linalg.norm(x)
                        x *= 100000

                        with mo.status.spinner(subtitle="Descending...") as _spinner:
                            for i in range(gd_num_iterations):
                                grad_i = f.calculate_gradient(x)
                                lambda_i = (grad_i.T @ grad_i) / (grad_i.T @ f.A @grad_i)

                                if True or i < 100 or i % 100 == 0:

                                    results["dimension"].append(curr_dim)
                                    results["kernel_size"].append(null_space_dim_curr)
                                    results["max_eigenvalue"].append(f.eigen_vals[-1])
                                    results["min_eigenvalue"].append(f.eigen_vals[null_space_dim_curr])
                                    results["max_eigenvalue_inverse"].append(1 / f.eigen_vals[-1])
                                    results["initial_point_index"].append(init_point_idx)
                                    results["iteration"].append(i)
                                    results["function_value"].append(f.calculate_function(x))

                                    grad_norm = jnp.linalg.norm(grad_i)

                                    results["gradient_norm"].append(grad_norm)
                                    results["learning_rate"].append(lambda_i)

                                    _spinner.update(f"max_ev: {f.eigen_vals[-1]}, min_ev: {f.eigen_vals[null_space_dim_curr]}, max_ev_inv: {1 / f.eigen_vals[-1]}, lr: {lambda_i}, gradient norm: {grad_norm}") 

                                x -= lambda_i * grad_i

                results_table = pl.DataFrame(results)

                save_path = Path(f"./data/{experiment_name}")
                save_path.mkdir(parents=True, exist_ok=True)

                results_table.write_parquet(save_path / f"gd_results_{curr_dim}.parquet")
    return (run_experiment,)


@app.cell
def _(
    exp_name,
    gd_num_iterations,
    initial_point_samples_num,
    lower_dim,
    max_ev,
    min_ev,
    mo,
    upper_dim,
):
    mo.vstack(
        [
            mo.hstack([mo.md("Minimum eigenvalue"), min_ev]),
            mo.hstack([mo.md("Maximum eigenvalue"), max_ev]),
            mo.hstack([mo.md("Starting dimension of quadratic form"), lower_dim]),
            mo.hstack([mo.md("Maximum dimension of quadratic form"), upper_dim]),
            mo.hstack([mo.md("Initial point samples per quadratic form"), initial_point_samples_num]),
            mo.hstack([mo.md("G.D. number of iterations"), gd_num_iterations]),
            mo.md(f"Input experiment name: {exp_name}")
        ]
    )
    return


@app.cell
def _(
    exp_name,
    gd_num_iterations,
    initial_point_samples_num,
    lower_dim,
    max_ev,
    min_ev,
    run_experiment,
    upper_dim,
):
    run_experiment(min_ev=min_ev.value, max_ev=max_ev.value, lower_dim = lower_dim.value, upper_dim = upper_dim.value, gd_num_iterations = gd_num_iterations.value, initial_point_samples_num=initial_point_samples_num.value, experiment_name=exp_name.value, device_num = 1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Hypothesis whether learning rate converges to $\frac{2}{\lambda_{\min} + \lambda_{\max}}$
    """)
    return


@app.cell
def _(Path, exp_name, pl):
    results_df_test = pl.read_parquet(
        list(Path(f"data/{exp_name.value}").rglob("*.parquet"))
    )
    return (results_df_test,)


@app.cell
def _(pl, results_df_test):
    results_lr_hyp = results_df_test.with_columns(
        pl.lit(2).truediv(pl.col("max_eigenvalue").add(pl.col("min_eigenvalue"))).alias("best_lr"),
        pl.lit(1).truediv(pl.col("max_eigenvalue").add(pl.col("min_eigenvalue"))).alias("half_best_lr"),
        pl.lit(1).truediv(pl.col("max_eigenvalue")).alias("max_ev_inv"),
        pl.lit(1).truediv(pl.col("min_eigenvalue")).alias("min_ev_inv"),
    ).with_columns(
        pl.col("learning_rate").sub(pl.col("best_lr")).abs().alias("lr_error"),
        harmonic_mean_pair = (
            2 / (1 / pl.col("learning_rate")).sum()
        ).over(pl.int_range(0, pl.len()) // 2) 
    )
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
    what_to_plot = mo.ui.multiselect(options=["gradient_norm", "learning_rate", "best_lr", "lr_error", "harmonic_mean_pair", "min_ev_inv", "max_ev_inv"])
    return dim_to_plot, init_point_to_plot, ker_to_plot, what_to_plot


@app.cell
def _(dim_to_plot, init_point_to_plot, ker_to_plot, mo, what_to_plot):
    plot_constructor = mo.vstack(
        [
            mo.hstack([mo.md("Dim of function"), dim_to_plot]),
            mo.hstack([mo.md("Kernel size of function"), ker_to_plot]),
            mo.hstack([mo.md("Initial point index"), init_point_to_plot]),
            mo.hstack([mo.md("What to plot on y-axis?"), what_to_plot]),
        ]
    )
    return (plot_constructor,)


@app.cell
def _(
    dim_to_plot,
    go,
    init_point_to_plot,
    ker_to_plot,
    make_subplots,
    mo,
    pl,
    results_lr_hyp,
    what_to_plot,
):
    chosen_data = results_lr_hyp.filter(pl.col("dimension").eq(dim_to_plot.value) 
                                        & pl.col("kernel_size").eq(ker_to_plot.value)
                                        & pl.col("initial_point_index").eq(init_point_to_plot.value) 
                                       )



    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, shared_yaxes=True)
    colors = ["#FDFD96", "#FFD1DC", ]

    for i ,y_axis in enumerate(what_to_plot.value):

        fig.add_trace(go.Scatter(x=chosen_data["iteration"], y=chosen_data[y_axis],
                                 mode='markers',
                            name=y_axis),
                      1, 1)

    fig.update_xaxes(title_text="Iteration Number")
    fig.update_yaxes(title_text="Value (Inverse Eigenvalue / Learning Rate)")
    fig.update_layout()
    gd_results_plot = mo.ui.plotly(fig)
    return (gd_results_plot,)


@app.cell
def _(gd_results_plot, mo, plot_constructor):
    mo.vstack([plot_constructor, gd_results_plot])
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
