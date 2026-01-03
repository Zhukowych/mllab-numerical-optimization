import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    pass


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

    from scipy.stats import ortho_group, wilcoxon


    from pathlib import Path
    return (
        NDArray,
        Path,
        go,
        jax,
        jnp,
        make_subplots,
        np,
        ortho_group,
        pl,
        px,
        wilcoxon,
    )


@app.cell
def _(jax, np):
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(seed=42)
    return (rng,)


@app.cell
def _(mo):
    mo.md(r"""
    # Random quadratic form generation
    """)
    return


@app.cell
def _(NDArray, NDarray, np, ortho_group, rng):
    def gen_eigenvalues(
        min_ev: float = 0.0, max_ev: float = 1e9, dim: int = 3, null_space_dim: int = 0
    ) -> NDArray[np.float64]:
        non_zero_evs = rng.uniform(low=min_ev, high=max_ev, size=dim - null_space_dim)
        return np.hstack((np.zeros(null_space_dim), non_zero_evs))


    def gen_eigenvectors(dim: int = 3) -> NDarray[np.float64]:
        return ortho_group.rvs(dim=dim, random_state=rng)
    return gen_eigenvalues, gen_eigenvectors


@app.cell
def _(NDArray, gen_eigenvalues, gen_eigenvectors, jnp, np):
    class QuadraticForm:
        def __init__(self, min_ev: float = 0.0, max_ev: float = 1e9, dim: int = 3, null_space_dim: int = 0) -> None:
            self.dim = dim
            self.null_space_dim = null_space_dim

            self.eigen_vals = np.sort(gen_eigenvalues(min_ev=min_ev, max_ev=max_ev, dim=dim, null_space_dim=null_space_dim))

            self.min_ev = self.eigen_vals[null_space_dim]
            self.max_ev = self.eigen_vals[-1]

            jordan_form = np.diag(self.eigen_vals)

            eigen_vecs = gen_eigenvectors(dim=dim)
            inv_eigen_vec = np.linalg.inv(eigen_vecs)

            self.A = eigen_vecs @ jordan_form @ inv_eigen_vec
            self.b = self.A @ np.random.rand(dim)

            self.min_value = self.calculate_function(self._find_minimal_value())

        def convert_to_jax(self) -> None:
            self.A = jnp.array(self.A)
            self.b = jnp.array(self.b)

        def calculate_gradient(self, x: NDArray | jnp.ndarray) -> NDArray | jnp.ndarray:
            return self.A @ x - self.b

        def calculate_function(self, x: NDArray | jnp.ndarray) -> NDArray | jnp.ndarray:
            return 0.5 * x.T @ self.A @ x - self.b @ x

        def _find_minimal_value(self) -> float:
            x = jnp.linalg.solve(self.A, self.b)
            return x
    return (QuadraticForm,)


@app.cell
def _(mo):
    min_ev = mo.ui.number(start=0.01, stop=10000, step=0.1)
    max_ev = mo.ui.number(start=1, stop=10000, step=0.1)
    dim = mo.ui.number(start=0, stop=100, step=1)
    null_space_dim = mo.ui.number(start=0, stop=100, step=1)

    lower_dim = mo.ui.number(start=0, stop=5000, step=1)
    upper_dim = mo.ui.number(start=0, stop=5000, step=1)
    initial_point_samples_num = mo.ui.number(start=1, stop=100, step=1)
    gd_num_iterations = mo.ui.number(start=100, stop=3000, step=1)
    exp_name = mo.ui.text()
    dim_step = mo.ui.number()
    return (
        dim_step,
        gd_num_iterations,
        initial_point_samples_num,
        lower_dim,
        max_ev,
        min_ev,
        upper_dim,
    )


@app.cell
def _(QuadraticForm, jnp, np):
    def generate_quadratic_forms(
        min_ev: float,
        max_ev: float,
        lower_dim: int,
        upper_dim: int,
        dim_step: int,
        initial_point_samples_num: int,
        null_space_dims: tuple = (0, 5),
    ) -> tuple[list[QuadraticForm], list[jnp.array]]:
        q_forms = []
        initial_point_samples_per_form = []
        for dim in range(lower_dim, upper_dim + 1, dim_step):
            for null_space_dim in null_space_dims:
                q_form = QuadraticForm(
                    min_ev=min_ev,
                    max_ev=max_ev,
                    dim=dim,
                    null_space_dim=null_space_dim,
                )
                q_form.convert_to_jax()
                q_forms.append(q_form)

                initial_point_samples = [
                    q_form.A @ jnp.array(np.random.uniform(size=dim)) for _ in range(initial_point_samples_num)
                ]

                initial_point_samples_norms = [jnp.linalg.norm(pt) for pt in initial_point_samples]
                initial_point_samples = [
                    pt / norm * 10.0 if norm > 0 else pt for pt, norm in zip(initial_point_samples, initial_point_samples_norms)
                ]

                initial_point_samples_per_form.append(initial_point_samples)

        return q_forms, initial_point_samples_per_form
    return (generate_quadratic_forms,)


@app.cell
def _(Path, QuadraticForm, jax, jnp, mo, pl):
    def run_experiment_exact_ls(
        gd_num_iterations: int,
        experiment_name: str,
        quadratic_forms: list[QuadraticForm],
        initial_point_samples_per_form: list[jnp.array],
        device_num: int = 1,
    ) -> None:
        with jax.default_device(jax.devices("cpu")[device_num]):
            for f, initial_point_samples in mo.status.progress_bar(
                zip(quadratic_forms, initial_point_samples_per_form),
                title="Iterating over quadratic forms",
                show_eta=True,
                show_rate=True,
                total=len(quadratic_forms),
            ):
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
                }

                for init_point_idx, x in mo.status.progress_bar(
                    enumerate(initial_point_samples),
                    title="Initial point",
                    show_eta=True,
                    show_rate=True,
                    total=len(initial_point_samples),
                ):
                    with mo.status.spinner(subtitle="Descending...") as _spinner:
                        for i in range(gd_num_iterations):
                            grad_i = f.calculate_gradient(x)
                            lambda_i = (grad_i.T @ grad_i) / (grad_i.T @ f.A @ grad_i)

                            results["dimension"].append(f.dim)
                            results["kernel_size"].append(f.null_space_dim)
                            results["max_eigenvalue"].append(f.max_ev)
                            results["min_eigenvalue"].append(f.min_ev)
                            results["initial_point_index"].append(init_point_idx)
                            results["iteration"].append(i)
                            results["function_value"].append(f.calculate_function(x))

                            grad_norm = jnp.linalg.norm(grad_i)

                            results["gradient_norm"].append(grad_norm)
                            results["learning_rate"].append(lambda_i)

                            _spinner.update(
                                f"max_ev: {f.max_ev}, min_ev: {f.min_ev}, max_ev_inv: {1 / f.max_ev}, lr: {lambda_i}, gradient norm: {grad_norm}"
                            )

                            x -= lambda_i * grad_i

                results_table = pl.DataFrame(results)

                save_path = Path(f"./data/{experiment_name}")
                save_path.mkdir(parents=True, exist_ok=True)

                results_table.write_parquet(save_path / f"gd_results_{f.dim}_{f.null_space_dim}.parquet")
    return


@app.cell
def _(Path, QuadraticForm, jax, jnp, mo, np, pl):
    def run_experiment_adg(
        gd_num_iterations: int,
        experiment_name: str,
        quadratic_forms: list[QuadraticForm],
        initial_point_samples_per_form: list[jnp.array],
        device_num: int = 1,
    ) -> None:
        with jax.default_device(jax.devices("cpu")[device_num]):
            for f, initial_point_samples in mo.status.progress_bar(
                zip(quadratic_forms, initial_point_samples_per_form),
                title="Iterating over quadratic forms",
                show_eta=True,
                show_rate=True,
                total=len(quadratic_forms),
            ):
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
                    "left_choice_num": [],
                    "right_choice_num": [],
                    "loss": [],
                }

                for init_point_idx, x in mo.status.progress_bar(
                    enumerate(initial_point_samples),
                    title="Initial point",
                    show_eta=True,
                    show_rate=True,
                    total=len(initial_point_samples),
                ):
                    with mo.status.spinner(subtitle="Descending...") as _spinner:
                        lambda_prev = 1e-10
                        lambda_i = 1e-10
                        theta_i = np.inf
                        x_prev = x.copy()
                        grad_prev = f.calculate_gradient(x)

                        left_choice_num = 0
                        right_choice_num = 0

                        x -= lambda_i * grad_prev

                        for i in range(gd_num_iterations):
                            grad_i = f.calculate_gradient(x)

                            left_option = jnp.sqrt(1 + theta_i) * lambda_prev
                            right_option = jnp.linalg.norm(x - x_prev) / (2 * jnp.linalg.norm(grad_prev - grad_i))

                            if left_option < right_option:
                                lambda_i = left_option
                                left_choice_num += 1

                            else:
                                lambda_i = right_option
                                right_choice_num += 1

                            x_prev = x
                            grad_prev = grad_i
                            lambda_prev = lambda_i
                            theta_i = lambda_i / lambda_prev

                            x -= lambda_i * grad_i

                            results["dimension"].append(f.dim)
                            results["kernel_size"].append(f.null_space_dim)
                            results["max_eigenvalue"].append(f.max_ev)
                            results["min_eigenvalue"].append(f.min_ev)
                            results["initial_point_index"].append(init_point_idx)
                            results["iteration"].append(i)
                            results["function_value"].append(f.calculate_function(x))

                            grad_norm = jnp.linalg.norm(grad_i)

                            results["left_choice_num"].append(left_choice_num)
                            results["right_choice_num"].append(right_choice_num)

                            results["gradient_norm"].append(grad_norm)
                            results["learning_rate"].append(lambda_i)

                            results["loss"].append(f.calculate_function(x) - f.min_value)

                            _spinner.update(
                                f"max_ev: {f.max_ev}, min_ev: {f.min_ev}, max_ev_inv: {1 / f.max_ev}, lr: {lambda_i}, gradient norm: {grad_norm}"
                            )

                results_table = pl.DataFrame(results)

                save_path = Path(f"./data/{experiment_name}")
                save_path.mkdir(parents=True, exist_ok=True)

                results_table.write_parquet(save_path / f"gd_results_{f.dim}_{f.null_space_dim}.parquet")
    return (run_experiment_adg,)


@app.cell
def _(Path, QuadraticForm, jax, jnp, mo, np, pl):
    def run_experiment_adg_quadratic(
        gd_num_iterations: int,
        experiment_name: str,
        quadratic_forms: list[QuadraticForm],
        initial_point_samples_per_form: list[jnp.array],
        device_num: int = 1,
    ) -> None:
        with jax.default_device(jax.devices("cpu")[device_num]):
            for f, initial_point_samples in mo.status.progress_bar(
                zip(quadratic_forms, initial_point_samples_per_form),
                title="Iterating over quadratic forms",
                show_eta=True,
                show_rate=True,
                total=len(quadratic_forms),
            ):
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
                    "left_choice_num": [],
                    "right_choice_num": [],
                    "loss": [],
                }

                for init_point_idx, x in mo.status.progress_bar(
                    enumerate(initial_point_samples),
                    title="Initial point",
                    show_eta=True,
                    show_rate=True,
                    total=len(initial_point_samples),
                ):
                    with mo.status.spinner(subtitle="Descending...") as _spinner:
                        lambda_prev = 1e-10
                        lambda_i = 1e-10
                        theta_i = np.inf
                        x_prev = x.copy()
                        grad_prev = f.calculate_gradient(x)

                        left_choice_num = 0
                        right_choice_num = 0

                        x -= lambda_i * grad_prev

                        for i in range(gd_num_iterations):
                            grad_i = f.calculate_gradient(x)

                            left_option = jnp.sqrt(1 + theta_i) * lambda_prev
                            right_option = jnp.linalg.norm(x - x_prev) / (jnp.linalg.norm(grad_prev - grad_i))

                            if left_option < right_option:
                                lambda_i = left_option
                                left_choice_num += 1

                            else:
                                lambda_i = right_option
                                right_choice_num += 1

                            x_prev = x
                            grad_prev = grad_i
                            lambda_prev = lambda_i
                            theta_i = lambda_i / lambda_prev

                            x -= lambda_i * grad_i

                            results["dimension"].append(f.dim)
                            results["kernel_size"].append(f.null_space_dim)
                            results["max_eigenvalue"].append(f.max_ev)
                            results["min_eigenvalue"].append(f.min_ev)
                            results["initial_point_index"].append(init_point_idx)
                            results["iteration"].append(i)
                            results["function_value"].append(f.calculate_function(x))

                            grad_norm = jnp.linalg.norm(grad_i)

                            results["gradient_norm"].append(grad_norm)
                            results["learning_rate"].append(lambda_i)

                            results["left_choice_num"].append(left_choice_num)
                            results["right_choice_num"].append(right_choice_num)

                            results["loss"].append(f.calculate_function(x) - f.min_value)

                            _spinner.update(
                                f"max_ev: {f.max_ev}, min_ev: {f.min_ev}, max_ev_inv: {1 / f.max_ev}, lr: {lambda_i}, gradient norm: {grad_norm}"
                            )

                results_table = pl.DataFrame(results)

                save_path = Path(f"./data/{experiment_name}")
                save_path.mkdir(parents=True, exist_ok=True)

                results_table.write_parquet(save_path / f"gd_results_{f.dim}_{f.null_space_dim}.parquet")
    return (run_experiment_adg_quadratic,)


@app.cell
def _(
    dim_step,
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
            mo.md(f"Dimension step: {dim_step}"),
        ]
    )
    return


@app.cell
def _(
    dim_step,
    generate_quadratic_forms,
    initial_point_samples_num,
    lower_dim,
    max_ev,
    min_ev,
    upper_dim,
):
    quadratic_forms, initial_points = generate_quadratic_forms(
        min_ev=min_ev.value,
        max_ev=max_ev.value,
        lower_dim=lower_dim.value,
        upper_dim=upper_dim.value,
        dim_step=dim_step.value,
        initial_point_samples_num=initial_point_samples_num.value,
        null_space_dims=(0, 5, 10, 15),
    )
    return initial_points, quadratic_forms


@app.cell
def _(gd_num_iterations, initial_points, quadratic_forms, run_experiment_adg):
    run_experiment_adg(
        gd_num_iterations=gd_num_iterations.value,
        experiment_name="standard_adg_smaller",
        quadratic_forms=quadratic_forms,
        initial_point_samples_per_form=initial_points,
        device_num=0,
    )
    return


@app.cell
def _(
    gd_num_iterations,
    initial_points,
    quadratic_forms,
    run_experiment_adg_quadratic,
):
    run_experiment_adg_quadratic(
        gd_num_iterations=gd_num_iterations.value,
        experiment_name="adg_quadratic_improved_smaller",
        quadratic_forms=quadratic_forms,
        initial_point_samples_per_form=initial_points,
        device_num=0,
    )
    return


@app.cell
def _(pl):
    def preprocess_data(data: pl.DataFrame) -> pl.DataFrame:
        return data.with_columns(
            pl.lit(2).truediv(pl.col("max_eigenvalue").add(pl.col("min_eigenvalue"))).alias("best_lr"),
            pl.lit(1).truediv(pl.col("max_eigenvalue").add(pl.col("min_eigenvalue"))).alias("half_best_lr"),
            pl.lit(1).truediv(pl.col("max_eigenvalue")).alias("max_ev_inv"),
            pl.lit(1).truediv(pl.col("min_eigenvalue")).alias("min_ev_inv"),
        ).with_columns(
            pl.col("learning_rate").sub(pl.col("best_lr")).abs().alias("lr_error"),
            harmonic_mean_pair=(2 / (1 / pl.col("learning_rate")).sum()).over(pl.int_range(0, pl.len()) // 2),
        )
    return (preprocess_data,)


@app.cell
def _(Path):
    data_path = Path("data/")
    experiments_data_dirs = [exp_dir for exp_dir in data_path.iterdir() if exp_dir.is_dir()]
    return (experiments_data_dirs,)


@app.cell
def _(experiments_data_dirs, mo):
    left_plot_data = mo.ui.dropdown(options=experiments_data_dirs, label="Left data to plot")
    right_plot_data = mo.ui.dropdown(options=experiments_data_dirs, label="Left data to plot")
    return left_plot_data, right_plot_data


@app.cell
def _(left_plot_data, mo, right_plot_data):
    mo.vstack(
        [
            mo.hstack([left_plot_data, right_plot_data]),
        ]
    )
    return


@app.cell
def _(left_plot_data, pl, preprocess_data, right_plot_data):
    left_data = preprocess_data(pl.read_parquet(list(left_plot_data.value.rglob("*.parquet"))))
    right_data = preprocess_data(pl.read_parquet(list(right_plot_data.value.rglob("*.parquet"))))
    return left_data, right_data


@app.cell
def _(left_data, pl, right_data, wilcoxon):
    hypotheses_test: pl.DataFrame = left_data.join(
        right_data.select("dimension", "kernel_size", "initial_point_index", "iteration", pl.col("loss").alias("lossRight")), on=("dimension", "kernel_size", "initial_point_index", "iteration"), how="left"
    ).group_by("dimension", "kernel_size", "initial_point_index").agg("iteration","loss", "lossRight").with_columns(
       pl.struct(["loss", "lossRight"]).map_elements(
            lambda x: wilcoxon(x["loss"], x["lossRight"], alternative="greater").pvalue,
            return_dtype=pl.Float64
        ).alias("p_value") 
    )
    return (hypotheses_test,)


@app.cell
def _(hypotheses_test: "pl.DataFrame", mo, pl):
    mo.ui.dataframe(hypotheses_test.filter(pl.col("p_value").ge(0.05)))
    return


@app.cell
def _(hypotheses_test: "pl.DataFrame", pl):
    conv_speed_hypotheses_test = hypotheses_test.explode("iteration", "loss", "lossRight").filter(pl.col("iteration").le(100)).group_by("dimension", "kernel_size", "initial_point_index").agg(
        pl.col("loss").truediv(pl.col("lossRight")).mean().alias("conv_speed_ratio"),
    ).filter(pl.col("conv_speed_ratio").le(1.5))
    return (conv_speed_hypotheses_test,)


@app.cell
def _(conv_speed_hypotheses_test, mo):
    mo.ui.dataframe(conv_speed_hypotheses_test)
    return


@app.cell
def _(left_data, mo):
    dim_to_plot = mo.ui.dropdown(options=left_data["dimension"].unique())
    ker_to_plot = mo.ui.dropdown(options=left_data["kernel_size"].unique())
    init_point_to_plot = mo.ui.dropdown(options=left_data["initial_point_index"].unique())
    what_to_plot = mo.ui.multiselect(
        options=[
            "gradient_norm",
            "learning_rate",
            "best_lr",
            "lr_error",
            "harmonic_mean_pair",
            "min_ev_inv",
            "max_ev_inv",
            "left_choice_num",
            "right_choice_num",
            "loss",
        ]
    )
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
    conv_speed_hypotheses_test,
    dim_to_plot,
    go,
    init_point_to_plot,
    ker_to_plot,
    left_data,
    make_subplots,
    mo,
    pl,
    px,
    right_data,
    what_to_plot,
):
    left_chosen_data = left_data.filter(
        pl.col("dimension").eq(dim_to_plot.value)
        & pl.col("kernel_size").eq(ker_to_plot.value)
        & pl.col("initial_point_index").eq(init_point_to_plot.value)
    )
    right_chosen_data = right_data.filter(
        pl.col("dimension").eq(dim_to_plot.value)
        & pl.col("kernel_size").eq(ker_to_plot.value)
        & pl.col("initial_point_index").eq(init_point_to_plot.value)
    )


    fig = make_subplots(rows=1, cols=2, shared_xaxes=True, shared_yaxes=True)
    colors = [
        "#FDFD96",
        "#FFD1DC",
    ]

    for i, y_axis in enumerate(what_to_plot.value):
        fig.add_trace(
            go.Scatter(x=left_chosen_data["iteration"], y=left_chosen_data[y_axis], mode="markers", name=y_axis), 1, 1
        )
        fig.add_trace(
            go.Scatter(x=right_chosen_data["iteration"], y=right_chosen_data[y_axis], mode="markers", name=y_axis), 1, 2
        )

    fig.update_xaxes(title_text="Iteration Number")
    fig.update_yaxes(title_text="Value (Inverse Eigenvalue / Learning Rate)")
    fig.update_layout()


    joined_left_right = conv_speed_hypotheses_test.drop_nulls().filter(
        pl.col("dimension").eq(dim_to_plot.value)
        & pl.col("kernel_size").eq(ker_to_plot.value)
        & pl.col("initial_point_index").eq(init_point_to_plot.value)
    )

    fig_diff = px.scatter(joined_left_right, x="iteration", y = "conv_speed_ratio")


    gd_results_plot = mo.ui.plotly(fig)
    fig_diff_plot = mo.ui.plotly(fig_diff)
    return fig_diff_plot, gd_results_plot


@app.cell
def _(fig_diff_plot, gd_results_plot, mo, plot_constructor):
    mo.vstack([plot_constructor, gd_results_plot, fig_diff_plot])
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
