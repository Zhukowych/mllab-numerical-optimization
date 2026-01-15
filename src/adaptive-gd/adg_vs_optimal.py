import marimo

__generated_with = "0.18.1"
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
    def gen_eigenvalues(dim: int = 3, null_space_dim: int = 0, dist_method=rng.uniform, **kwargs) -> NDArray[np.float64]:
        non_zero_evs = dist_method(size=dim - null_space_dim, **kwargs)
        return np.hstack((np.zeros(null_space_dim), non_zero_evs))


    def gen_eigenvectors(dim: int = 3) -> NDarray[np.float64]:
        return ortho_group.rvs(dim=dim, random_state=rng)
    return (gen_eigenvalues,)


@app.cell
def _(NDArray, Path, gen_eigenvalues, jnp, np, rng):
    class QuadraticForm:
        def __init__(
            self,
            dim: int = 3,
            null_space_dim: int = 0,
            dist_method=rng.uniform,
            **kwargs,
        ) -> None:
            self.dim = dim
            self.null_space_dim = null_space_dim

            self.eigen_vals = np.sort(
                gen_eigenvalues(dim=dim, null_space_dim=null_space_dim, dist_method=dist_method, **kwargs)
            )

            self.min_ev = self.eigen_vals[null_space_dim]
            self.max_ev = self.eigen_vals[-1]

            jordan_form = np.diag(self.eigen_vals)

            self.A = jordan_form

            self.min_value = 0

        def convert_to_jax(self) -> None:
            self.A = jnp.array(self.A)

        def calculate_gradient(self, x: NDArray | jnp.ndarray) -> NDArray | jnp.ndarray:
            return self.A @ x

        def calculate_function(self, x: NDArray | jnp.ndarray) -> NDArray | jnp.ndarray:
            return 0.5 * x.T @ self.A @ x

        def save(self, save_path: Path) -> None:
            save_dir = save_path / f"form_{self.dim}_{self.null_space_dim}"
            save_dir.mkdir(exist_ok=True, parents=True)
            jnp.save(save_dir / "quadratic_form.npy", self.eigen_vals)
    return (QuadraticForm,)


@app.cell
def _(mo):
    min_ev = mo.ui.number(start=0.01, stop=10000, step=0.1)
    max_ev = mo.ui.number(start=1, stop=10_000, step=0.1, value=10_000)

    lower_dim = mo.ui.number(start=0, stop=5000, step=1, value=30)
    upper_dim = mo.ui.number(start=0, stop=5000, step=1, value=100)
    initial_point_samples_num = mo.ui.number(start=1, stop=100, step=1, value=10)
    gd_num_iterations = mo.ui.number(start=100, stop=3000, step=1, value=300)
    exp_name = mo.ui.text()
    dim_step = mo.ui.number(value=5)
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
def _(Path, QuadraticForm, jnp, np, rng):
    def generate_quadratic_forms(
        lower_dim: int,
        upper_dim: int,
        dim_step: int,
        initial_point_samples_num: int,
        save_path: Path,
        dist_method=rng.uniform,
        null_space_dims: tuple = (0, 5),
        **kwargs,
    ) -> tuple[list[QuadraticForm], list[jnp.array]]:
        q_forms = []
        initial_point_samples_per_form = []

        for dim in range(lower_dim, upper_dim + 1, dim_step):
            for null_space_dim in null_space_dims:
                q_form = QuadraticForm(dim=dim, null_space_dim=null_space_dim, dist_method=dist_method, **kwargs)
                q_form.save(save_path)
                q_form.convert_to_jax()
                q_forms.append(q_form)

                initial_point_samples = [
                    q_form.A @ jnp.array(np.random.uniform(size=dim)) for _ in range(initial_point_samples_num)
                ]

                initial_point_samples_norms = [jnp.linalg.norm(pt) for pt in initial_point_samples]
                initial_point_samples = [
                    pt / norm * 1_000.0 if norm > 0 else pt
                    for pt, norm in zip(initial_point_samples, initial_point_samples_norms)
                ]

                initial_point_samples_per_form.append(initial_point_samples)

        return q_forms, initial_point_samples_per_form
    return (generate_quadratic_forms,)


@app.cell
def _(Path, QuadraticForm, jax, jnp, mo, np, pl):
    def run_experiment_adg(
        gd_num_iterations: int,
        experiment_name: str,
        save_path: Path,
        quadratic_forms: list[QuadraticForm],
        initial_point_samples_per_form: list[jnp.array],
        beta: float = 0.5,
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
                            right_option = beta * jnp.linalg.norm(x - x_prev) / (jnp.linalg.norm(grad_prev - grad_i))

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

                save_path_exp = save_path / f"form_{f.dim}_{f.null_space_dim}"

                results_table.write_parquet(save_path_exp / f"{experiment_name}.parquet")
    return (run_experiment_adg,)


@app.cell
def _(Path, QuadraticForm, jax, jnp, mo, pl):
    def run_experiment_optimal_step(
        gd_num_iterations: int,
        experiment_name: str,
        save_path: Path,
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
                        lr = 1 / f.max_ev

                        for i in range(gd_num_iterations):
                            grad_i = f.calculate_gradient(x)

                            x -= lr * grad_i

                            results["dimension"].append(f.dim)
                            results["kernel_size"].append(f.null_space_dim)
                            results["max_eigenvalue"].append(f.max_ev)
                            results["min_eigenvalue"].append(f.min_ev)
                            results["initial_point_index"].append(init_point_idx)
                            results["iteration"].append(i)
                            results["function_value"].append(f.calculate_function(x))

                            grad_norm = jnp.linalg.norm(grad_i)

                            results["gradient_norm"].append(grad_norm)
                            results["learning_rate"].append(lr)

                            results["loss"].append(f.calculate_function(x) - f.min_value)

                            _spinner.update(
                                f"max_ev: {f.max_ev}, min_ev: {f.min_ev}, max_ev_inv: {1 / f.max_ev}, lr: {lr}, gradient norm: {grad_norm}"
                            )

                results_table = pl.DataFrame(results)

                save_path_exp = save_path / f"form_{f.dim}_{f.null_space_dim}"

                results_table.write_parquet(save_path_exp / f"{experiment_name}.parquet")
    return (run_experiment_optimal_step,)


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
def _():
    general_experiment_path = "data/adg_vs_optimal"
    return (general_experiment_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Uniform EV distribution
    """)
    return


@app.cell
def _(general_experiment_path, mo):
    save_path_uniform = mo.ui.text(value=f"{general_experiment_path}/uniform/", full_width=True)
    mo.md(f"Save path for this experiment: {save_path_uniform}")
    return (save_path_uniform,)


@app.cell
def _(
    Path,
    dim_step,
    generate_quadratic_forms,
    initial_point_samples_num,
    lower_dim,
    max_ev,
    min_ev,
    rng,
    save_path_uniform,
    upper_dim,
):
    quadratic_forms, initial_points = generate_quadratic_forms(
        lower_dim=lower_dim.value,
        upper_dim=upper_dim.value,
        dim_step=dim_step.value,
        initial_point_samples_num=initial_point_samples_num.value,
        save_path=Path(save_path_uniform.value),
        dist_method=rng.uniform,
        null_space_dims=(0, 5, 10, 15),
        low=min_ev.value,
        high=max_ev.value,
    )
    return initial_points, quadratic_forms


@app.cell
def _(
    Path,
    gd_num_iterations,
    initial_points,
    quadratic_forms,
    run_experiment_adg,
    save_path_uniform,
):
    run_experiment_adg(
        gd_num_iterations=gd_num_iterations.value,
        experiment_name="adg_quadratic_quadratic_3%4",
        save_path=Path(save_path_uniform.value),
        quadratic_forms=quadratic_forms,
        initial_point_samples_per_form=initial_points,
        beta=0.75,
        device_num=0,
    )
    return


@app.cell
def _(
    Path,
    gd_num_iterations,
    initial_points,
    quadratic_forms,
    run_experiment_optimal_step,
    save_path_uniform,
):
    run_experiment_optimal_step(
        gd_num_iterations=gd_num_iterations.value,
        experiment_name="gd_optimal_step",
        save_path=Path(save_path_uniform.value),
        quadratic_forms=quadratic_forms,
        initial_point_samples_per_form=initial_points,
        device_num=0,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### $B(\alpha = 1, \beta = 20)$ distribution
    """)
    return


@app.cell
def _(general_experiment_path, mo):
    save_path_beta_1_20 = mo.ui.text(value=f"{general_experiment_path}/beta_1_20/", full_width=True)
    mo.md(f"Save path for this experiment: {save_path_beta_1_20}")
    return (save_path_beta_1_20,)


@app.cell
def _(
    Path,
    dim_step,
    generate_quadratic_forms,
    initial_point_samples_num,
    lower_dim,
    rng,
    save_path_beta_1_20,
    upper_dim,
):
    quadratic_forms_beta_1_20, initial_points_beta_1_20 = generate_quadratic_forms(
        lower_dim=lower_dim.value,
        upper_dim=upper_dim.value,
        dim_step=dim_step.value,
        initial_point_samples_num=initial_point_samples_num.value,
        save_path=Path(save_path_beta_1_20.value),
        dist_method=rng.beta,
        null_space_dims=(0, 5, 10, 15),
        a=1,
        b=20,
    )
    return initial_points_beta_1_20, quadratic_forms_beta_1_20


@app.cell
def _(
    Path,
    gd_num_iterations,
    initial_points_beta_1_20,
    quadratic_forms_beta_1_20,
    run_experiment_adg,
    save_path_beta_1_20,
):
    run_experiment_adg(
        gd_num_iterations=gd_num_iterations.value,
        experiment_name="adg_quadratic_quadratic_3%4",
        save_path=Path(save_path_beta_1_20.value),
        quadratic_forms=quadratic_forms_beta_1_20,
        initial_point_samples_per_form=initial_points_beta_1_20,
        beta=0.75,
        device_num=0,
    )
    return


@app.cell
def _(
    Path,
    gd_num_iterations,
    initial_points_beta_1_20,
    quadratic_forms_beta_1_20,
    run_experiment_optimal_step,
    save_path_beta_1_20,
):
    run_experiment_optimal_step(
        gd_num_iterations=gd_num_iterations.value,
        experiment_name="gd_optimal_step",
        save_path=Path(save_path_beta_1_20.value),
        quadratic_forms=quadratic_forms_beta_1_20,
        initial_point_samples_per_form=initial_points_beta_1_20,
        device_num=0,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data reading
    """)
    return


@app.cell
def _(pl):
    def preprocess_data(data: pl.DataFrame) -> pl.DataFrame:
        return data.with_columns(
            pl.lit(1).truediv(pl.col("max_eigenvalue")).alias("max_ev_inv"),
            pl.lit(1).truediv(pl.col("min_eigenvalue")).alias("min_ev_inv"),
        )
    return (preprocess_data,)


@app.cell
def _(Path, general_experiment_path):
    data_path = Path(general_experiment_path)
    experiments_data_dirs = {exp_dir.stem: exp_dir for exp_dir in data_path.iterdir() if exp_dir.is_dir()}
    return (experiments_data_dirs,)


@app.cell
def _(experiments_data_dirs, mo):
    distribution_experiment = mo.ui.dropdown(options=experiments_data_dirs, label="Distribution experiment")
    distribution_experiment
    return (distribution_experiment,)


@app.cell
def _(distribution_experiment, pl, preprocess_data):
    adg_data = preprocess_data(pl.read_parquet(list(distribution_experiment.value.rglob("*adg*.parquet"))))
    optimal_step_data = preprocess_data(pl.read_parquet(list(distribution_experiment.value.rglob("*optimal*.parquet"))))
    return adg_data, optimal_step_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Basic hypothesis test
    """)
    return


@app.cell
def _(adg_data, optimal_step_data, pl, wilcoxon):
    hypotheses_test: pl.DataFrame = (
        adg_data.join(
            optimal_step_data.select(
                "dimension", "kernel_size", "initial_point_index", "iteration", pl.col("loss").alias("lossRight")
            ),
            on=("dimension", "kernel_size", "initial_point_index", "iteration"),
            how="left",
        )
        .group_by("dimension", "kernel_size", "initial_point_index")
        .agg("iteration", "loss", "lossRight")
        .with_columns(
            pl.struct(["loss", "lossRight"])
            .map_elements(lambda x: wilcoxon(x["loss"], x["lossRight"]).pvalue, return_dtype=pl.Float64)
            .alias("p_value")
        )
    )
    return (hypotheses_test,)


@app.cell
def _(hypotheses_test: "pl.DataFrame", mo, pl):
    mo.ui.dataframe(hypotheses_test.filter(pl.col("p_value").ge(0.05)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Plotting logic
    """)
    return


@app.cell
def _(adg_data, mo):
    dim_to_plot = mo.ui.dropdown(options=adg_data["dimension"].unique())
    ker_to_plot = mo.ui.dropdown(options=adg_data["kernel_size"].unique())
    init_point_to_plot = mo.ui.dropdown(options=adg_data["initial_point_index"].unique())
    what_to_plot = mo.ui.multiselect(
        options=[
            "gradient_norm",
            "learning_rate",
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
    adg_data,
    dim_to_plot,
    distribution_experiment,
    go,
    init_point_to_plot,
    jnp,
    ker_to_plot,
    make_subplots,
    mo,
    optimal_step_data,
    pl,
    px,
    what_to_plot,
):
    left_chosen_data = adg_data.filter(
        pl.col("dimension").eq(dim_to_plot.value)
        & pl.col("kernel_size").eq(ker_to_plot.value)
        & pl.col("initial_point_index").eq(init_point_to_plot.value)
    )
    right_chosen_data = optimal_step_data.filter(
        pl.col("dimension").eq(dim_to_plot.value)
        & pl.col("kernel_size").eq(ker_to_plot.value)
        & pl.col("initial_point_index").eq(init_point_to_plot.value)
    )


    fig = make_subplots(rows=1, cols=2, shared_yaxes=True)
    colors = [
        "#FDFD96",
        "#FFD1DC",
    ]

    for i, y_axis in enumerate(what_to_plot.value):
        fig.add_trace(
            go.Scatter(x=left_chosen_data["iteration"], y=left_chosen_data[y_axis], mode="markers", name=f"{y_axis}, adg"), 1, 1
        )
        fig.add_trace(
            go.Scatter(x=right_chosen_data["iteration"], y=right_chosen_data[y_axis], mode="markers", name=f"{y_axis}, optimal step"), 1, 2
        )

    fig.update_xaxes(title_text="Iteration (adg)", row=1, col=1)
    fig.update_xaxes(title_text="Iteration (optimal step)", row=1, col=2)
    fig.update_xaxes(matches="x")
    fig.update_yaxes(title_text=f"Value ({' / '.join(what_to_plot.value)})")
    fig.update_layout()

    eigen_vals = jnp.load(
        distribution_experiment.value / f"form_{dim_to_plot.value}_{ker_to_plot.value}/quadratic_form.npy"
    ) if (dim_to_plot.value is not None and ker_to_plot.value is not None) else [0]
    eigen_vals_plot = mo.ui.plotly(
        px.scatter(x=list(eigen_vals), y=list(eigen_vals))
    )

    gd_results_plot = mo.ui.plotly(fig)
    return eigen_vals_plot, gd_results_plot


@app.cell
def _(eigen_vals_plot, gd_results_plot, mo, plot_constructor):
    mo.vstack([plot_constructor, gd_results_plot, eigen_vals_plot])
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
