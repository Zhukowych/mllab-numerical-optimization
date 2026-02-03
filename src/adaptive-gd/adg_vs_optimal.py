import marimo

__generated_with = "0.19.4"
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
    import polars.selectors as cs
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


    def gen_eigenvectors(dim: int = 3) -> NDArray[np.float64]:
        return ortho_group.rvs(dim=dim, random_state=rng)
    return (gen_eigenvalues,)


@app.cell
def _(NDArray, Path, gen_eigenvalues, jnp, np, rng):
    class QuadraticForm:
        def __init__(
            self,
            dim: int = 3,
            null_space_dim: int = 0,
            index: int = 0,
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
            self.index = index

            self.min_value = 0

        def convert_to_jax(self) -> None:
            self.A = jnp.array(self.A)

        def calculate_gradient(self, x: NDArray | jnp.ndarray) -> NDArray | jnp.ndarray:
            return self.A @ x

        def calculate_function(self, x: NDArray | jnp.ndarray) -> NDArray | jnp.ndarray:
            return 0.5 * x.T @ self.A @ x

        def save(self, save_path: Path) -> None:
            save_dir = save_path / f"form_{self.dim}_{self.null_space_dim}_idx{self.index}"
            save_dir.mkdir(exist_ok=True, parents=True)
            jnp.save(save_dir / "quadratic_form.npy", self.eigen_vals)
    return (QuadraticForm,)


@app.cell
def _(mo):
    min_ev = mo.ui.number(start=0.01, stop=10000, step=0.1)
    max_ev = mo.ui.number(start=1, stop=1_000, step=0.1, value=1_000)

    lower_dim = mo.ui.number(start=0, stop=5000, step=1, value=30)
    upper_dim = mo.ui.number(start=0, stop=5000, step=1, value=100)
    initial_point_samples_num = mo.ui.number(start=1, stop=100, step=1, value=10)
    gd_num_iterations = mo.ui.number(start=100, stop=3000, step=1, value=500)
    exp_name = mo.ui.text()
    dim_step = mo.ui.number(value=5)

    convergence_threshold = mo.ui.number(value=1e-4, label="Convergence threshold")
    forms_per_size = mo.ui.number(value=5, label="Number of forms per size")
    return (
        convergence_threshold,
        dim_step,
        forms_per_size,
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
        forms_per_size: int,
        dist_method=rng.uniform,
        null_space_dims: tuple = (0, 5),
        **kwargs,
    ) -> tuple[list[QuadraticForm], list[jnp.array]]:
        q_forms = []
        initial_point_samples_per_form = []

        for dim in range(lower_dim, upper_dim + 1, dim_step):
            for null_space_dim in null_space_dims:
                initial_point_samples = [
                        jnp.array(np.hstack((np.zeros(null_space_dim), np.random.uniform(size=dim-null_space_dim)))) for _ in range(initial_point_samples_num)
                    ]

                initial_point_samples_norms = [jnp.linalg.norm(pt) for pt in initial_point_samples]
                initial_point_samples = [
                    (pt / norm) if norm > 0 else pt
                    for pt, norm in zip(initial_point_samples, initial_point_samples_norms)
                ]
                for form_idx in range(forms_per_size):
                    q_form = QuadraticForm(dim=dim, null_space_dim=null_space_dim, dist_method=dist_method, index=form_idx,**kwargs)
                    q_form.save(save_path)
                    q_form.convert_to_jax()
                    q_forms.append(q_form)

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
        convergence_threshold: float,
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
                    "gradient_norm": [],
                    "learning_rate": [],
                    "max_eigenvalue": [],
                    "min_eigenvalue": [],
                    "left_choice_num": [],
                    "right_choice_num": [],
                    "loss": [],
                    "index": [],
                }

                for init_point_idx, x in mo.status.progress_bar(
                    enumerate(initial_point_samples),
                    title="Initial point",
                    show_eta=True,
                    show_rate=True,
                    total=len(initial_point_samples),
                    remove_on_exit=True,
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

                            grad_norm = jnp.linalg.norm(grad_i)

                            results["left_choice_num"].append(left_choice_num)
                            results["right_choice_num"].append(right_choice_num)

                            results["gradient_norm"].append(grad_norm)
                            results["learning_rate"].append(lambda_i)

                            function_value = f.calculate_function(x)
                            results["loss"].append(function_value)
                            results["index"].append(f.index)

                            if function_value < convergence_threshold:
                                break

                            _spinner.update(
                                f"max_ev: {f.max_ev}, min_ev: {f.min_ev}, max_ev_inv: {1 / f.max_ev}, lr: {lambda_i}, gradient norm: {grad_norm}"
                            )

                results_table = pl.DataFrame(results)

                save_path_exp = save_path / f"form_{f.dim}_{f.null_space_dim}_idx{f.index}"

                results_table.write_parquet(save_path_exp / f"{experiment_name}.parquet")
    return (run_experiment_adg,)


@app.cell
def _(Path, QuadraticForm, jax, jnp, mo, pl):
    def run_experiment_optimal_step(
        gd_num_iterations: int,
        experiment_name: str,
        convergence_threshold: float,
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
                    "gradient_norm": [],
                    "learning_rate": [],
                    "max_eigenvalue": [],
                    "min_eigenvalue": [],
                    "loss": [],
                    "index": [],
                }

                for init_point_idx, x in mo.status.progress_bar(
                    enumerate(initial_point_samples),
                    title="Initial point",
                    show_eta=True,
                    show_rate=True,
                    total=len(initial_point_samples),
                    remove_on_exit=True,
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

                            grad_norm = jnp.linalg.norm(grad_i)

                            results["gradient_norm"].append(grad_norm)
                            results["learning_rate"].append(lr)

                            function_value = f.calculate_function(x)
                            results["loss"].append(function_value)
                            results["index"].append(f.index)

                            _spinner.update(
                                f"max_ev: {f.max_ev}, min_ev: {f.min_ev}, max_ev_inv: {1 / f.max_ev}, lr: {lr}, gradient norm: {grad_norm}"
                            )

                            if function_value < convergence_threshold:
                                break

                results_table = pl.DataFrame(results)

                save_path_exp = save_path / f"form_{f.dim}_{f.null_space_dim}_idx{f.index}"

                results_table.write_parquet(save_path_exp / f"{experiment_name}.parquet")
    return (run_experiment_optimal_step,)


@app.cell
def _(
    convergence_threshold,
    dim_step,
    forms_per_size,
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
            mo.md(f"{forms_per_size}"),
            mo.md(f"{convergence_threshold}"),
            mo.md(f"Dimension step: {dim_step}"),
        ]
    )
    return


@app.cell
def _(Path):
    general_experiment_path = Path("data/adg_vs_optimal")
    return (general_experiment_path,)


@app.cell
def _(
    Path,
    dim_step,
    gd_num_iterations,
    general_experiment_path,
    generate_quadratic_forms,
    initial_point_samples_num,
    lower_dim,
    rng,
    run_experiment_adg,
    run_experiment_optimal_step,
    upper_dim,
):
    def run_experiment_adg_vs_optimal(
        exp_name: Path,
        forms_per_size: int,
        convergence_threshold: float,
        dist_method=rng.uniform,
        null_space_dims: tuple = (0, 5),
        device_num: int = 0,
        **kwargs,
    ):

        quadratic_forms, initial_points = generate_quadratic_forms(
            lower_dim=lower_dim.value,
            upper_dim=upper_dim.value,
            dim_step=dim_step.value,
            initial_point_samples_num=initial_point_samples_num.value,
            save_path=general_experiment_path / exp_name,
            dist_method=dist_method,
            null_space_dims=null_space_dims,
            forms_per_size=forms_per_size,
            **kwargs,
        )

        run_experiment_adg(
            gd_num_iterations=gd_num_iterations.value,
            experiment_name="adg_quadratic_quadratic_3%4",
            save_path=general_experiment_path / exp_name,
            quadratic_forms=quadratic_forms,
            initial_point_samples_per_form=initial_points,
            beta=0.75,
            convergence_threshold=convergence_threshold,
            device_num=device_num,
        )

        run_experiment_optimal_step(
            gd_num_iterations=gd_num_iterations.value,
            experiment_name="gd_optimal_step",
            save_path=general_experiment_path / exp_name,
            quadratic_forms=quadratic_forms,
            initial_point_samples_per_form=initial_points,
            convergence_threshold=convergence_threshold,
            device_num=device_num,
        )
    return (run_experiment_adg_vs_optimal,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Uniform EV distribution
    """)
    return


@app.cell
def _(
    convergence_threshold,
    forms_per_size,
    max_ev,
    min_ev,
    rng,
    run_experiment_adg_vs_optimal,
):
    run_experiment_adg_vs_optimal(
        exp_name="uniform",  # CHANGE THIS
        dist_method=rng.uniform,
        null_space_dims=(0, 5, 10, 15),
        device_num=0,
        convergence_threshold=convergence_threshold.value,
        forms_per_size=forms_per_size.value,
        # KWARGS
        low=min_ev.value,
        high=max_ev.value,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### $B(\alpha=2, \beta=100)$ distrubution scaled to $(\lambda_{\min}, \lambda_{\max})$. $\lambda_{\max}$ is forsed to be in the form.
    """)
    return


@app.cell
def _(jnp, max_ev, min_ev, rng, run_experiment_adg_vs_optimal):
    def scaling_and_forcing(a: float, b: float, size: int, max_ev: float, min_ev: float) -> jnp.array:
        initial_eigen_vals = jnp.sort(rng.beta(a=a, b=b, size=size))

        scaled_eigen_vals = min_ev + (max_ev - min_ev) * initial_eigen_vals
        scaled_eigen_vals = scaled_eigen_vals.at[-1].set(max_ev)
        scaled_eigen_vals = scaled_eigen_vals.at[0].set(min_ev)

        return scaled_eigen_vals


    run_experiment_adg_vs_optimal(
        exp_name="beta_2_100_scaled_forced",  # CHANGE THIS
        dist_method=scaling_and_forcing,
        null_space_dims=(0, 5, 10, 15),
        device_num=0,
        # KWARGS
        a=2,
        b=100,
        max_ev=max_ev.value,
        min_ev=min_ev.value,
    )
    return (scaling_and_forcing,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Duplicated eigenvalues
    """)
    return


@app.cell
def _(jnp, max_ev, min_ev, run_experiment_adg_vs_optimal):
    def generate_duplicates(size: int, max_ev: float, min_ev: float) -> jnp.array:
        return jnp.ones(size) * max_ev / 2


    run_experiment_adg_vs_optimal(
        exp_name="duplicated_evs",  # CHANGE THIS
        dist_method=generate_duplicates,
        null_space_dims=(0, 5, 10, 15),
        device_num=0,
        # KWARGS
        max_ev=max_ev.value,
        min_ev=min_ev.value,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### $\dim - 1$ of space is duplicated minimial eigenvalue, the other ev is max ev
    """)
    return


@app.cell
def _(jnp, max_ev, min_ev, run_experiment_adg_vs_optimal):
    def generate_only_one_max_ev(size: int, max_ev: float, min_ev: float) -> jnp.array:
        return jnp.array([min_ev for _ in range(size - 1)] + [max_ev])


    run_experiment_adg_vs_optimal(
        exp_name="only_one_max_ev",  # CHANGE THIS
        dist_method=generate_only_one_max_ev,
        null_space_dims=(0, 5, 10, 15),
        device_num=0,
        # KWARGS
        max_ev=max_ev.value,
        min_ev=min_ev.value,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### $\dim - 1$ of space is duplicated maximal eigenvalue, the other ev is min ev
    """)
    return


@app.cell
def _(jnp, max_ev, min_ev, run_experiment_adg_vs_optimal):
    def generate_only_one_min_ev(size: int, max_ev: float, min_ev: float) -> jnp.array:
        return jnp.array([max_ev for _ in range(size - 1)] + [min_ev])


    run_experiment_adg_vs_optimal(
        exp_name="only_one_min_ev",  # CHANGE THIS
        dist_method=generate_only_one_min_ev,
        null_space_dims=(0, 5, 10, 15),
        device_num=0,
        # KWARGS
        max_ev=max_ev.value,
        min_ev=min_ev.value,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### $\dim - 2$ of space is duplicated minimal eigenvalue, the other ev is max ev and $\lambda_{\max} - 1$
    """)
    return


@app.cell
def _(jnp, max_ev, min_ev, run_experiment_adg_vs_optimal):
    def generate_two_big_evs(size: int, max_ev: float, min_ev: float) -> jnp.array:
        return jnp.array([min_ev for _ in range(size - 2)] + [max_ev - 1, max_ev])


    run_experiment_adg_vs_optimal(
        exp_name="two_big_evs",  # CHANGE THIS
        dist_method=generate_two_big_evs,
        null_space_dims=(0, 5, 10, 15),
        device_num=0,
        # KWARGS
        max_ev=max_ev.value,
        min_ev=min_ev.value,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### $B(\alpha=100, \beta=2)$ distrubution scaled to $(\lambda_{\min}, \lambda_{\max})$. $\lambda_{\max}$ is forsed to be in the form.
    """)
    return


@app.cell
def _(max_ev, min_ev, run_experiment_adg_vs_optimal, scaling_and_forcing):
    run_experiment_adg_vs_optimal(
        exp_name="beta_100_2_scaled_forced",  # CHANGE THIS
        dist_method=scaling_and_forcing,
        null_space_dims=(0, 5, 10, 15),
        device_num=0,
        # KWARGS
        a=100,
        b=2,
        max_ev=max_ev.value,
        min_ev=min_ev.value,
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
            pl.col("loss").log().diff().name.suffix("_rate_of_change")
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
    mo.md(f"{distribution_experiment}")
    return (distribution_experiment,)


@app.cell
def _(distribution_experiment, pl, preprocess_data):
    adg_data = preprocess_data(pl.read_parquet(list(distribution_experiment.value.rglob("*adg*.parquet"))))
    optimal_step_data = preprocess_data(pl.read_parquet(list(distribution_experiment.value.rglob("*optimal*.parquet"))))
    return adg_data, optimal_step_data


@app.cell
def _(adg_data, convergence_threshold, mo, optimal_step_data, pl):
    adg_data_convergence = adg_data.group_by("dimension", "kernel_size", "initial_point_index").agg(pl.col("loss").min())
    optimal_step_data_convergence = optimal_step_data.group_by("dimension", "kernel_size", "initial_point_index").agg(
        pl.col("loss").min()
    )

    adg_data_convergence_mean = adg_data_convergence.select(pl.col("loss").mean())
    optimal_step_data_convergence_mean = optimal_step_data_convergence.select(pl.col("loss").mean())

    adg_data_convergence_failed = adg_data_convergence.filter(pl.col("loss").gt(convergence_threshold.value))
    optimal_step_data_convergence_failed = optimal_step_data_convergence.filter(
        pl.col("loss").gt(convergence_threshold.value)
    )

    mo.vstack(
        [
            mo.md(
                f"AGD mean loss: {adg_data_convergence_mean['loss'].item()}, Optimal step mean loss: {optimal_step_data_convergence_mean['loss'].item()}"
            ),
            mo.md(
                f"AGD number of failed to converge: {adg_data_convergence_failed.height}, Optimal step failed to converge: {optimal_step_data_convergence_failed.height}"
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Basic hypothesis test
    """)
    return


@app.cell
def _(adg_data, convergence_threshold, optimal_step_data, pl, wilcoxon):
    hypotheses_test: pl.DataFrame = (
        adg_data.join(
            optimal_step_data.select(
                "dimension",
                "kernel_size",
                "initial_point_index",
                "iteration",
                pl.col("loss").name.suffix("_optimal"),
                pl.col("learning_rate").name.suffix("_optimal"),
                pl.col("loss_rate_of_change").name.suffix("_optimal")
            ),
            on=("dimension", "kernel_size", "initial_point_index", "iteration"),
            how="left",
        )
        .filter(
            (pl.col("loss").ge(convergence_threshold.value) | pl.col("loss_optimal").ge(convergence_threshold.value))
            & (
                pl.col("loss").lt(
                    pl.col("loss")
                    .quantile(0.98, interpolation="linear")
                    .over("dimension", "kernel_size", "initial_point_index")
                )
            )
            & (
                pl.col("loss_optimal").lt(
                    pl.col("loss_optimal")
                    .quantile(0.98, interpolation="linear")
                    .over("dimension", "kernel_size", "initial_point_index")
                )
            )
        )
        .group_by("dimension", "kernel_size", "initial_point_index")
        .agg("iteration", "loss", "loss_optimal", "learning_rate", "learning_rate_optimal", "loss_rate_of_change", "loss_rate_of_change_optimal")
        .with_columns(
            pl.struct(["loss", "loss_optimal"])
            .map_elements(
                lambda x: wilcoxon(x["loss"], x["loss_optimal"], alternative="less").pvalue, return_dtype=pl.Float64
            )
            .alias("loss_p_value"),
            pl.struct(["learning_rate", "learning_rate_optimal"])
            .map_elements(
                lambda x: wilcoxon(x["learning_rate"], x["learning_rate_optimal"], alternative="greater").pvalue,
                return_dtype=pl.Float64,
            )
            .alias("lr_p_value"),
        )
    )
    return (hypotheses_test,)


@app.cell
def _(hypotheses_test: "pl.DataFrame", mo, pl):
    mo.md(f"""
    #Hypothesis of adg converging faster {mo.ui.dataframe(hypotheses_test.filter(pl.col("loss_p_value").gt(0.05)))}
    """)
    return


@app.cell
def _(hypotheses_test: "pl.DataFrame", mo, pl):
    mo.md(f"""
    #Hypothesis of adg having bigger step {mo.ui.dataframe(hypotheses_test.filter(pl.col("lr_p_value").gt(0.05)))}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Plotting logic
    """)
    return


@app.cell
def _(adg_data, mo, optimal_step_data):
    dim_to_plot = mo.ui.dropdown(options=adg_data["dimension"].unique())
    ker_to_plot = mo.ui.dropdown(options=adg_data["kernel_size"].unique())
    init_point_to_plot = mo.ui.dropdown(options=adg_data["initial_point_index"].unique())
    what_to_plot = mo.ui.multiselect(options=set(adg_data.columns) & set(optimal_step_data.columns))
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


    fig = make_subplots(rows=1, cols=1, shared_yaxes=True)
    colors = [
        "#FDFD96",
        "#FFD1DC",
    ]

    for i, y_axis in enumerate(what_to_plot.value):
        fig.add_trace(
            go.Line(x=left_chosen_data["iteration"], y=left_chosen_data[y_axis], mode="lines", name=f"{y_axis}, adg"),
            1,
            1,
        )
        fig.add_trace(
            go.Scatter(
                x=right_chosen_data["iteration"],
                y=right_chosen_data[y_axis],
                mode="lines",
                name=f"{y_axis}, optimal step",
            ),
            1,
            1,
        )

    fig.update_xaxes(title_text="Iteration (adg)", row=1, col=1)
    fig.update_xaxes(title_text="Iteration (optimal step)", row=1, col=2)
    fig.update_xaxes(matches="x")
    fig.update_yaxes(title_text=f"Value ({' / '.join(what_to_plot.value)})")
    fig.update_layout()

    eigen_vals = (
        jnp.load(distribution_experiment.value / f"form_{dim_to_plot.value}_{ker_to_plot.value}/quadratic_form.npy")
        if (dim_to_plot.value is not None and ker_to_plot.value is not None)
        else [0]
    )
    eigen_vals_plot = mo.ui.plotly(px.scatter(x=list(eigen_vals), y=list(eigen_vals)))

    lr_distribution_figure = make_subplots(rows=1, cols=2, shared_yaxes=True)
    lr_distribution_figure.add_trace(
        go.Histogram(x=left_chosen_data["learning_rate"], nbinsx=50, name="step distribution ADG"), row=1, col=1
    )
    lr_distribution_figure.add_trace(
        go.Histogram(x=right_chosen_data["learning_rate"], nbinsx=50, name="optimal step distribution"), row=1, col=2
    )


    gd_results_plot = mo.ui.plotly(fig)
    lr_distribution_plot = mo.ui.plotly(lr_distribution_figure)
    return eigen_vals_plot, gd_results_plot, lr_distribution_plot


@app.cell
def _(
    eigen_vals_plot,
    gd_results_plot,
    lr_distribution_plot,
    mo,
    plot_constructor,
):
    mo.vstack([plot_constructor, gd_results_plot, eigen_vals_plot, lr_distribution_plot])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Notes

    #### Experiment setup
    For each quadratic form run both AGD and optimal step GD for $n$ steps. Specify some convergence threshold $\mathcal{L}$, i.e. if loss is smaller than $\mathcal{L}$ we consider g.d. converged. Filter out all unconverged steps and apply wilxocon test to vectors with loss and learning rate values. By default we apply alternative hypothesis than loss/lr vector of AGD has **bigger** values in distribution than optimal step. *Usually if null hypothesis failed to disprove it means than optimal step data vector is statistically bigger.*

    #### Experiment notes
    - uniform distribution $[\lambda_{\min}, \lambda_{\max}]$, converges faster and has bigger step
    - Beta($lpha=2$, $eta=100$) AGD converges faster than optimal step, step is bigger
    - Beta($lpha=100$, $eta=24$) AGD converges slower, and learning rate is smaller too

    I tried to create more *extreme* spectral gap:
    - All the eigenvalue are the $\lambda_{\min}$ and one is $\lambda_{\max}$. ADG converged slower. What is interesting that there are $pprox$ 200 functions that have this reversed. What is more interesting, that those are 200 functions that haven't converged.
    - All the eigenvalue are the $\lambda_{\max}$ and one is $\lambda_{\min}$. ADG converged slower add had smaller step.
    - All the eigenvalue are the $\lambda_{\min}$ and two are $\lambda_{\max}$ and $\lambda_{\max}-1$. ADG converged slower. What is interesting that there are $pprox$ 50 functions that have this reversed. What is more interesting, that those are 50 functions that haven't converged.
    """)
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
