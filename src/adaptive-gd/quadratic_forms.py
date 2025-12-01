import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np

    from numpy.typing import NDArray

    import polars as pl
    import plotly.express as px

    from scipy.stats import ortho_group
    return NDArray, np, ortho_group


@app.cell
def _(NDArray, NDarray, np, ortho_group):
    def gen_eigenvalues(
        min_ev: float = 0.0, max_ev: float = 1e9, dim: int = 3, null_space_dim: int = 0
    ) -> NDArray[np.float64]:
        non_zero_evs = np.random.uniform(low=min_ev, high=max_ev, size=dim - null_space_dim)
        return np.hstack((np.zeros(null_space_dim), non_zero_evs))


    def gen_eigenvectors(dim: int = 3) -> NDarray[np.float64]:
        return ortho_group.rvs(dim=dim)


    def gen_quadratic_form(
        min_ev: float = 0.0, max_ev: float = 1e9, dim: int = 3, null_space_dim: int = 0
    ) -> tuple[NDarray[np.float64], NDArray[np.float64]]:
        if null_space_dim > dim:
            raise ValueError("Null space dimension cannot be greater than dimensions of matrix")

        eigen_vals = gen_eigenvalues(min_ev=min_ev, max_ev=max_ev, dim=dim)
        jordan_form = np.diag(eigen_vals)

        eigen_vecs = gen_eigenvectors(dim=dim)
        inv_eigen_vec = np.linalg.inv(eigen_vecs)

        b = np.random.rand(dim)

        A = eigen_vecs @ jordan_form @ inv_eigen_vec
        return A, b
    return (gen_quadratic_form,)


@app.cell
def _(mo):
    min_ev = mo.ui.slider(start=0, stop=100, step=0.1)
    max_ev = mo.ui.slider(start=0, stop=100, step=0.1)
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
def _(dim, gen_quadratic_form, max_ev, min_ev, null_space_dim):
    A1, b1 = gen_quadratic_form(min_ev.value, max_ev.value, dim.value, null_space_dim.value)
    return A1, b1


@app.cell
def _(A1, b1):
    A1, b1
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
