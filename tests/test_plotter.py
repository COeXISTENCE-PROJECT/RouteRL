import matplotlib

matplotlib.use("Agg", force=True)

from routerl.keychain import Keychain as kc
from routerl.services.plotter import Plotter, sns


def test_travel_time_distribution_boxplot_supports_current_matplotlib(
    tmp_path, monkeypatch
):
    plotter = Plotter(
        {
            kc.PLOT_CHOICES: kc.PLOT_ALL,
            kc.PHASES: [0, 2],
            kc.PHASE_NAMES: ["Human learning", "Machine learning"],
            kc.SMOOTH_BY: -1,
            kc.RECORDS_FOLDER: str(tmp_path / "records"),
            kc.PLOTS_FOLDER: str(tmp_path / "plots"),
        }
    )
    plotter.saved_episodes = [1, 2]

    mean_travel_times = {"(0, 1)": {1: 2.0, 2: 2.5}}
    variance_travel_times = {
        kc.ALL: {1: 0.2, 2: 0.3},
        kc.TYPE_HUMAN: {1: 0.2, 2: 0.3},
    }
    travel_times = {
        kc.ALL: {1: [1.0, 2.0, 3.0], 2: [1.5, 2.5, 3.5]},
        kc.TYPE_HUMAN: {1: [1.0, 2.0, 3.0], 2: [1.5, 2.5, 3.5]},
    }

    monkeypatch.setattr(
        plotter,
        "_retrieve_data_per_od",
        lambda data_key, transform=None: mean_travel_times,
    )
    monkeypatch.setattr(
        plotter,
        "_retrieve_data_per_kind",
        lambda data_key, transform=None: (
            variance_travel_times if transform == "variance" else travel_times
        ),
    )
    monkeypatch.setattr(sns, "kdeplot", lambda *args, **kwargs: None)

    plotter.visualize_tt_distributions()

    assert (tmp_path / "plots" / kc.TT_DIST_PLOT_FILE_NAME).is_file()
