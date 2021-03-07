import functools
from pathlib import Path
import warnings
import covsirphy as cs
import text as te

# Country to analyze
country_use = "Portugal"
abbr = "pt"

# Model type (SIR, SIR-F, etc...)
model = cs.SIRF

# Days to sim
sim_days = 365

# Each "dotted line" is the start of a phase

# Phase: A sequential dates in which parameter values of SIR-derived models are fixed

# Predictions: Prediction of parameter values in the future phases using relationship of
# estimated parameters and indexes regarding measures taken by countries and individuals.


def main():
    run()


def run():
    warnings.simplefilter("error")

    # Create output and input directories
    input_dir, output_dir = set_dirs()

    # Load datasets
    data_loader, jhu_data, population_data = set_datasets(input_dir)

    # Where figures will be saved
    figpath = functools.partial(
        filepath, output_dir=output_dir, country=abbr, ext="jpg")

    # Start scenario analysis
    snl = cs.Scenario(jhu_data, population_data, country_use)

    # Starting date
    print("BEGIN: " + snl.first_date)

    # Show records
    record_df = snl.records(filename=figpath("records"))
    save_df(record_df, "records", output_dir, abbr, use_index=False)

    # Show S-R trend
    snl.trend(filename=figpath("trend"))
    print(snl.summary())

    # Show summary
    # print("------------- SUMMARY OF PHASES: -------------")
    # print(snl.summary())
    # print("----------------------------------------------")

    # Parameter estimation
    snl.estimate(model)

    all_phases = snl.summary(name="Main").index.tolist()
    snl.history("Rt", phases=all_phases[1:], filename=figpath("history_rt_past"))

    # Add future phase to main scenario
    snl.add(name="Main", days=sim_days)

    # Short-term prediction with linear regression and OxCGRT data
    oxcgrt_data = data_loader.oxcgrt()
    print(oxcgrt_data.subset("Portugal"))
    snl.fit_predict(oxcgrt_data, name="Forecast")
    save_df(oxcgrt_data.subset("Portugal"), "oxcgrt", output_dir, abbr, use_index=False)
    # Simulation of the number of cases
    sim_df = snl.simulate(name="Main", filename=figpath("simulate"))
    save_df(sim_df, "simulate", output_dir, abbr, use_index=False)

    # Parameter history
    for item in ["Rt", "rho", "sigma", "Confirmed", "Infected", "Recovered", "Fatal"]:
        snl.history(item, filename=figpath(f"history_{item.lower()}"))

    # Change rate of parameters in main scenario
    snl.history_rate(name="Main", filename=figpath("history_rate"))
    snl.history_rate(
        params=["kappa", "sigma", "rho"],
        name="Main", filename=figpath("history_rate_without_theta"))

    # Save summary as a CSV file
    summary_df = snl.summary()
    save_df(summary_df, "summary", output_dir, abbr)

    # Score of main scenario
    metrics_list = ["MAE", "MSE", "MSLE", "RMSE", "RMSLE"]
    for metrics in metrics_list:
        metrics_name = metrics.rjust(len(max(metrics_list, key=len)))
        print(f"{metrics_name}: {snl.score(metrics=metrics)}")

    # Last date
    print("END: " + snl.last_date)


def filepath(name, output_dir, country, ext="jpg"):
    """
    Return filepath of a figure.
    Args:
        name (str): name of the figure
        output_dir (pathlib.Path): path of the directory to save the figure
        country (str): country name or abbr
        ext (str, optional): Extension of the output file. Defaults to "jpg".
    Returns:
        pathlib.Path: filepath of the output file
    """
    return output_dir.joinpath(f"{country}_{name}.{ext}")


def save_df(df, name, output_dir, country, use_index=True):
    """
    Save the dataframe as a CSV file.
    Args:
        df (pandas.DataFrame): dataframe
        name (str): name of the dataframe
        output_dir (pathlib.Path): path of the directory to save the figure
        country (str): country name or abbr
        use_index (bool): if True, include index
    """
    df.to_csv(output_dir.joinpath(f"{name}.csv"), index=use_index)


def set_dirs():
    code_path = Path(__file__)
    input_dir = code_path.parent.with_name("input")
    output_dir = code_path.with_name("output").joinpath(code_path.stem)
    output_dir.mkdir(exist_ok=True, parents=True)

    return input_dir, output_dir


def set_datasets(input_dir):
    data_loader = cs.DataLoader(input_dir)
    jhu_data = data_loader.jhu(verbose=2)
    print("------------- DATALOADER DATA CITATION -------------")
    print(jhu_data.citation)
    print("----------------------------------------------------")
    population_data = data_loader.population()

    return data_loader, jhu_data, population_data


if __name__ == "__main__":
    main()
