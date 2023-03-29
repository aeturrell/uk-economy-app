import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path
import statsmodels.formula.api as smf
from datetime import datetime
import pandasdmx as pdmx
from typing import Tuple

plt.style.use("plot_style.txt")
plt.rcParams["figure.dpi"] = 300


@st.cache_data
def prep_gdp_output_codes() -> pd.DataFrame():
    """Extracts the hierarchical GDP codes from a static ONS file with them all in.

    Returns:
        pd.DataFrame: Data frame that includes columns section and code, mapping GDP sections into their codes.
    """
    hdf = pd.read_excel(Path("data", "uk_gdp_output_hierarchy.xlsx"), header=None)
    hdf = hdf.dropna(how="all", axis=1)
    for i in range(3):
        hdf.iloc[i, :] = hdf.iloc[i, :].fillna(method="ffill")
    hdf = hdf.T
    hdf["total"] = hdf[3].str.contains("Total")
    non_total_hdf = hdf.query("total==False")
    non_total_hdf = non_total_hdf.drop("total", axis=1)
    for col in range(5):
        non_total_hdf[col] = non_total_hdf[col].str.lstrip().str.rstrip()
    non_total_hdf = non_total_hdf.rename(columns={4: "section", 5: "code"})
    return non_total_hdf


@st.cache_data(max_entries=1000)
def grab_ONS_time_series_data(dataset_id: str, timeseries_id: str):
    """This function grabs specified time series from the ONS API.

    Args:
        dataset_id (str): eg Consumer Price Inflation time series (MM23)
        timeseries_id (str): eg RPI for UK holidays (CHMS)

    Returns:
        JSON: Contains series.
    """
    api_endpoint = "https://api.ons.gov.uk/"
    api_params = {"dataset": dataset_id, "timeseries": timeseries_id}
    url = (
        api_endpoint
        + "/".join(
            [x + "/" + y for x, y in zip(api_params.keys(), api_params.values())][::-1]
        )
        + "/data"
    )
    return requests.get(url).json()


def get_uk_regional_gdp():
    # current year
    latest_year = datetime.now().year - 1
    # Tell pdmx we want OECD data
    oecd = pdmx.Request("OECD")
    # Set out everything about the request in the format specified by the OECD API
    data = oecd.data(
        resource_id="REGION_ECONOM",
        key="1+2.UKC.SNA_2008.GDP.REG+CURR_PR.ALL.2017+2018+2019+2020/all?",
    ).to_pandas()
    # example that works:
    "https://stats.oecd.org/restsdmx/sdmx.ashx/GetData/REGION_ECONOM/1+2.GBR+UKC+UKC11+UKC12.SNA_2008.GDP.REG+CURR_PR+USD_PPP+REAL_PR+REAL_PPP+PC+PC_CURR_PR+PC_USD_PPP+PC_REAL_PR+PC_REAL_PPP.ALL.2001+2002+2003+2004+2005+2006+2007+2008+2009+2010+2011+2012+2013+2014+2015+2016+2017+2018+2019+2020/all?"
    df = pd.DataFrame(data).reset_index()
    df.head()


@st.cache_data(max_entries=1000)
def ons_blue_book_data(code: str) -> pd.DataFrame:
    """Retrieve data from ONS Blue Book given GDP component code.

    Args:
        code (str): GDP component code, eg KKN5.

    Returns:
        pd.DataFrame: Blue Book entry for this particular code.
    """
    data = grab_ONS_time_series_data("BB", code)
    xf = pd.DataFrame(pd.json_normalize(data["years"]))
    xf = xf[["year", "value"]]
    xf["year"] = xf["year"].astype(int)
    xf["value"] = xf["value"].astype(float)
    xf["title"] = data["description"]["title"]
    xf["code"] = code
    xf = pd.DataFrame(xf.loc[xf["year"].argmax(), :]).T
    return xf


@st.cache_data(show_spinner="Fetching data from ONS API...", max_entries=100)
def ons_get_gdp_output_with_breakdown() -> pd.DataFrame:
    df = prep_gdp_output_codes()
    xf = pd.DataFrame()
    for code in df["code"].unique():
        xf = pd.concat([xf, ons_blue_book_data(code)], axis=0)
    df = pd.merge(df, xf, on=["code"], how="inner")
    # for later treemap use, only use highest level name if hierachy has
    # missing levels
    df.loc[(df[1] == df[2]) & (df[3] == df[2]) & (df[3] == df[0]), [3, 2, 1]] = None
    df.loc[(df[1] == df[2]) & (df[3] == df[2]), [3, 2]] = None
    df.loc[(df[1] == df[2]), [2]] = None
    # now, any nones with non-none children must be swapped
    df.loc[(df[2].isnull()) & (~df[3].isnull()), [2, 3]] = df.loc[
        (df[2].isnull()) & (~df[3].isnull()), [3, 2]
    ].values
    df.loc[(df[0] == df[1]), [1]] = df.loc[(df[0] == df[1]), [2]].values
    df.loc[(df[1] == df[2]), [2]] = df.loc[(df[1] == df[2]), [3]].values
    # another round of this
    df.loc[(df[1] == df[2]) & (df[3] == df[2]) & (df[3] == df[0]), [3, 2, 1]] = None
    df.loc[(df[1] == df[2]) & (df[3] == df[2]), [3, 2]] = None
    df.loc[(df[1] == df[2]), [2]] = None
    df.loc[(df[3] == df[2]), [3]] = None
    return df


@st.cache_data()
def ons_clean_qna_data(data) -> pd.DataFrame:
    """Cleans Quarterly National Accounts data retrieved from ONS API.

    Args:
        data (JSON): JSON as returned from particular code retrieval.

    Returns:
        pd.DataFrame: Cleaned data in dataframe.
    """
    if data["quarters"] != []:
        df = pd.DataFrame(pd.json_normalize(data["quarters"]))
        df["date"] = (
            pd.to_datetime(
                df["date"].apply(lambda x: x[:4] + "-" + str(int(x[-1]) * 3)),
                format="%Y-%m",
            )
            + pd.tseries.offsets.MonthEnd()
        )
    else:
        df = pd.DataFrame(pd.json_normalize(data["months"]))
        df["date"] = (
            pd.to_datetime(df["date"], format="%Y %b") + pd.tseries.offsets.MonthEnd()
        )
    df = df.drop([x for x in df.columns if x not in ["date", "value"]], axis=1)
    return df


@st.cache_data(show_spinner="Fetching data from ONS API...", max_entries=1000)
def ons_qna_data(dataset_id: str, timeseries_id: str) -> Tuple[pd.DataFrame, str]:
    data = grab_ONS_time_series_data(dataset_id, timeseries_id)
    desc_text = data["description"]["title"]
    df = ons_clean_qna_data(data)
    return df, desc_text


@st.cache_data
def data_on_output_indices() -> pd.DataFrame:
    """Retrieve production, construction, and services indices.

    Returns:
        pd.DataFrame: dataframe with series
    """
    indices_dicts = {"Production": "L2KQ", "Construction": "L2N8", "Services": "L2NC"}
    df = pd.DataFrame()
    for key, value in indices_dicts.items():
        xf, x_text = ons_qna_data("QNA", value)
        xf["Name"] = key
        df = pd.concat([df, xf], axis=0)
    return df


def plot_index_of_output(df: pd.DataFrame) -> None:
    """Plots indices of output using plotly.

    Args:
        series_name (str): One of "Production", "Construction", or "Services"
    """
    df = df.rename(columns={"date": "Date", "value": "Index"})
    fig = px.line(
        df,
        x="Date",
        y="Index",
        color="Name",
        line_dash="Name",
        title=f"Indices of output",
    )
    st.plotly_chart(fig)


@st.cache_data()
def data_for_labour_market_indicators() -> pd.DataFrame:
    """Retrieve Employment, Unemployment, and Inactivity. TODO change to monthly LMS (series codes are same).

    Returns:
        pd.DataFrame: Dataframe with E, U, and Inact.
    """
    indices_dicts_lms = {
        "Employment": "LF24",
        "Unemployment": "MGSX",
        "Inactivity": "LF2S",
    }
    df_lms = pd.DataFrame()
    for key, value in indices_dicts_lms.items():
        xf, x_text = ons_qna_data("LMS", value)
        xf["Name"] = key
        df_lms = pd.concat([df_lms, xf], axis=0)
    df_lms["value"] = pd.to_numeric(df_lms["value"])
    return df_lms


def plotly_labour_market_indicators(df_lms: pd.DataFrame) -> None:
    """Plot E, U, and inactivity."""

    fig = px.line(
        df_lms.reset_index(),
        x="date",
        y="value",
        color="Name",
        line_dash="Name",
        title="Employment and activity, %",
    )
    st.plotly_chart(fig)


@st.cache_data()
def data_for_beveridge_curve() -> pd.DataFrame:
    """Retrieves ONS series relevant to Beveridge curve.

    Returns:
        pd.DataFrame: all series in dataframe.
    """
    indices_dicts_lms = {"Vacancies": "AP2Y", "Unemployment": "MGSX", "Active": "LF2K"}
    df = pd.DataFrame()
    for key, value in indices_dicts_lms.items():
        xf, x_text = ons_qna_data("LMS", value)
        xf["Name"] = key
        df = pd.concat([df, xf], axis=0)
    df["value"] = df["value"].astype(np.double)
    df = pd.pivot(df, index="date", columns="Name")
    df.columns = df.columns.droplevel()
    df = df.dropna()
    df["Date"] = df.index
    df["Vacancies"] = 100 * df["Vacancies"].divide(df["Active"])
    return df


@st.cache_data()
def data_cpi_cpih() -> pd.DataFrame():
    """Retrieves data on ONS CPI, CPIH (change over 12 months)"""
    indices_dicts_cpi = {"CPI": "D7G7", "CPIH": "L55O"}
    dataset_id = "MM23"
    df = pd.DataFrame()
    for key, value in indices_dicts_cpi.items():
        json_data = grab_ONS_time_series_data(dataset_id, value)
        unit = json_data["description"]["unit"]
        xf = (
            pd.DataFrame(pd.json_normalize(json_data["months"]))
            .assign(
                Date=lambda x: pd.to_datetime(x["date"]),
                Percent=lambda x: pd.to_numeric(x["value"]),
            )
            .set_index("Date")
            .drop(
                ["label", "month", "quarter", "sourceDataset", "updateDate", "year"],
                axis=1,
            )
        )
        xf["Name"] = key
        xf["unit"] = unit
        df = pd.concat([df, xf], axis=0)
    return df


def plotly_inflation_indicators(df_cpi: pd.DataFrame) -> None:
    """Plot CPI and CPIH % over 12 months."""

    fig = px.line(
        df_cpi.reset_index(),
        x="Date",
        y="Percent",
        color="Name",
        line_dash="Name",
        title="Inflation (change over 12 months)",
    )
    st.plotly_chart(fig)


def plot_beveridge_curve(df: pd.DataFrame) -> None:
    """Plots Beveridge curve.

    Args:
        df (pd.DataFrame): dataframe with datetime index, and columns for U and V.
    """
    # Need to divide vacs by labour force size
    # Need to label most extremal u value
    fig, ax = plt.subplots()
    max_u = df["Unemployment"].argmax()
    quivx = -df["Unemployment"].diff(-1)
    quivy = -df["Vacancies"].diff(-1)
    # This connects the points
    ax.quiver(
        df["Unemployment"],
        df["Vacancies"],
        quivx,
        quivy,
        scale_units="xy",
        angles="xy",
        scale=1,
        width=0.006,
        alpha=0.3,
    )
    ax.scatter(
        df["Unemployment"],
        df["Vacancies"],
        marker="o",
        s=35,
        edgecolor="black",
        linewidth=0.2,
        alpha=0.9,
    )
    for j in [0, max_u, -1]:
        ax.annotate(
            f'{df["Date"].iloc[j].year} Q{df["Date"].iloc[j].quarter}',
            xy=(df[["Unemployment", "Vacancies"]].iloc[j].tolist()),
            xycoords="data",
            xytext=(20, 20),
            textcoords="offset points",
            arrowprops=dict(
                arrowstyle="->", connectionstyle="angle3,angleA=0,angleB=-90"
            ),
        )
    ax.set_title("Vacancy rate, %")
    ax.set_xlabel("Unemployment rate, %")
    ax.grid(which="major", axis="both", lw=0.2)
    plt.tight_layout()
    plt.suptitle("Beveridge Curve")
    st.pyplot(fig)


@st.cache_data
def data_phillips_curve() -> pd.DataFrame:
    """Retrieves weekly earnings and unemployment from ONS API.

    Returns:
        pd.DataFrame: Dataframe with date, AWE, and U.
    """
    indices_dicts = {
        "Average weekly earnings": ("LMS", "KAC3"),
        "Unemployment": ("LMS", "MGSX"),
    }
    df = pd.DataFrame()
    for key, value in indices_dicts.items():
        xf, x_text = ons_qna_data(*value)
        xf["Name"] = key
        df = pd.concat([df, xf], axis=0)
    df["value"] = df["value"].astype(np.double)
    df = pd.pivot(df, index="date", columns="Name")
    df.columns = df.columns.droplevel()
    df = df.dropna()
    df = df.groupby(pd.Grouper(freq="Y")).mean()
    return df


def plot_phillips_curve(df: pd.DataFrame) -> None:
    """Plot Phillips curve.

    Args:
        df (pd.DataFrame): Dataframe with date, avg weekly earnings, and unemp.
    """
    # create year groupings
    df["group"] = pd.cut(
        df.index,
        bins=[
            df.index.min() - pd.offsets.YearEnd(),
            pd.to_datetime("2009") + pd.offsets.YearEnd(),
            df.index.max(),
        ],
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    df["awe"] = df["Average weekly earnings"]
    max_u = df["Unemployment"].argmax()
    fig, ax = plt.subplots()
    for i, grp in enumerate(df["group"].unique()):
        mod = smf.ols(formula="awe ~ Unemployment", data=df[df["group"] == grp])
        res = mod.fit()
        ax.scatter(
            df.loc[df["group"] == grp, "Unemployment"],
            df.loc[df["group"] == grp, "Average weekly earnings"],
            marker="o",
            s=35,
            edgecolor="black",
            color=colors[i],
            linewidth=0.2,
            alpha=0.9,
        )
        ax.plot(
            df.loc[df["group"] == grp, "Unemployment"],
            res.fittedvalues,
            label=f"{grp.left.year+1}—{grp.right.year}",
            lw=3,
            alpha=0.7,
        )
    for j in [0, max_u, -1]:
        ax.annotate(
            f"{df.index[j].year}",
            xy=(df[["Unemployment", "Average weekly earnings"]].iloc[j].tolist()),
            xycoords="data",
            xytext=(-40, 0),
            textcoords="offset points",
            va="center",
            ha="center",
            arrowprops=dict(
                arrowstyle="->", connectionstyle="angle3,angleA=0,angleB=-90"
            ),
        )
    ax.grid(which="major", axis="both", lw=0.2)
    ax.set_xlabel("Mean unemployment rate, %")
    ax.tick_params(axis="both", which="both", labelsize=10)
    ax.legend(frameon=False)
    ax.set_title("Mean nominal wage growth, %")
    plt.suptitle("Phillips Curve")
    plt.tight_layout()
    st.pyplot(fig)


def main():
    """Func description"""
    # st.set_page_config(layout="wide")
    r"""
    # The UK economy in charts

    This dashboard provides an overview of the UK economy.

    ## Output

    ### Indices of Production, Construction, and Services
    """
    df_outputs = data_on_output_indices()
    plot_index_of_output(df_outputs)
    st.write(
        """
    ### UK Blue Book breakdown of GDP
    """
    )
    df = ons_get_gdp_output_with_breakdown()
    year_blue_book = df.loc[0, "year"]
    fig = px.treemap(
        df,
        path=[px.Constant(f"GDP (£m, current prices, {year_blue_book})"), 0, 1, 2, 3],
        values="value",
        hover_data=["title"],
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    st.write(fig)
    st.write(
        """
        ## Inflation

        ### Headline measures
        """
    )
    df_cpi = data_cpi_cpih()
    plotly_inflation_indicators(df_cpi)
    st.write(
        """
    ## Labour Market

    ### Labour market indicators
    """
    )
    # Labour market indicators
    df_lms = data_for_labour_market_indicators()
    list_of_cols = st.columns(3)
    date_since_chg = df_lms["date"].max() - pd.DateOffset(years=1)
    delta_colours = ["normal", "inverse", "inverse"]
    for i, name in enumerate(df_lms["Name"].unique()):
        xf = df_lms.loc[df_lms["Name"] == name, :]
        latest_val = xf.iloc[-1, 1]
        val_at_ref = xf.loc[xf["date"] == date_since_chg, "value"].values[0]
        ppt_change = latest_val - val_at_ref
        str_to_display = f"{ppt_change:.2f} pp since {date_since_chg.date()}"
        list_of_cols[i].metric(
            name, f"{latest_val}", str_to_display, delta_color=delta_colours[i]
        )

    plotly_labour_market_indicators(df_lms)
    df_bev = data_for_beveridge_curve()
    df_phl = data_phillips_curve()
    plot_beveridge_curve(df_bev)
    plot_phillips_curve(df_phl)


if __name__ == "__main__":
    main()
