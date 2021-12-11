import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path
from functools import lru_cache
import statsmodels.formula.api as smf

plt.style.use(
    "https://github.com/aeturrell/coding-for-economists/raw/main/plot_style.txt"
)


@st.cache
def prep_gdp_output_codes():
    hdf = pd.read_excel(Path("data", "uk_gdp_output_hierarchy.xlsx"), header=None)
    hdf = hdf.dropna(how="all", axis=1)
    for i in range(3):
        hdf.iloc[i, :] = hdf.iloc[i, :].fillna(method="ffill")
    hdf = hdf.T
    hdf["total"] = hdf[3].str.contains("Total")
    hdf = hdf.query("total==False")
    hdf = hdf.drop("total", axis=1)
    for col in range(5):
        hdf[col] = hdf[col].str.lstrip().str.rstrip()
    hdf = hdf.rename(columns={4: "section", 5: "code"})
    return hdf


@st.cache
def ons_blue_book_data(code):
    data = grab_ONS_time_series_data("BB", code)
    xf = pd.DataFrame(pd.json_normalize(data["years"]))
    xf = xf[["year", "value"]]
    xf["year"] = xf["year"].astype(int)
    xf["value"] = xf["value"].astype(float)
    xf["title"] = data["description"]["title"]
    xf["code"] = code
    xf = pd.DataFrame(xf.loc[xf["year"].argmax(), :]).T
    return xf


@st.cache
@lru_cache(maxsize=32)
def ons_get_gdp_output_with_breakdown():
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


@st.cache
def grab_ONS_time_series_data(dataset_id, timeseries_id):
    """
    This function grabs specified time series from the ONS API.

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


def ons_clean_qna_data(data):
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


@lru_cache(maxsize=32)
def ons_qna_data(dataset_id, timeseries_id):
    data = grab_ONS_time_series_data(dataset_id, timeseries_id)
    desc_text = data["description"]["title"]
    df = ons_clean_qna_data(data)
    return df, desc_text


def visualize_line(df, x_axis, y_axis, scale, widths, ylabel, title):
    height = 350
    graph = (
        alt.Chart(df)
        .mark_line(strokeWidth=4)
        .encode(
            x=x_axis + ":T",
            y=alt.Y(y_axis + ":Q", scale=scale, title=ylabel),
            tooltip=[y_axis],
        )
        .properties(width=widths, title=title, height=height)
        .interactive()
    )
    st.write(graph)


def plot_indices_of_output():
    # Grab the three UK time series
    indices_dicts = {"Production": "L2KQ", "Construction": "L2N8", "Services": "L2NC"}
    df = pd.DataFrame()
    for key, value in indices_dicts.items():
        xf, x_text = ons_qna_data("QNA", value)
        xf["Name"] = key
        df = pd.concat([df, xf], axis=0)

    graph = (
        alt.Chart(df)
        .mark_line(strokeWidth=4)
        .encode(
            x=alt.X("date:T"),
            y="value:Q",
            color=alt.Color("Name:N", legend=None),
            tooltip=["value"],
        )
        .properties(
            width=200,
            height=200,
        )
        .facet(column="Name:N")
        .interactive()
    )
    st.write(graph)


def plot_labour_market_indicators():
    # The labour market. TODO change to monthly LMS (series codes are same)
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
    graph_lms = (
        alt.Chart(df_lms)
        .mark_line(strokeWidth=4)
        .encode(
            x=alt.X("date:T", title=""),
            y=alt.Y("value:Q", title="%"),
            color="Name:N",
            tooltip=["value"],
        )
        .properties(
            title="Labour market indicators",
            width=600,
        )
        .interactive()
    )
    st.write(graph_lms)


def plot_beveridge_curve():
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
    max_u = df["Unemployment"].argmax()
    # Need to divide vacs by labour force size
    # Need to label most extremal u value
    fig, ax = plt.subplots()
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
    ax.set_xlabel("Unemployment rate, %")
    ax.set_ylabel("Vacancy rate, %")
    ax.grid(which="major", axis="both", lw=0.2)
    plt.tight_layout()
    st.pyplot(fig)


def plot_phillips_curve():
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
    ax.set_ylabel("Mean nominal wage growth, %")
    ax.tick_params(axis="both", which="both", labelsize=10)
    ax.legend(frameon=False)
    ax.set_title("Phillips curve")
    plt.tight_layout()
    st.pyplot(fig)


def main():
    """Func description"""
    r"""
    # The UK economy in charts

    This dashboard provides an overview of the UK economy.

    ## Output

    ### Indices of Production, Construction, and Services
    """
    plot_indices_of_output()
    st.write(
        """
    ### UK Blue Book breakdown of GDP
    """
    )
    df = ons_get_gdp_output_with_breakdown()
    fig = px.treemap(
        df,
        path=[px.Constant("GDP (£m, current prices)"), 0, 1, 2, 3],
        values="value",
        hover_data=["title"],
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    st.write(fig)

    st.write(
        """
    ## Labour Market

    ### Labour market indicators
    """
    )
    # Labour market indicators
    plot_labour_market_indicators()
    st.write(
        """
    ### Beveridge curve
    """
    )
    plot_beveridge_curve()
    st.write("""
    ### Phillips curve
    """)
    plot_phillips_curve()


if __name__ == "__main__":
    main()
