import streamlit as st
import altair as alt
import pandas as pd
import os
import requests
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path
from functools import lru_cache
from datetime import datetime
import pandasdmx as pdmx


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


def ons_blue_book_data(code):
    data = grab_ONS_time_series_data("BB", code)
    xf = pd.DataFrame(pd.json_normalize(data['years']))
    xf = xf[["year", "value"]]
    xf["year"] = xf["year"].astype(int)
    xf["value"] = xf["value"].astype(float)
    xf["title"] = data["description"]["title"]
    xf["code"] = code
    xf = pd.DataFrame(xf.loc[xf["year"].argmax(), :]).T
    return xf


@lru_cache(maxsize=32)
def ons_get_gdp_output_with_breakdown():
    df = prep_gdp_output_codes()
    xf = pd.DataFrame()
    for code in df["code"].unique():
        xf = pd.concat([xf, ons_blue_book_data(code)], axis=0)
    df = pd.merge(df, xf, on=["code"], how="inner")
    # for later treemap use, only use highest level name if hierachy has
    # missing levels
    df.loc[(df[1] == df[2]) & (df[3] == df[2]) & (df[3]==df[0]), [3, 2, 1]] = None
    df.loc[(df[1] == df[2]) & (df[3] == df[2]), [3, 2]] = None
    df.loc[(df[1] == df[2]), [2]] = None
    # now, any nones with non-none children must be swapped
    df.loc[(df[2].isnull()) & (~df[3].isnull()), [2, 3]] = df.loc[(df[2].isnull()) & (~df[3].isnull()), [3, 2]].values
    df.loc[(df[0] == df[1]), [1]] = df.loc[(df[0] == df[1]), [2]].values
    df.loc[(df[1] == df[2]), [2]] = df.loc[(df[1] == df[2]), [3]].values
    # another round of this
    df.loc[(df[1] == df[2]) & (df[3] == df[2]) & (df[3]==df[0]), [3, 2, 1]] = None
    df.loc[(df[1] == df[2]) & (df[3] == df[2]), [3, 2]] = None
    df.loc[(df[1] == df[2]), [2]] = None
    df.loc[(df[3] == df[2]), [3]] = None
    return df


def grab_ONS_time_series_data(dataset_id, timeseries_id):
    """
    This function grabs specified time series from the ONS API.

    """
    api_endpoint = "https://api.ons.gov.uk/"
    api_params = {'dataset': dataset_id, 'timeseries': timeseries_id}
    url = (api_endpoint + '/'.join(
        [x + '/' + y
         for x, y in zip(api_params.keys(), api_params.values())][::-1]) +
           '/data')
    return requests.get(url).json()


def ons_clean_qna_data(data):
    df = pd.DataFrame(pd.json_normalize(data['quarters']))
    df['date'] = (pd.to_datetime(
        df['date'].apply(lambda x: x[:4] + '-' + str(int(x[-1]) * 3)),
        format='%Y-%m') + pd.tseries.offsets.MonthEnd())
    df = df.drop([x for x in df.columns if x not in ['date', 'value']], axis=1)
    return df


def ons_qna_data(dataset_id, timeseries_id):
    data = grab_ONS_time_series_data(dataset_id, timeseries_id)
    desc_text = data['description']['title']
    df = ons_clean_qna_data(data)
    return df, desc_text


def visualize_line(df, x_axis, y_axis, scale, widths,
                   ylabel, title):
    height = 350
    graph = alt.Chart(df).mark_line(strokeWidth=4).encode(
        x=x_axis+':T',
        y=alt.Y(y_axis+':Q', scale=scale, title=ylabel),
        tooltip=[y_axis]
    ).properties(width=widths, title=title, height=height).interactive()
    st.write(graph)


def main():
    """ Func description
    """
    r'''
    # The UK economy in charts

    This dashboard provides an overview of the UK economy.
    This is written in ```markdown``` and is latex friendly.
    Here is the equation to predict:
    $$y = 3x\sin\left(\frac{\pi}{20}\right) + \varepsilon$$;
    ML model is re-run in real-time as no. of samples changes (test set is 20%
    of full datasets).

    ## Output

    ### Indices of Production, Construction, and Services

    '''
    # Grab the three UK time series
    indices_dicts = {'Production': 'L2KQ',
                     'Construction': 'L2N8',
                     'Services': 'L2NC'}
    df = pd.DataFrame()
    for key, value in indices_dicts.items():
        xf, x_text = ons_qna_data('QNA', value)
        xf['Name'] = key
        df = pd.concat([df, xf], axis=0)
    # Construct the charts
    graph = alt.Chart(df).mark_line(strokeWidth=4).encode(
        x=alt.X('date:T'),
        y='value:Q',
        color=alt.Color('Name:N', legend=None),
        tooltip=['value']
    ).properties(
        width=200,
        height=200,
    ).facet(
        column='Name:N'
    ).interactive()
    st.write(graph)

    # UK Blue Book breakdown of GDP
    text_for_GDP = r"""
    ### GDP by output
    This is output according to the ONS' Blue Book. Click on any component to see more details of that component.
    """
    st.write(text_for_GDP)

    df = px.data.gapminder().query("year == 2007")
    df = ons_get_gdp_output_with_breakdown()
    fig = px.treemap(df, path=[px.Constant("GDP (£m, current prices)"), 0, 1, 2, 3], values='value',
                    hover_data=["title"],
                    color_discrete_sequence=px.colors.qualitative.Bold
                    )
    fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
    st.write(fig)

    # UK regional GVA
    uk_nuts1_gva = pd.read_csv(os.path.join('data', 'nuts1_gva_2016.csv'))

    # The labour market. TODO change to monthly LMS (series codes are same)
    indices_dicts_lms = {'Employment': 'LF24',
                         'Unemployment': 'MGSX',
                         'Inactivity': 'LF2S'}
    df_lms = pd.DataFrame()
    for key, value in indices_dicts_lms.items():
        xf, x_text = ons_qna_data('LMS', value)
        xf['Name'] = key
        df_lms = pd.concat([df_lms, xf], axis=0)
    graph_lms = alt.Chart(df_lms).mark_line(strokeWidth=4).encode(
        x=alt.X('date:T', title=''),
        y=alt.Y('value:Q', title='%'),
        color='Name:N',
        tooltip=['value'],
    ).properties(
        title='Labour market indicators',
        width=600,
    ).interactive()
    st.write(graph_lms)


if __name__ == "__main__":
    main()
