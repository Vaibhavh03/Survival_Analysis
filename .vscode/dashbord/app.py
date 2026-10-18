from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_PATH = (
    BASE_DIR
    / "dataset"
    / "heart_failure_cleaned.csv"
)

CSS_PATH = (
    Path(__file__).resolve().parent
    / "style.css"
)

HEART_PATH = (
    Path(__file__).resolve().parent.parent
    / "assets"
    / "hearts.png"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Heart Failure Survival Analysis",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD CSS
# ============================================================

if CSS_PATH.exists():

    with open(CSS_PATH, "r", encoding="utf-8") as file:

        st.markdown(
            f"<style>{file.read()}</style>",
            unsafe_allow_html=True
        )


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(DATA_PATH)

if not DATA_PATH.exists():

    st.error(
        "Cleaned dataset not found.\n\n"
        "Run:\n\n"
        "`python src/data_cleaning.py`"
    )

    st.stop()


df = load_data()


# ============================================================
# CREATE AGE GROUP
# ============================================================

df["Age Group"] = pd.cut(
    df["age"],
    bins=[0, 50, 60, 70, np.inf],
    labels=[
        "40–50",
        "51–60",
        "61–70",
        "71+"
    ],
    include_lowest=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <h1 style="text-align:center;">
        ❤️
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "Dashboard navigation",
        [
            "Overview",
            "Survival Analysis",
            "Clinical Analysis",
            "Patient Data",
            "Filters"
        ],
        label_visibility="collapsed"
    )


# ============================================================
# HEADER
# ============================================================

header_col1, header_col2 = st.columns(
    [5, 5]
)


with header_col1:

    st.markdown(
        '<div class="dashboard-title">'
        'Heart Failure Survival Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dashboard-subtitle">'
        f'{len(df)} patient records • '
        'Clinical Records • Survival Analytics'
        '</div>',
        unsafe_allow_html=True
    )


with header_col2:

    st.markdown(
        '<div class="age-label">'
        'AGE GROUP'
        '</div>',
        unsafe_allow_html=True
    )

    age_filter = st.radio(
        "Age group filter",
        [
            "All",
            "40–50",
            "51–60",
            "61–70",
            "71+"
        ],
        horizontal=True,
        label_visibility="collapsed"
    )


# ============================================================
# APPLY AGE FILTER
# ============================================================

if age_filter == "All":

    filtered = df.copy()

else:

    filtered = df[
        df["Age Group"] == age_filter
    ].copy()


if page != "Overview":

    st.markdown("---")

    if page == "Survival Analysis":

        st.header("Survival Analysis")
        st.caption("Explore survival outcomes by age group and follow-up time.")

        survival_col1, survival_col2 = st.columns(2)

        with survival_col1:
            age_survival = (
                filtered.groupby("Age Group", observed=True)["DEATH_EVENT"]
                .agg(["count", "sum"])
                .reset_index()
            )
            age_survival["Survival Rate"] = (
                1 - age_survival["sum"] / age_survival["count"]
            ) * 100

            survival_by_age = go.Figure(
                go.Bar(
                    x=age_survival["Age Group"].astype(str),
                    y=age_survival["Survival Rate"],
                    marker_color="#bd1738",
                    text=age_survival["Survival Rate"].round(1),
                    texttemplate="%{text}%",
                    textposition="outside"
                )
            )
            survival_by_age.update_layout(
                title="Survival Rate by Age Group",
                yaxis=dict(title="Survival rate (%)", range=[0, 100]),
                template="plotly_white",
                height=400
            )
            st.plotly_chart(survival_by_age, width="stretch", theme=None)

        with survival_col2:
            from lifelines import KaplanMeierFitter

            km_figure = go.Figure()
            for age_group, group_df in filtered.groupby("Age Group", observed=True):
                if group_df.empty:
                    continue
                kmf = KaplanMeierFitter()
                kmf.fit(
                    durations=group_df["time"],
                    event_observed=group_df["DEATH_EVENT"],
                    label=str(age_group)
                )
                km_curve = kmf.survival_function_
                km_figure.add_trace(
                    go.Scatter(
                        x=km_curve.index,
                        y=km_curve.iloc[:, 0],
                        mode="lines",
                        name=str(age_group)
                    )
                )
            km_figure.update_layout(
                title="Kaplan–Meier Survival by Age Group",
                xaxis_title="Follow-up time (days)",
                yaxis_title="Estimated survival probability",
                yaxis=dict(range=[0, 1.05]),
                template="plotly_white",
                height=400
            )
            st.plotly_chart(km_figure, width="stretch", theme=None)

        st.info(
            "Kaplan–Meier curves describe observed group-level survival; "
            "they are not individual medical predictions."
        )

    elif page == "Clinical Analysis":

        st.header("Clinical Analysis")
        st.caption("Compare clinical measurements across patient outcomes.")

        clinical_columns = [
            ("Ejection Fraction", "ejection_fraction", "%"),
            ("Serum Creatinine", "serum_creatinine", "mg/dL"),
            ("Serum Sodium", "serum_sodium", "mEq/L")
        ]

        clinical_cols = st.columns(len(clinical_columns))
        outcome_names = {0: "Survived", 1: "Deceased"}
        for column, (title, variable, unit) in zip(clinical_cols, clinical_columns):
            with column:
                clinical_figure = go.Figure()
                for outcome, outcome_name in outcome_names.items():
                    values = filtered.loc[
                        filtered["DEATH_EVENT"] == outcome,
                        variable
                    ].dropna()
                    clinical_figure.add_trace(
                        go.Box(
                            y=values,
                            name=outcome_name,
                            marker_color="#bd1738" if outcome else "#f27a32",
                            boxmean=True
                        )
                    )
                clinical_figure.update_layout(
                    title=title,
                    yaxis_title=unit,
                    template="plotly_white",
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(clinical_figure, width="stretch", theme=None)

        st.subheader("Average clinical measurements by outcome")
        means = (
            filtered.groupby("DEATH_EVENT")[
                [column for _, column, _ in clinical_columns]
            ]
            .mean()
            .rename(index=outcome_names)
            .rename_axis("Outcome")
            .reset_index()
        )
        st.dataframe(means, width="stretch", hide_index=True)

    elif page == "Patient Data":

        st.header("Patient Clinical Records")
        st.caption(f"{len(filtered)} records match the selected age filter.")

        display_columns = [
            "age",
            "sex",
            "anaemia",
            "diabetes",
            "ejection_fraction",
            "high_blood_pressure",
            "serum_creatinine",
            "serum_sodium",
            "smoking",
            "time",
            "DEATH_EVENT"
        ]
        patient_table = filtered[display_columns].rename(
            columns={"DEATH_EVENT": "Deceased", "sex": "Sex (1=Male)"}
        )
        st.dataframe(patient_table, width="stretch", hide_index=True)

        csv_data = patient_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered patient records",
            data=csv_data,
            file_name="heart_failure_filtered.csv",
            mime="text/csv"
        )

    elif page == "Filters":

        st.header("Filter Patient Records")
        st.caption("Choose criteria below. Filters apply to the table on this page.")

        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            selected_outcome = st.selectbox(
                "Patient outcome",
                ["All outcomes", "Survived", "Deceased"]
            )
        with filter_col2:
            selected_sex = st.selectbox(
                "Sex",
                ["All", "Male", "Female"]
            )

        page_filtered = filtered.copy()
        if selected_outcome == "Survived":
            page_filtered = page_filtered[page_filtered["DEATH_EVENT"] == 0]
        elif selected_outcome == "Deceased":
            page_filtered = page_filtered[page_filtered["DEATH_EVENT"] == 1]

        if selected_sex == "Male":
            page_filtered = page_filtered[page_filtered["sex"] == 1]
        elif selected_sex == "Female":
            page_filtered = page_filtered[page_filtered["sex"] == 0]

        st.metric("Matching patients", len(page_filtered))
        st.dataframe(
            page_filtered[
                ["age", "sex", "ejection_fraction", "serum_creatinine", "time", "DEATH_EVENT"]
            ].rename(columns={"sex": "Sex (1=Male)", "DEATH_EVENT": "Deceased"}),
            width="stretch",
            hide_index=True
        )

        csv_data = page_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered records",
            data=csv_data,
            file_name="heart_failure_filtered.csv",
            mime="text/csv"
        )

    st.stop()


    # ============================================================
    # KPI CALCULATIONS
# ============================================================

total_patients = len(filtered)

deaths = int(
    filtered["DEATH_EVENT"].sum()
)

male_count = int(filtered["sex"].eq(1).sum())
female_count = int(filtered["sex"].eq(0).sum())

survivors = (
    total_patients - deaths
)

survival_rate = (
    survivors / total_patients * 100
    if total_patients > 0
    else 0
)

avg_survivor_age = (
    filtered.loc[
        filtered["DEATH_EVENT"] == 0,
        "age"
    ].mean()
    if survivors > 0
    else 0
)

avg_deceased_age = (
    filtered.loc[
        filtered["DEATH_EVENT"] == 1,
        "age"
    ].mean()
    if deaths > 0
    else 0
)


# ============================================================
# MAIN CONTENT
# ============================================================

left_col, right_col = st.columns(
    [3, 7]
)


# ============================================================
# LEFT HEART PANEL
# ============================================================

with left_col:

    with st.container(border=True):

        if HEART_PATH.exists():

            st.image(
                str(HEART_PATH),
                width="stretch"
            )

        else:

            st.markdown(
                """
                <div style="font-size:120px; text-align:center; padding:80px 0;">
                ❤️
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            f"""
            <div class="patient-network" role="group" aria-label="Patient population by sex">
              <div class="patient-connector-stem"></div>
              <div class="patient-connector-branch"></div>
              <div class="patient-avatars">
                <div class="patient-node">
                  <svg class="patient-avatar" viewBox="0 0 100 100" role="img" aria-label="Male patient">
                    <circle cx="50" cy="50" r="47" fill="#bd1738" stroke="#ffffff" stroke-width="5"/>
                    <path d="M23 83c2-14 12-22 27-22s25 8 27 22" fill="#52616b"/>
                    <path d="M36 59h28v9H36z" fill="#d6a078"/>
                    <path d="M31 40c0-17 8-26 19-26 13 0 21 10 20 27l-5 16H36z" fill="#503b32"/>
                    <ellipse cx="50" cy="43" rx="17" ry="19" fill="#e6b991"/>
                    <path d="M33 39c1-17 8-25 19-25 10 0 17 7 19 21-8-2-15-6-20-12-5 7-11 12-18 16" fill="#503b32"/>
                    <rect x="36" y="39" width="12" height="9" rx="3" fill="none" stroke="#39434b" stroke-width="2"/>
                    <rect x="52" y="39" width="12" height="9" rx="3" fill="none" stroke="#39434b" stroke-width="2"/>
                    <path d="M48 43h4" stroke="#39434b" stroke-width="2"/>
                  </svg>
                  <div class="patient-label">Male</div>
                  <div class="patient-count">{male_count}</div>
                </div>
                <div class="patient-node">
                  <svg class="patient-avatar" viewBox="0 0 100 100" role="img" aria-label="Female patient">
                    <circle cx="50" cy="50" r="47" fill="#bd1738" stroke="#ffffff" stroke-width="5"/>
                    <path d="M23 84c2-14 12-22 27-22s25 8 27 22" fill="#f1f2f3"/>
                    <path d="M37 57h26v10H37z" fill="#d6a078"/>
                    <path d="M30 43c-2-18 7-30 21-30 17 0 23 14 20 34l-2 17-8-7H38l-7 8z" fill="#49352e"/>
                    <ellipse cx="50" cy="43" rx="16" ry="19" fill="#e6b991"/>
                    <path d="M34 39c1-16 8-25 18-25 12 0 19 8 20 24-7-3-14-8-19-14-5 7-11 12-19 15" fill="#49352e"/>
                    <path d="M44 47h2m12 0h2" stroke="#49352e" stroke-width="2.5" stroke-linecap="round"/>
                    <path d="M46 54q4 3 8 0" fill="none" stroke="#a45f54" stroke-width="2" stroke-linecap="round"/>
                  </svg>
                  <div class="patient-label">Female</div>
                  <div class="patient-count">{female_count}</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# RIGHT CONTENT
# ============================================================

with right_col:

    # KPI cards sit alongside the patient illustration, as in the reference layout.
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(
            f"""
            <div class="kpi-container">
            <div class="kpi-title">Survival Rate</div>
            <div class="kpi-value">{survival_rate:.2f}%</div>
            <div class="kpi-description">{total_patients} patients in view</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with k2:
        st.markdown(
            f"""
            <div class="kpi-container">
            <div class="kpi-title">Avg Age of Survivors</div>
            <div class="kpi-value">{avg_survivor_age:.2f}</div>
            <div class="kpi-description">years · deceased avg {avg_deceased_age:.1f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with k3:
        st.markdown(
            f"""
            <div class="kpi-container">
            <div class="kpi-title">Total Survived</div>
            <div class="kpi-value">{survivors}</div>
            <div class="kpi-description">patients alive at last visit</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with k4:
        st.markdown(
            f"""
            <div class="kpi-container">
            <div class="kpi-title">Total Deaths</div>
            <div class="kpi-value-danger">{deaths}</div>
            <div class="kpi-description">deaths during follow-up</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # CHART 1
    # --------------------------------------------------------

    c1, c2 = st.columns(2)


    with c1:

        st.markdown(
            "### Survival Count and Avg Serum Creatinine by Age Group"
        )

        age_stats = (
            filtered
            .groupby(
                "Age Group",
                observed=False
            )
            .agg(
                Survivors=(
                    "DEATH_EVENT",
                    lambda x: (x == 0).sum()
                ),
                Avg_Creatinine=(
                    "serum_creatinine",
                    "mean"
                )
            )
            .reset_index()
        )

        fig1 = go.Figure()


        # Survivor bars

        fig1.add_trace(
            go.Bar(
                x=age_stats["Age Group"],
                y=age_stats["Survivors"],
                name="Survivors",
                marker_color="#bd1738",
                text=age_stats["Survivors"],
                textposition="outside"
            )
        )


        # Creatinine line

        fig1.add_trace(
            go.Scatter(
                x=age_stats["Age Group"],
                y=age_stats["Avg_Creatinine"],
                name="Avg serum creatinine",
                mode="lines+markers+text",
                text=age_stats["Avg_Creatinine"].round(2),
                textposition="top center",
                line=dict(
                    color="#f27a32",
                    width=3
                ),
                marker=dict(
                    size=10
                ),
                yaxis="y2"
            )
        )


        fig1.update_layout(

            height=390,

            template="plotly_white",

            plot_bgcolor="white",

            paper_bgcolor="white",

            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20
            ),

            yaxis=dict(
                title="Patients",
                gridcolor="#eeeeee"
            ),

            yaxis2=dict(
                title="Creatinine (mg/dL)",
                overlaying="y",
                side="right"
            ),

            legend=dict(
                orientation="h",
                y=1.1
            )
        )


        st.plotly_chart(
            fig1,
            width="stretch",
            theme=None
        )


    # --------------------------------------------------------
    # CHART 2
    # --------------------------------------------------------

    with c2:

        st.markdown(
            "### Survival Count and Avg Ejection Fraction by Age Group"
        )


        ef_stats = (
            filtered
            .groupby(
                "Age Group",
                observed=False
            )
            .agg(
                Survivors=(
                    "DEATH_EVENT",
                    lambda x: (x == 0).sum()
                ),
                Avg_EF=(
                    "ejection_fraction",
                    "mean"
                )
            )
            .reset_index()
        )


        fig2 = go.Figure()


        fig2.add_trace(
            go.Bar(
                x=ef_stats["Age Group"],
                y=ef_stats["Survivors"],
                name="Survivors",
                marker_color="#bd1738",
                text=ef_stats["Survivors"],
                textposition="outside"
            )
        )


        fig2.add_trace(
            go.Scatter(
                x=ef_stats["Age Group"],
                y=ef_stats["Avg_EF"],
                name="Avg ejection fraction",
                mode="lines+markers+text",
                text=ef_stats["Avg_EF"].round(1),
                textposition="top center",
                line=dict(
                    color="#f27a32",
                    width=3
                ),
                marker=dict(
                    size=10
                ),
                yaxis="y2"
            )
        )


        fig2.update_layout(

            height=390,

            template="plotly_white",

            plot_bgcolor="white",

            paper_bgcolor="white",

            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20
            ),

            yaxis=dict(
                title="Patients",
                gridcolor="#eeeeee"
            ),

            yaxis2=dict(
                title="Ejection Fraction (%)",
                overlaying="y",
                side="right"
            ),

            legend=dict(
                orientation="h",
                y=1.1
            )
        )


        st.plotly_chart(
            fig2,
            width="stretch",
            theme=None
        )


# ============================================================
# SURVIVAL ANALYSIS SECTION
# ============================================================

st.markdown("---")

st.markdown(
    "## 📈 Survival Analysis"
)


survival_col1, survival_col2 = st.columns(2)


# ============================================================
# SURVIVAL BY AGE
# ============================================================

with survival_col1:

    st.markdown(
        "### Survival Rate by Age Group"
    )


    age_survival = (
        df.groupby(
            "Age Group",
            observed=False
        )["DEATH_EVENT"]
        .agg(
            ["count", "sum"]
        )
        .reset_index()
    )


    age_survival["Survival Rate"] = (
        1 -
        age_survival["sum"]
        / age_survival["count"]
    ) * 100


    fig3 = go.Figure()


    fig3.add_trace(
        go.Bar(
            x=age_survival["Age Group"],
            y=age_survival["Survival Rate"],
            marker_color="#bd1738",
            text=age_survival["Survival Rate"].round(1),
            texttemplate="%{text}%",
            textposition="outside"
        )
    )


    fig3.update_layout(
        height=350,
        template="plotly_white",
        yaxis=dict(
            title="Survival Rate (%)",
            range=[0, 100]
        ),
        plot_bgcolor="white"
    )


    st.plotly_chart(
        fig3,
        width="stretch",
        theme=None
    )


# ============================================================
# CLINICAL INDICATORS
# ============================================================

with survival_col2:

    st.markdown(
        "### Key Clinical Indicators"
    )


    clinical_data = {

        "Ejection Fraction":
        filtered["ejection_fraction"].mean(),

        "Serum Sodium":
        filtered["serum_sodium"].mean(),

        "Serum Creatinine":
        filtered["serum_creatinine"].mean()
    }


    clinical_df = pd.DataFrame(
        {
            "Indicator":
            list(clinical_data.keys()),

            "Average":
            list(clinical_data.values())
        }
    )


    fig4 = go.Figure()


    fig4.add_trace(
        go.Bar(
            x=clinical_df["Indicator"],
            y=clinical_df["Average"],
            marker_color="#bd1738",
            text=clinical_df["Average"].round(2),
            textposition="outside"
        )
    )


    fig4.update_layout(
        height=350,
        template="plotly_white",
        plot_bgcolor="white"
    )


    st.plotly_chart(
        fig4,
        width="stretch",
        theme=None
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="dashboard-footer">

    Heart Failure Survival Analysis |
    Healthcare Data Analytics

    <br>

    Educational and analytical purposes only.
    Not intended for clinical decision-making.

    </div>
    """,
    unsafe_allow_html=True
)