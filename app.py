# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import json

# Set page configuration
st.set_page_config(
    page_title="UCDP Conflict Analysis",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>stra
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


class ConflictAnalyzer:
    def __init__(self, data):
        self.data = data
        self.setup_data()

    def setup_data(self):
        """Initialize and clean the data"""
        self.data['Civilian_Ratio'] = (self.data['Deaths_Civilians'] / self.data['Total_Deaths']) * 100
        self.data['Battle_Ratio'] = (self.data['Deaths_Battle'] / self.data['Total_Deaths']) * 100

    def get_summary_stats(self):
        """Calculate summary statistics"""
        stats = {
            'total_deaths': self.data['Total_Deaths'].sum(),
            'total_conflicts': len(self.data),
            'avg_deaths_per_conflict': self.data['Total_Deaths'].mean(),
            'civilian_ratio': (self.data['Deaths_Civilians'].sum() / self.data['Total_Deaths'].sum()) * 100,
            'countries_affected': self.data['Country'].nunique(),
            'regions_affected': self.data['Region'].nunique()
        }
        return stats


def main():
    # Title and introduction
    st.markdown('<h1 class="main-header">⚔️ UCDP Conflict Data Analysis Dashboard</h1>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("Navigation")
    section = st.sidebar.radio("Go to:", [
        "Project Overview",
        "Data Exploration",
        "Visual Analysis",
        "Geographic Analysis",
        "Python Concepts Demo"
    ])

    # Load data
    data = pd.read_csv('ucdp_cleaned_conflict_data.csv')

    df = pd.DataFrame(data)
    analyzer = ConflictAnalyzer(df)

    if section == "Project Overview":
        show_project_overview(df, analyzer)
    elif section == "Data Exploration":
        show_data_exploration(df)
    elif section == "Visual Analysis":
        show_visual_analysis(df)
    elif section == "Geographic Analysis":
        show_geographic_analysis(df)
    elif section == "Python Concepts Demo":
        show_python_concepts_demo(df)


def show_project_overview(df, analyzer):
    """Display project overview and summary statistics"""

    st.markdown('<h2 class="section-header">📊 Project Overview</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("""
        This project analyzes armed conflict data from the Uppsala Conflict Data Program (UCDP) 
        for the period 2018-2021. The dashboard demonstrates various Python programming concepts 
        covered in the syllabus including data analysis, visualization, and Streamlit web development.

        **Dataset Features:**
        - Temporal analysis (2018-2021)
        - Regional conflict distribution
        - Conflict type categorization
        - Civilian vs battle death analysis
        - Geographic visualization
        """)

    with col2:
        stats = analyzer.get_summary_stats()
        st.metric("Total Deaths", f"{stats['total_deaths']:,}")
        st.metric("Total Conflicts", stats['total_conflicts'])
        st.metric("Countries Affected", stats['countries_affected'])

    # Key metrics in cards
    st.markdown('<h3 class="section-header">📈 Key Metrics</h3>', unsafe_allow_html=True)

    cols = st.columns(4)
    stats = analyzer.get_summary_stats()

    with cols[0]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Average Deaths/Conflict", f"{stats['avg_deaths_per_conflict']:.0f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with cols[1]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Civilian Death Ratio", f"{stats['civilian_ratio']:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    with cols[2]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Regions Affected", stats['regions_affected'])
        st.markdown('</div>', unsafe_allow_html=True)

    with cols[3]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        deadliest_year = df.groupby('Year')['Total_Deaths'].sum().idxmax()
        st.metric("Deadliest Year", deadliest_year)
        st.markdown('</div>', unsafe_allow_html=True)

    # Quick data preview
    st.markdown('<h3 class="section-header">🔍 Data Preview</h3>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)


def show_data_exploration(df):
    """Interactive data exploration section"""

    st.markdown('<h2 class="section-header">🔎 Data Exploration</h2>', unsafe_allow_html=True)

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        years = st.multiselect("Select Years", options=sorted(df['Year'].unique()),
                               default=sorted(df['Year'].unique()))

    with col2:
        regions = st.multiselect("Select Regions", options=df['Region'].unique(),
                                 default=df['Region'].unique())

    with col3:
        conflict_types = st.multiselect("Select Conflict Types", options=df['Conflict_Type'].unique(),
                                        default=df['Conflict_Type'].unique())

    # Apply filters
    filtered_df = df[
        (df['Year'].isin(years)) &
        (df['Region'].isin(regions)) &
        (df['Conflict_Type'].isin(conflict_types))
        ]

    # Display filtered data
    st.dataframe(filtered_df, use_container_width=True)

    # Statistical summary
    st.markdown('<h3 class="section-header">📊 Statistical Summary</h3>', unsafe_allow_html=True)

    if not filtered_df.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Numerical Summary:**")
            st.write(filtered_df[['Deaths_Battle', 'Deaths_Civilians', 'Total_Deaths']].describe())

        with col2:
            st.write("**Categorical Summary:**")
            st.write(f"Countries: {', '.join(filtered_df['Country'].unique())}")
            st.write(f"Conflict Types: {', '.join(filtered_df['Conflict_Type'].unique())}")
            st.write(f"Total Records: {len(filtered_df)}")


def show_visual_analysis(df):
    """Interactive visualizations"""

    st.markdown('<h2 class="section-header">📊 Visual Analysis</h2>', unsafe_allow_html=True)

    # Chart type selection
    chart_type = st.selectbox("Select Chart Type", [
        "Death Trends Over Time",
        "Regional Analysis",
        "Conflict Type Distribution",
        "Civilian vs Battle Deaths",
        "Country Comparison"
    ])

    if chart_type == "Death Trends Over Time":
        fig = px.line(df.groupby('Year').agg({'Total_Deaths': 'sum'}).reset_index(),
                      x='Year', y='Total_Deaths', title='Total Deaths Over Time',
                      markers=True)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Regional Analysis":
        regional_data = df.groupby('Region').agg({'Total_Deaths': 'sum'}).reset_index()
        fig = px.bar(regional_data, x='Region', y='Total_Deaths',
                     title='Total Deaths by Region', color='Region')
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Conflict Type Distribution":
        conflict_data = df.groupby('Conflict_Type').agg({'Total_Deaths': 'sum'}).reset_index()
        fig = px.pie(conflict_data, values='Total_Deaths', names='Conflict_Type',
                     title='Death Distribution by Conflict Type')
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Civilian vs Battle Deaths":
        death_types = df[['Deaths_Battle', 'Deaths_Civilians']].sum()
        fig = px.bar(x=death_types.index, y=death_types.values,
                     title='Battle vs Civilian Deaths',
                     labels={'x': 'Death Type', 'y': 'Number of Deaths'})
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Country Comparison":
        country_data = df.groupby('Country').agg({'Total_Deaths': 'sum'}).reset_index()
        fig = px.bar(country_data.sort_values('Total_Deaths', ascending=False),
                     x='Country', y='Total_Deaths', title='Total Deaths by Country',
                     color='Total_Deaths')
        st.plotly_chart(fig, use_container_width=True)

    # Additional interactive analysis
    st.markdown('<h3 class="section-header">🔍 Interactive Analysis</h3>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        selected_country = st.selectbox("Select Country for Detailed Analysis", df['Country'].unique())
        country_data = df[df['Country'] == selected_country]

        if not country_data.empty:
            st.write(f"**{selected_country} Conflict History:**")
            st.dataframe(country_data)

    with col2:
        selected_year = st.slider("Select Year for Analysis",
                                  min_value=int(df['Year'].min()),
                                  max_value=int(df['Year'].max()),
                                  value=2020)
        year_data = df[df['Year'] == selected_year]

        if not year_data.empty:
            st.write(f"**Conflicts in {selected_year}:**")
            st.write(f"Total Deaths: {year_data['Total_Deaths'].sum():,}")
            st.write(f"Countries Affected: {len(year_data)}")


def show_geographic_analysis(df):
    """Geographic visualization of conflicts"""

    st.markdown('<h2 class="section-header">🌍 Geographic Analysis</h2>', unsafe_allow_html=True)

    # Create Folium map
    m = folium.Map(location=[20, 40], zoom_start=3)

    # Add conflict markers
    for idx, row in df.iterrows():
        # Determine color based on conflict type
        color_map = {
            'State-based': 'red',
            'Non-state': 'blue',
            'One-sided violence': 'orange'
        }

        popup_text = f"""
        <b>Country:</b> {row['Country']}<br>
        <b>Year:</b> {row['Year']}<br>
        <b>Conflict Type:</b> {row['Conflict_Type']}<br>
        <b>Total Deaths:</b> {row['Total_Deaths']}<br>
        <b>Actors:</b> {row['Actors']}
        """

        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=row['Total_Deaths'] / 50 + 5,
            popup=folium.Popup(popup_text, max_width=300),
            color=color_map.get(row['Conflict_Type'], 'gray'),
            fill=True,
            fillOpacity=0.7,
            tooltip=f"{row['Country']} ({row['Year']})"
        ).add_to(m)

    # Display the map
    st_data = st_folium(m, width=1000, height=500)

    # Conflict hotspot analysis
    st.markdown('<h3 class="section-header">🔥 Conflict Hotspots</h3>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Most affected countries
        country_hotspots = df.groupby('Country')['Total_Deaths'].sum().nlargest(5)
        st.write("**Top 5 Most Affected Countries:**")
        for country, deaths in country_hotspots.items():
            st.write(f"• {country}: {deaths:,} deaths")

    with col2:
        # Most violent years
        year_hotspots = df.groupby('Year')['Total_Deaths'].sum().nlargest(3)
        st.write("**Top 3 Most Violent Years:**")
        for year, deaths in year_hotspots.items():
            st.write(f"• {year}: {deaths:,} deaths")


def show_python_concepts_demo(df):
    """Demonstrate Python concepts from the syllabus"""

    st.markdown('<h2 class="section-header">🐍 Python Concepts Demonstration</h2>', unsafe_allow_html=True)

    concept = st.selectbox("Select Python Concept to Demonstrate", [
        "Data Types & Variables",
        "Lists & Dictionaries",
        "Functions & Lambda",
        "Pandas Operations",
        "Data Visualization"
    ])

    if concept == "Data Types & Variables":
        st.code('''
# Python Data Types Demonstration
year = 2020  # Integer
country = "Syria"  # String
deaths = 1500.0  # Float
is_conflict = True  # Boolean

print(f"Data Types: {type(year)}, {type(country)}, {type(deaths)}, {type(is_conflict)}")
        ''', language='python')

        # Live demonstration
        year = 2020
        country = "Syria"
        deaths = 1500.0
        is_conflict = True

        st.write("**Live Output:**")
        st.write(f"Data Types: {type(year)}, {type(country)}, {type(deaths)}, {type(is_conflict)}")

    elif concept == "Lists & Dictionaries":
        st.code('''
# Lists and Dictionaries with Conflict Data
countries = ["India", "Pakistan", "Afghanistan", "Syria"]
conflict_data = {
    'India': [150, 190, 70],
    'Syria': [1500],
    'Afghanistan': [800]
}

# List comprehension to get countries with high deaths
high_death_countries = [country for country in countries 
                       if sum(conflict_data.get(country, [0])) > 100]

print(f"Countries with >100 deaths: {high_death_countries}")
        ''', language='python')

        # Live demonstration
        countries = ["India", "Pakistan", "Afghanistan", "Syria"]
        conflict_data = {
            'India': [150, 190, 70],
            'Syria': [1500],
            'Afghanistan': [800]
        }

        high_death_countries = [country for country in countries
                                if sum(conflict_data.get(country, [0])) > 100]

        st.write("**Live Output:**")
        st.write(f"Countries with >100 deaths: {high_death_countries}")

    elif concept == "Functions & Lambda":
        st.code('''
# Functions and Lambda expressions
def calculate_civilian_ratio(battle_deaths, civilian_deaths):
    """Calculate civilian death ratio"""
    total = battle_deaths + civilian_deaths
    return (civilian_deaths / total) * 100 if total > 0 else 0

# Using lambda for quick calculations
death_calculator = lambda b, c: b + c

# Apply to dataset
sample_row = df.iloc[0]
ratio = calculate_civilian_ratio(sample_row['Deaths_Battle'], 
                                sample_row['Deaths_Civilians'])
total = death_calculator(sample_row['Deaths_Battle'], 
                        sample_row['Deaths_Civilians'])

print(f"Civilian Ratio: {ratio:.1f}%, Total Deaths: {total}")
        ''', language='python')

        # Live demonstration
        def calculate_civilian_ratio(battle_deaths, civilian_deaths):
            total = battle_deaths + civilian_deaths
            return (civilian_deaths / total) * 100 if total > 0 else 0

        death_calculator = lambda b, c: b + c

        sample_row = df.iloc[0]
        ratio = calculate_civilian_ratio(sample_row['Deaths_Battle'], sample_row['Deaths_Civilians'])
        total = death_calculator(sample_row['Deaths_Battle'], sample_row['Deaths_Civilians'])

        st.write("**Live Output:**")
        st.write(f"Civilian Ratio: {ratio:.1f}%, Total Deaths: {total}")

    elif concept == "Pandas Operations":
        st.code('''
# Pandas DataFrame Operations
# Filtering data
state_based_conflicts = df[df['Conflict_Type'] == 'State-based']

# Grouping and aggregation
yearly_deaths = df.groupby('Year')['Total_Deaths'].sum()

# Adding new columns
df['Civilian_Ratio'] = (df['Deaths_Civilians'] / df['Total_Deaths']) * 100

print("State-based conflicts:")
print(state_based_conflicts[['Country', 'Year', 'Total_Deaths']])
print("\\nYearly Deaths:")
print(yearly_deaths)
        ''', language='python')

        # Live demonstration
        state_based_conflicts = df[df['Conflict_Type'] == 'State-based']
        yearly_deaths = df.groupby('Year')['Total_Deaths'].sum()

        st.write("**State-based conflicts:**")
        st.dataframe(state_based_conflicts[['Country', 'Year', 'Total_Deaths']])
        st.write("**Yearly Deaths:**")
        st.write(yearly_deaths)

    elif concept == "Data Visualization":
        st.code('''
# Data Visualization with Matplotlib and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Create a bar plot of deaths by country
plt.figure(figsize=(10, 6))
country_deaths = df.groupby('Country')['Total_Deaths'].sum().sort_values(ascending=False)
sns.barplot(x=country_deaths.values, y=country_deaths.index)
plt.title('Total Deaths by Country')
plt.xlabel('Total Deaths')
plt.tight_layout()
plt.show()
        ''', language='python')

        # Live demonstration
        fig, ax = plt.subplots(figsize=(10, 6))
        country_deaths = df.groupby('Country')['Total_Deaths'].sum().sort_values(ascending=False)
        sns.barplot(x=country_deaths.values, y=country_deaths.index, ax=ax)
        ax.set_title('Total Deaths by Country')
        ax.set_xlabel('Total Deaths')
        plt.tight_layout()
        st.pyplot(fig)


if __name__ == "__main__":
    main()