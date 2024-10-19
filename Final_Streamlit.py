#Importing Libraries
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import base64
import seaborn as sns
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import altair as alt
from sklearn.feature_selection import mutual_info_classif
import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import statsmodels.api as sm

# Reading Dataframe
# Winemag
winemag1 = pd.read_csv('Datas/winemg-data/winemag-data_first150k.csv', index_col=0)
winemag2 = pd.read_csv('Datas/winemg-data/winemag-data_first150k.csv', index_col=0)
winemag = pd.concat([winemag1, winemag2], ignore_index=True)
winemag= winemag.drop_duplicates()
# Global Wines
global_wines= pd.read_excel('Datas/global_wines/Wines.xlsx')
global_wines.columns = global_wines.columns.str.lower()
df_wine = pd.merge(winemag, global_wines, on=['province', 'variety', 'winery'], how='inner')
# Dropping irrelevant columns and editing the proce column to float
df_wine_copy1= df_wine.copy()
df_wine_copy1['price'] = df_wine_copy1['price'].replace('[\$,]', '', regex=True).astype(float)
df_wine_copy1.drop(['designation_x','designation_y', 'region_2', 'county'], axis=1, inplace= True)

# Imputing values based on most frequently occuring values in a group of a dataset
grouping_cols = ['province', 'variety', 'winery', 'country']
for col in df_wine_copy1.columns:
    if df_wine_copy1[col].isnull().any():  # Check if the column has missing values
        most_frequent = df_wine_copy1.groupby(grouping_cols)[col].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
        # Map the most frequent values back to the original DataFrame
        df_wine_copy1[col] = df_wine_copy1[col].fillna(df_wine_copy1[grouping_cols].agg(tuple, axis=1).map(most_frequent))

# Apply label encoder
df_wine_copy = df_wine_copy1.copy()
label_encoders = {}
categorical_cols = ['description', 'province', 'region_1', 'variety', 'winery', 'vintage', 'country', 'title']

for col in categorical_cols:
    le = LabelEncoder()
    df_wine_copy[col] = le.fit_transform(df_wine_copy[col].astype(str))
    label_encoders[col] = le


columns_for_imputation = ['region_1', 'price']
# Applying KNN Imputation for missing values
imputer = KNNImputer(n_neighbors=5)
df_wine_copy[columns_for_imputation] = imputer.fit_transform(df_wine_copy[columns_for_imputation])

df= pd.read_csv('Datas/Final_Dataset.csv').iloc[:,1:]
df['vintage'] = pd.to_datetime(df['vintage'], errors='coerce')
df['vintage_year'] = df['vintage'].dt.year




# Set Streamlit app config for a wider layout and light theme
st.set_page_config(layout="wide", page_title="Wine Recommendation App", initial_sidebar_state="expanded")

# Set background image using HTML and CSS
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        base64_image = base64.b64encode(image_data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/webp;base64,{base64_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
 
# Background image
pic = 'pics/wine-gone-bad.jpg'
set_background(pic)

# Page navigation state
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a Section", ["Introduction", "Overview", "Visualization", "Comparison"])

# Function to load different sections
if options == "Introduction":
    st.markdown(
        "<h1 style='color: red;'>Introduction</h1>", 
        unsafe_allow_html=True
    )
    

    # First Text Block: Introduction to Wine
    st.markdown("""
    ### Let’s savor the timeless pour of the perfect red

    Wine is an alcoholic beverage made from fermented grapes or other fruits. Its origins trace back to around **6000 BC** in regions of modern-day Georgia, Iran, and Armenia. Early wine production spread to ancient Egypt and Mesopotamia, where it became a staple in religious ceremonies and royal banquets.

    The **Greeks and Romans** played key roles in advancing winemaking techniques and expanding wine culture. The Romans, in particular, cultivated vineyards across their empire, spreading wine throughout Europe, including present-day France, Spain, and Italy, regions that later became famous for their wine production.

    As European explorers and settlers ventured to the New World, they introduced wine to regions like **South Africa, Australia, and the Americas**. Wine took root in **California, Argentina, and Chile**, now major global wine producers. Today, wine is produced and enjoyed worldwide, with each region adding its own unique flavors, traditions, and styles to the global wine landscape.
    """)

    # Embed the YouTube video
    st.markdown("#### Learn More About Wine")
    components.iframe("https://www.youtube.com/embed/hGJWUg4wx78", width=560, height=315)

    # Second Text Block: Regional and Vintage Differences in Wine
    st.markdown("""
    ### Regional and Vintage Differences in Wine

    Wines originating from the same grape variety but different regions and vintages can vary significantly in taste due to several factors:

    **Terroir (Region-Specific Factors):**

    - **Climate:** Warmer regions produce wines with riper, bolder flavors (e.g., more fruit-forward), while cooler regions create more acidic and lighter wines. Soil type, altitude, and sunlight also influence flavor nuances.
    - **Soil Composition:** Different minerals in the soil can impart subtle flavors, such as earthy, mineral, or even floral notes.

    **Vintage (Year of Harvest):**

    - **Weather Conditions:** The growing season's temperature, rainfall, and sunshine affect the grapes' ripeness and sugar levels. A sunny year can produce a rich, full-bodied wine, while a cooler, wetter year might result in higher acidity and lighter body.
    - **Aging Potential:** Different vintages age differently, leading to changes in tannin structure, acidity, and overall complexity over time.

    **Winemaking Practices:**

    The winemaker’s approach, including fermentation techniques, use of oak barrels, and aging methods, can also introduce variations in flavor, even within the same variety.

    Ultimately, regional and vintage differences can create a broad spectrum of flavors, from fruity and bold to subtle, nuanced, and earthy, even in wines made from the same type of grape.
    """)
# Overview page
elif options == "Overview":
    st.markdown(
        "<h1 style='color: red;'>Overview</h1>", 
        unsafe_allow_html=True
    )
    st.markdown("""
    ### About the Data

    #### Source
    - **Winemag**: Winemag is a platform dedicated to South African fine wines. Initially published as a magazine from 1993 to 2011, it transitioned to a digital format, offering comprehensive wine reviews, ratings, and industry insights. The site features over 10,000 wine ratings, including the latest evaluations of local wines, such as pinot noir and shiraz, with an emphasis on detailed analysis by wine experts.
    - **Global Wines**: Global Wines is a wine and spirits importer and distributor based in Turkey. They focus on high-quality wines from family-owned estates and distilleries in countries like the USA, Spain, New Zealand, and Australia. Known for their diverse portfolio, they offer kosher wines and emphasize ethical practices, innovation, and customer service. Their mission is to provide an exceptional selection while fostering strong relationships within the industry.

    #### Tables
    - **Winemag**: Contains descriptions of different types of wines worldwide.
    - **Global Wines**: Contains the wine prices, timeline, points, and many other details.
    """)


    # Dropdown filter for selecting dataset
    st.markdown(""" ### Wine Data Summary""")
    option = st.selectbox(
        "Select a dataset to view details:",
        ("Winemag", "Global Wines")
    )

    # Data description for Winemag
    winemag_data = {
        "Column Name": ["Description", "Designation", "Province", "Region 1", "Region 2", "Variety", "Winery"],
        "Description": [
            "A detailed description of the wine.",
            "The vineyard or brand designation for the wine.",
            "The specific province or region within a country where the wine was produced.",
            "A more specific sub-region where the wine was produced.",
            "A secondary, more specific sub-region.",
            "The type or variety of grape used in the wine (e.g., Sparkling Blend, Sangiovese).",
            "Name of the winery that produced the wine."
        ]
    }

    # Data description for Global Wines
    global_wines_data = {
        "Column Name": ["Vintage", "Country", "County", "Designation", "Points", "Price", "Title", "Variety", "Winery"],
        "Description": [
            "The year of the wine's production or bottling.",
            "The country where the wine originates.",
            "The specific county where the wine is produced.",
            "The specific vineyard or designation given to the wine by the producer.",
            "The score or rating given to the wine, usually ranging from 0-100.",
            "The price of the wine in dollars.",
            "The full title of the wine, which typically includes the brand and variety.",
            "The type or variety of grape used in the wine (e.g., Sparkling Blend, Sangiovese).",
            "The name of the winery or producer of the wine."
        ]
    }

    # Conditional rendering based on the selected option
    if option == "Winemag":
        st.markdown("##### **Winemag Data**")
        st.table(winemag_data)  # Displaying Winemag Data Table
    elif option == "Global Wines":
        st.markdown("##### **Global Wines Data**")
        st.table(global_wines_data)  # Displaying Global Wines Data Table

    # Set up two columns for side-by-side heatmaps
    col1, col2 = st.columns(2)

    # First heatmap for Global Wines
    with col1:
        st.write("##### Global Wines - Missing Value Heatmap")
        plt.figure(figsize=(4, 8))
        sns.heatmap(global_wines.isna(), cmap="magma", cbar=False)
        st.pyplot(plt.gcf())  # Render the current figure

    # Second heatmap for Winemag
    with col2:
        st.write("##### Winemag - Missing Value Heatmap")
        plt.figure(figsize=(4, 8))
        sns.heatmap(winemag.isna(), cmap="magma", cbar=False)
        st.pyplot(plt.gcf())  # Render the current figure

    st.markdown(""" 
                ### Data Cleaning and Merging

                Merged the dataset using 'province', 'variety' & 'winery' columns
                
                Dropped the columns 'designation_x','designation_y', 'region_2', 'county' as it contained ambigious data  """)
    
    st.write("##### Merged Dataset - Missing Value Heatmap")
    plt.figure(figsize=(8,3))
    sns.heatmap(df_wine.isna(), cmap="magma", cbar=False, cbar_kws={'orientation': 'horizontal'})
    st.pyplot(plt.gcf())  # Render the current figure

    st.markdown(""" 
                ##### Basic Imputation

                Grouped by  'province', 'variety', 'winery'and 'country' to find the most frequent occuring value for the missing column and filling it accordingly """)

    # First imputation using frequency
    st.write("##### Imputed with groupby frequency - Missing Value Heatmap")
    plt.figure(figsize=(8,3))
    sns.heatmap(df_wine_copy1.isna(), cmap="magma", cbar=False, cbar_kws={'orientation': 'horizontal'})
    st.pyplot(plt.gcf())  # Render the current figure

    st.markdown(""" 
                ##### Label Encoding & KNN imputation

                Applied KNN imputation to fill the rest of the missing values after applying label encoder to the categorical columns and de-encoded the variables after the imputation is done """)

    # Final imputation using KNN
    st.write("##### KNN - Missing Value Heatmap")
    plt.figure(figsize=(8,3))
    sns.heatmap(df_wine_copy.isna(), cmap="magma", cbar=False, cbar_kws={'orientation': 'horizontal'})
    st.pyplot(plt.gcf())  # Render the current figure
    st.write('Thus there are no more missing values in the dataset.')

elif options == "Visualization":
    st.markdown(
        "<h1 style='color: red;'>Visualization</h1>", 
        unsafe_allow_html=True
    )
    st.markdown("""Lets View the final Dataset..""")

    # Custom CSS for dropdown boxes and slider
    st.markdown(
        """
        <style>
        .stSelectbox, .stSlider {
            background-color: #ffcccc !important;
            color: black;
        }
        .stSelectbox {
            padding: 0.5rem;
            border-radius: 0.25rem;
            border: 2px solid #ff0000;
        }
        .stSlider > div > div {
            color: #ff0000 !important;
        }
        .css-1aumxhk.e1fqkh3o4 {
            display: flex;
            justify-content: space-between;
        }
        .css-1aumxhk.e1fqkh3o4 > div {
            flex-grow: 0.25;
            margin-right: 1rem;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # Sidebar filters (country, variety, and province)
    with st.sidebar:
        st.write("## Filter Options")
        
        # Slider for year range (colored in red)
        year_slider = st.slider("Select Year Range", min_value=int(df["vintage_year"].min()), max_value=int(df["vintage_year"].max()), value=(1950, 2016))

        # Dropdowns for country, variety, and province
        variety = st.selectbox("variety", options=["All"] + list(df["variety"].unique()))
        country = st.selectbox("country", options=["All"] + list(df["country"].unique()))
        province = st.selectbox("province", options=["All"] + list(df["province"].unique()))

    # Filter the dataframe based on the selected values
    filtered_data = df[(df["vintage_year"] >= year_slider[0]) & (df["vintage_year"] <= year_slider[1])]

    # Apply dropdown filters only if a specific value is selected
    if country != "All":
        filtered_data = filtered_data[filtered_data["country"] == country]

    if variety != "All":
        filtered_data = filtered_data[filtered_data["variety"] == variety]

    if province != "All":
        filtered_data = filtered_data[filtered_data["province"] == province]

    # Display the filtered dataframe
    st.write("#### Final Wine Data")
    st.dataframe(filtered_data)

    # Relationship between different featres
    st.write("##### Relationship between different features: ")
    st.write("##### Categorical features: ")
    # Define categorical columns
    categorical_cols = ['description', 'province', 'region_1', 'variety', 'winery', 'vintage', 'country', 'title']

    # Create an empty DataFrame to store mutual information
    mutual_info = pd.DataFrame(index=categorical_cols, columns=categorical_cols)

    # Calculate mutual information for each pair of categorical columns
    for col1 in categorical_cols:
        for col2 in categorical_cols:
            if col1 != col2:
                mi = mutual_info_classif(df_wine_copy[[col1]], df_wine_copy[col2], discrete_features=True)
                mutual_info.loc[col1, col2] = mi[0]

    # Fill diagonal values with 0 (since mutual info with itself is irrelevant)
    mutual_info.fillna(0, inplace=True)

    # Convert the mutual information DataFrame to long format for Altair visualization
    mutual_info_long = mutual_info.reset_index().melt(id_vars='index')
    mutual_info_long.columns = ['Column1', 'Column2', 'Mutual_Information']

    # Create an interactive heatmap with Altair
    heatmap = alt.Chart(mutual_info_long).mark_rect().encode(
        x='Column1:O',
        y='Column2:O',
        color=alt.Color('Mutual_Information:Q', scale=alt.Scale(scheme='blues')),
        tooltip=['Column1', 'Column2', 'Mutual_Information']
    ).properties(
        width=800,
        height=800,
        title='Mutual Information Between Categorical Columns'
    ).interactive()

    # Display the heatmap in Streamlit
    st.altair_chart(heatmap)

    st.write("""##### Numerical features: """)
    # Fit and transform the data (scaling both 'points' and 'price')
    scaled_data = filtered_data[['points', 'price']].copy()
    scaled_data['log_price'] = np.log(scaled_data['price'])

    # Fit OLS regression using statsmodels
    X = sm.add_constant(scaled_data['points'])  # Add constant (intercept)
    ols_model = sm.OLS(scaled_data['log_price'], X).fit()  # Fit the model

    # Extract slope and intercept
    intercept, slope = ols_model.params

    # Print the slope and intercept
    st.write(f"Slope of ols line: {slope}")

    # Create a scatter plot with scaled data
    fig = px.scatter(
        scaled_data,
        x='points',  # Scaled points on the x-axis
        y='log_price',   # Log-transformed price on the y-axis
        title="Wine Points vs Log-Transformed Price",
        labels={'points': 'Wine Points', 'log_price': 'Log-Transformed Price'},
        trendline="ols"  # Optional: Add a trendline (Ordinary Least Squares regression)
    )

    # Show the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Vizualizations based on Region, price, pints and description

    st.markdown('#### Vizualizations based on Region, Price, Points and Description:')
    st.write("""##### Global Distribution: """)
    # Calculate the mean points by country
    country_avg_points = filtered_data.groupby('country')['points'].mean().reset_index()
    country_avg_points.columns = ['country', 'Mean points']

    # Define a maroon-pink color scale
    maroon_pink_scale = [
        [0.0, 'rgb(128, 0, 0)'],   # Maroon
        [0.2, 'rgb(165, 42, 42)'],  # Brown
        [0.3, 'rgb(178, 34, 34)'],  # Firebrick Red
        [0.4, 'rgb(205, 92, 92)'],  # Indian Red
        [0.5, 'rgb(219, 112, 147)'], # Pale Violet Red (Pinkish)
        [0.6, 'rgb(255, 105, 180)'], # Hot Pink
        [0.7, 'rgb(255, 182, 193)'], # Light Pink
        [0.8, 'rgb(255, 192, 203)'], # Pink
        [0.9, 'rgb(255, 228, 225)'], # Misty Rose
        [1.0, 'rgb(255, 240, 245)']  # Lavender Blush
    ]

    # Create an interactive map with Plotly
    fig = px.choropleth(
        country_avg_points,
        locations="country",
        locationmode="country names",  # Map the countries to their names
        color="Mean points",  # Color by the mean rating
        hover_name="country",  # Show country name when hovering
        hover_data={"Mean points": True},  # Show the mean rating when hovering
        title="Mean Wine Ratings by Country",
        color_continuous_scale=maroon_pink_scale  # Custom maroon-pink color scale
    )

    # Set some visual styles for the map
    fig.update_layout(
        geo=dict(
            showframe=False,  # No frame around the map
            showcoastlines=False,  # No coastlines
            projection_type='equirectangular'  # Flat map projection
        ),
        coloraxis_colorbar=dict(
            title="Mean points"
        )
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)

    st.write("""##### Link between variety and region: """)

    fig = px.parallel_categories(
        filtered_data.sort_values(by='points', ascending=False).head(100),
        dimensions=['variety', 'country', 'province', 'region_1'],  # Categorical dimensions for parallel plot
        color="points",  # Color by points
        color_continuous_scale=px.colors.sequential.Inferno  # Color scale for the plot
    )

    #  Update the layout for better visuals
    fig.update_layout(
        title="Parallel Plot of Wine Variety, Country, Province, and Region",
        plot_bgcolor='white',
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    st.write("""##### Price vs Points based on variety: """)
    st.write(""" Note: These are random samples from the overall dataset and may not fully represent the characteristics of the entire dataset. """)

    # Define the list of wine varieties for the dropdown
    varieties = df['variety'].unique().tolist()
    default_variety = 'Pinot Noir'  # Default selection

    # Create a dropdown for selecting the wine variety
    selected_variety = st.selectbox("Select Wine Variety:", varieties, index=varieties.index(default_variety))

    # Filter the data based on the selected variety
    filtered_data = df[df['variety'] == selected_variety].sample(n=1000, random_state=42)

    # Create a scatter plot for points vs. price, colored by country
    scatter = alt.Chart(filtered_data).mark_circle(size=60).encode(
        x=alt.X('points:Q', scale=alt.Scale(domain=[80, 100]), title='Wine Points'),
        y=alt.Y('price:Q', title='Wine Price'),
        color=alt.Color('country:N', title='Country'),  # Color by country
        tooltip=['points', 'price', 'country']
    ).properties(
        width=400,
        height=400,
        title=f'Joint Plot: Wine Points vs. Wine Price (Colored by Country) for {selected_variety}'
    )

    # Create density plot for the x-axis (points)
    x_density = alt.Chart(filtered_data).transform_density(
        'points',
        as_=['points', 'density'],
        bandwidth=0.5
    ).mark_area(
        opacity=0.5,
        color='lightblue'
    ).encode(
        x=alt.X('points:Q', title=None),
        y=alt.Y('density:Q', title=None),
    ).properties(
        width=400,
        height=100
    )

    # Create density plot for the y-axis (price)
    y_density = alt.Chart(filtered_data).transform_density(
        'price',
        as_=['price', 'density'],
        bandwidth=5
    ).mark_area(
        opacity=0.5,
        color='lightcoral'
    ).encode(
        y=alt.Y('density:Q', title=None),
        x=alt.X('price:Q', title=None),
    ).properties(
        width=100,
        height=400
    )

    # Combine the plots using the `hconcat` and `vconcat` layout
    joint_plot = alt.hconcat(
        y_density,
        alt.vconcat(x_density, scatter)
    ).resolve_scale(
        x='independent',
        y='independent'
    )

    # Display the joint plot in Streamlit
    st.altair_chart(joint_plot, use_container_width=True)

    st.write(f"Most frequent word in the descriptions for {selected_variety}:")

    # Download stopwords if you haven't already
    nltk.download('stopwords')

    # Load English stopwords
    stop_words = set(stopwords.words('english'))

    # Sample DataFrame (replace with your actual DataFrame)
    # df = pd.read_csv('your_data.csv')  # Uncomment this line to load your actual data

    # Combine all text into one large string
    all_text = ' '.join(df['description'])

    # Clean the text (remove punctuation, convert to lowercase)
    all_text_cleaned = re.sub(r'[^\w\s]', '', all_text.lower())  # Removing punctuation, converting to lowercase

    # Tokenize the text (split into words)
    words = all_text_cleaned.split()

    # Remove stopwords
    filtered_words = [word for word in words if word not in stop_words]

    # Generate the word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color= None, 
        colormap='autumn',  # You can change the color scheme if you like
        max_words=100  # Limit the number of words displayed
    ).generate(' '.join(filtered_words))

    # Plot the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Hide axes
    plt.title('Most Frequent Words in Text')

    # Show the plot in Streamlit
    st.pyplot(plt)
elif options == "Comparison":
    st.markdown(
        "<h1 style='color: red;'>Comparison</h1>", 
        unsafe_allow_html=True
    )
    st.markdown(
        """Lets compare different wines"""
    )

    # Get unique varieties for dropdowns
    varieties = df['variety'].unique()

    # Create dropdowns for selecting two varieties
    variety1 = st.selectbox("Select the first variety:", varieties)
    variety2 = st.selectbox("Select the second variety:", varieties)

    # Function to get details of selected variety
    def get_variety_details(variety):
        filtered_data = df[df['variety'] == variety]
        avg_price = filtered_data['price'].mean()
        avg_points = filtered_data['points'].mean()
        available_countries = filtered_data['country'].unique().tolist()
        available_province = filtered_data['province'].unique().tolist()
        vintage_years = filtered_data['vintage_year'].unique().tolist()
        
        return avg_price, avg_points, available_countries, available_province, vintage_years

    # Get details for both selected varieties
    price1, points1, countries1, province1, vintages1 = get_variety_details(variety1)
    price2, points2, countries2, province2, vintages2 = get_variety_details(variety2)
    
    # Add a title for the overall comparison
    st.write(f"### Comparison of {variety1} and {variety2}")

    # Create two columns for side-by-side display
    col1, col2, col3 = st.columns(3)

    # Display details for Variety 1 in the first column
    with col1:
        st.write("##### Features")
        st.write("**Average Price:**")
        st.write("**Average Points**")
        st.write("**Available Countries:**")
        st.write("**Available Province:**")
        st.write("**Vintage Years:**")
    
    # Display details for Variety 1 in the first column
    with col2:
        st.write(f"##### {variety1}")
        st.write(f"${price1:.2f}")
        st.write(f"{points1}")
        st.write(f"{', '.join(countries1)}")
        st.write(f"{', '.join(province1)}")
        st.write(f"{', '.join(map(str, vintages1))}")

    # Display details for Variety 2 in the second column
    with col3: 
        st.write(f"##### {variety2}")
        st.write(f"${price2:.2f}")
        st.write(f"{points2}")
        st.write(f"{', '.join(countries2)}")
        st.write(f"{', '.join(province2)}")
        st.write(f"{', '.join(map(str, vintages2))}")



# Footer
st.sidebar.markdown("---")
st.sidebar.write("Created by Roshni Bhowmik")
