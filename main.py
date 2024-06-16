## Installing Data and Importing Libraries

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import scipy.stats
from PIL import Image
import requests
from io import BytesIO
import base64
import os

# Set up the page
st.set_page_config(
    page_title="International Happiness Accross Nations",
    page_icon="😊",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# Function to load an image from file path
def load_image_from_path(image_path):
    try:
        if os.path.exists(image_path):
            img = Image.open(image_path)
            return img
        else:
            st.error(f"Image file not found at: {image_path}")
            return None
    except (IOError, OSError) as e:
        st.error(f"Error opening image: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None


# File path of the local image
fallback_image_path = "fallback_image.jpg"

# Load the image from file path
image = load_image_from_path(fallback_image_path)

# Display the image in Streamlit if successfully loaded
if image:
    st.image(image, caption="Fallback Image", use_column_width=True)
else:
    st.write("Failed to load the image from file path.")

# Sidebar Configuration and other pages omitted for brevity


# Sidebar Configuration
st.sidebar.title("Summary")
pages = ["Introduction", "Data Exploration", "Data Visualization", "Modelling", "Conclusion and Insights"]
page = st.sidebar.radio("Go to", pages)
st.sidebar.markdown(
    """
    - **Course**: Data Analyst
    - **Type**: Bootcamp
    - **Month**: April 2024
    - **Group**:
        - Amira Ben Salem
        - Julian Buß
        - Subash Chandra Pal
    """
)

# Introduction Page
if page == pages[0]:
    st.caption("""**Course**: Data Analyst
    | **Type**: Bootcamp
    | **Month**: April 2024
    | **Group**: Amira Ben Salem, Julian Buß, Subash Chandra Pal
    """)

    # Introduction page content
    st.markdown("""
    <div class="intro-container">
        <h1 class="intro-header">👋 Welcome to the International Happiness Report Analysis </h1>
        <h2 class="intro-subheader">Discover What Makes People Happy accross nations </h2>
    </div>
    """, unsafe_allow_html=True)

# Load data
data = pd.read_csv("merged_df_happy (1).csv")

# Define the list of variables to drop
variables_to_drop = [
    'Standard error of ladder score',
    'upperwhisker',
    'lowerwhisker',
    'Ladder score in Dystopia',
    'Explained by: Log GDP per capita',
    'Explained by: Social support',
    'Explained by: Healthy life expectancy',
    'Explained by: Freedom to make life choices',
    'Explained by: Generosity',
    'Explained by: Perceptions of corruption',
    'Dystopia + residual'
]

# Drop the specified variables from the DataFrame
data.drop(columns=variables_to_drop, inplace=True)

from sklearn.impute import SimpleImputer

# Initialize SimpleImputer with mean strategy
imputer = SimpleImputer(strategy='mean')

# Define columns with missing values
columns_with_missing = ['Log GDP per capita', 'Social support',
                        'Healthy life expectancy at birth',
                        'Freedom to make life choices',
                        'Generosity', 'Perceptions of corruption',
                        'Positive affect', 'Negative affect']

# Impute missing values
data[columns_with_missing] = imputer.fit_transform(data[columns_with_missing])

# List of countries without regional indicators
countries_in_question = ['Angola', 'Belize', 'Bhutan', 'Central African Republic',
                         'Congo (Kinshasa)', 'Cuba', 'Djibouti', 'Guyana', 'Oman', 'Qatar',
                         'Somalia', 'Somaliland region', 'South Sudan', 'Sudan', 'Suriname',
                         'Syria', 'Trinidad and Tobago']

# Check if any of the countries in question have missing regional indicators
missing_regions = data[data['Country name'].isin(countries_in_question) & data['Regional indicator'].isnull()]

# Data Audit, object types
data_types = data.dtypes

# Data Exploration Page
if page == pages[1]:
    st.header("Data Exploration")
    st.subheader("Dataset on happiness around the world between 2005 et 2021")

    st.write("Here are the first few rows of the dataset, providing an initial glimpse of the data:")
    st.write(data.head())

    st.write("The shape of the dataset indicates the number of rows and columns present:")
    st.write("- Number of rowns :", data.shape[0])
    st.write("- Number of columns :", data.shape[1])

    st.write("These are the columns present in the dataset, each representing different attributes:")
    st.write(data.columns)

    st.write("The data types of each column describe the format and nature of the data stored:")
    st.write(data.dtypes)

    logged_gdp_per_capita = data['Log GDP per capita']

    life_ladder = data['Life Ladder']

    plt.figure(figsize=(8, 6))
    plt.hist(life_ladder, bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Life Ladder')
    plt.xlabel('Life Ladder')
    plt.ylabel('Frequency')
    st.pyplot(plt)
    st.write(
        "The histogram above illustrates the distribution of the 'Life Ladder' variable, which represents happiness scores.")
    st.write(
        "The distribution appears to be approximately normal, suggesting a central tendency around certain happiness levels.")
    # Shapiro-Wilk Test
    # # Extract the "Life Ladder" data from the dataset
    life_ladder_data = data['Life Ladder']
    # Perform Shapiro-Wilk test for Life Ladder
    statistic, p_value = stats.shapiro(life_ladder_data)
    # Explanation for the Shapiro-Wilk Test
    st.write("""We will use the Shapiro-Wilk test to confirm the distribution of the life ladder variable. This test checks if a sample comes from a normal distribution, providing a test statistic (0.9877) and a p-value (1.8879e-12).
             - The test statistic close to 1 suggests normality.
             - The very small p-value (< 0.05) strongly rejects the null hypothesis of normality.
             - Despite the high test statistic, the tiny p-value indicates that the "life_ladder" data does not follow a normal distribution.
             """)
    st.write(f"**Shapiro-Wilk Test Statistic for Life Ladder:** {statistic}")
    st.write(f"**P-value for Life Ladder:** {p_value}")

# Data Visualisation Page
if page == pages[2]:
    st.header("Data Visualisation")
    # Filter data for the year 2021
    df_2021 = data[data['year'] == 2021]
    # Filter data for the year 2021
    # # Fill missing values using the regional average
    df_2021['Life Ladder'] = df_2021.groupby('Regional indicator')['Life Ladder'].transform(
        lambda x: x.fillna(x.mean()))
    # Identify countries still missing Life Ladder values
    missing_countries = df_2021[df_2021['Life Ladder'].isnull()]['Country name'].tolist()
    # Ensure the correct column names and no remaining NaN values
    df_2021 = df_2021[['Country name', 'Life Ladder']]
    # Create a copy of the dataframe to work with
    df_filled = df_2021.copy()
    # Set a placeholder value for missing data (choose a value outside the expected range)
    placeholder_value = -999
    # Fill NaN values with the placeholder value
    df_filled['Life Ladder'].fillna(placeholder_value, inplace=True)
    # Create the map with conditional coloring
    fig = px.choropleth(
        df_filled,
        locations="Country name",
        locationmode='country names',
        color="Life Ladder",
        hover_name="Country name",
        color_continuous_scale=px.colors.sequential.Plasma,  # Use a predefined colorscale
        range_color=(df_filled['Life Ladder'].min(), df_filled['Life Ladder'].max()),  # Specify actual data range
        labels={'Life Ladder': 'Happiness Index'},
        color_continuous_midpoint=placeholder_value, )  # Ensure the midpoint aligns with the placeholder value)
    # Update colorscale to map the placeholder_value to gray
    fig.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=1)
    fig.update_layout(
        title_text='Happiness around the world in 2021',  # Title above the graph
        geo=dict(showframe=False, showcoastlines=False, projection_type='equirectangular')
    )
    # Streamlit app
    st.write("This map shows the Happiness Index (Life Ladder) for different countries in the year 2021. \
             Countries with no data are shown in white.")
    st.plotly_chart(fig)
    st.markdown("###### Healthy Life Expectancy at Birth for the period 2005 to 2021")
    data = data.dropna(subset=['Life Ladder', 'Healthy life expectancy at birth'])
    df_grouped = data.groupby(['Country name', 'Regional indicator']).agg({
        'Life Ladder': 'mean',
        'Healthy life expectancy at birth': 'mean'
    }).reset_index()

    fig = px.sunburst(df_grouped,
                      path=['Regional indicator', 'Country name'],
                      values='Life Ladder',
                      color='Healthy life expectancy at birth',
                      color_continuous_scale='RdBu',
                      color_continuous_midpoint=np.average(df_grouped['Healthy life expectancy at birth'],
                                                           weights=df_grouped['Life Ladder']))
    st.plotly_chart(fig)
    fig.update_layout(title_text='Happiness around the world in 2021')
    st.write("""
    The graph below shows the average life ladder score for several countries including the United States, Denmark, Netherlands, Canada, Sweden, Australia, Finland, Switzerland, Norway, New Zealand, Costa Rica, Israel, Venezuela, Austria, and Iceland. The life ladder score is a measure of subjective well-being that ranges from 0 to 10, with higher scores indicating greater happiness. The graph shows that Denmark has the highest average life ladder score over the time period covered by the graph, followed by Finland and Switzerland. The United States ranks tenth.
    """)

    df = data.dropna(subset=['Life Ladder'])
    df_top5 = df.groupby('year').apply(lambda x: x.nlargest(5, 'Life Ladder')).reset_index(drop=True)
    fig = px.line(df_top5,
                  x='year',
                  y='Life Ladder',
                  color='Country name',
                  line_group='Country name',
                  hover_name='Country name',
                  labels={'Life Ladder': 'Life Ladder Score', 'year': 'Year'},
                  title='Top 5 Happiest Countries (2005-2021)')
    st.plotly_chart(fig)

    st.write("""
    Given their consistently low scores, Burundi, Central African Republic, South Sudan, and Afghanistan likely remained among the unhappiest throughout the period.
    """)

    df_bottom5 = df.groupby('year').apply(lambda x: x.nsmallest(5, 'Life Ladder')).reset_index(drop=True)
    fig = px.line(df_bottom5,
                  x='year',
                  y='Life Ladder',
                  color='Country name',
                  line_group='Country name',
                  hover_name='Country name',
                  labels={'Life Ladder': 'Life Ladder Score', 'year': 'Year'},
                  title='Top 5 Unhappiest Countries (2005-2021)')
    st.plotly_chart(fig)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x='Regional indicator', y='Life Ladder')
    plt.title('Box Plot of Life Ladder by Regional Indicator')
    plt.xticks(rotation=45)
    st.pyplot(plt)

    plt.figure(figsize=(8, 6), dpi=100)
    average_life_ladder_by_region = data.groupby('Regional indicator')['Life Ladder'].mean().sort_values(
        ascending=False).reset_index()
    sns.barplot(data=average_life_ladder_by_region, x='Life Ladder', y='Regional indicator')
    plt.title('Average Life Ladder by Regional Indicator')
    st.pyplot(plt)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x='Log GDP per capita', y='Life Ladder')
    plt.title('Scatter Plot of Life Ladder vs Log GDP per capita')
    st.pyplot(plt)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x='Generosity', y='Life Ladder', alpha=0.7)
    plt.title('Scatter Plot: Life Ladder vs Generosity')
    st.pyplot(plt)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=data, x='Freedom to make life choices', y='Life Ladder', alpha=0.7)
    plt.title('Scatter Plot: Life Ladder vs Freedom to make life choices')
    st.pyplot(plt)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=data, x='Perceptions of corruption', y='Life Ladder', alpha=0.7)
    plt.title('Scatter Plot: Life Ladder vs Perceptions of corruption')
    st.pyplot(plt)

    plt.figure(figsize=(10, 8))
    correlation_matrix = data[
        ['Life Ladder', 'Log GDP per capita', 'Social support', 'Healthy life expectancy at birth',
         'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    st.pyplot(plt)

# Modelling Page
if page == pages[3]:
    st.header(" Modelling")

    st.subheader("Objectif")
    st.write(
        "To determine the best machine learning model for predicting the happiness score, or life ladder, using the available data from the World Happiness Report ")

    st.write("")
    st.write("")

    if st.button("Models developped "):
        st.subheader("Models ")
        st.markdown("""

                1- Linear Regression Model

                2- Decision Tree Model

                3- Gradiant Boosting Model

                4- LassoCV Model

                5- Random Forest 

                6- Ridge 


        """)

        st.write("")
        st.write("")

        st.subheader("Models creation and development")
        st.markdown("""
                    For each developed model, we followed these steps: :
                    1. Model instantiation
                    2. Training each model on training set  X_train and  y_train (80%, 20%)
                    3. Making predictions on test set  X_test and  y_test
                    4. Eavaluating model performance using specific metrics 
                    5. Interpreting features importance for each model
                    6. Visualizing and analyzing the results  
                """)

    st.write("")
    st.write("")
    if st.button("Decision Tree Model "):
        if 'dt_button' not in st.session_state:
            st.session_state.dt_button = False

        label_encoder = LabelEncoder()
        data['Regional indicator'] = label_encoder.fit_transform(data['Regional indicator'])
        data['Country name'] = label_encoder.fit_transform(data['Country name'])
        X = data.drop(['Life Ladder'], axis=1)
        y = data['Life Ladder']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        decision_tree_model = DecisionTreeRegressor(random_state=42)
        decision_tree_model.fit(X_train, y_train)
        train_predictions = decision_tree_model.predict(X_train)
        test_predictions = decision_tree_model.predict(X_test)
        r2_train = r2_score(y_train, train_predictions)
        mae_train = mean_absolute_error(y_train, train_predictions)
        mse_train = mean_squared_error(y_train, train_predictions)
        rmse_train = np.sqrt(mse_train)
        r2_test = r2_score(y_test, test_predictions)
        mae_test = mean_absolute_error(y_test, test_predictions)
        mse_test = mean_squared_error(y_test, test_predictions)
        rmse_test = np.sqrt(mse_test)
        st.subheader("Metrics Results ")
        st.write("Decision Tree Model training Results:")
        st.write("R²:", r2_train)
        st.write("MAE:", mae_train)
        st.write("MSE:", mse_train)
        st.write("RMSE:", rmse_train)
        st.write("Decision Tree Model testing Results::")
        st.write("R²:", r2_test)
        st.write("MAE:", mae_test)
        st.write("MSE:", mse_test)
        st.write("RMSE:", rmse_test)
        st.subheader("Features Importance ")
        feature_importances = decision_tree_model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
        plt.title('Feature Importance')
        st.pyplot(plt)
        plt.figure(figsize=(8, 6))
        plt.scatter(y_train, train_predictions, color='blue', alpha=0.5)
        plt.title('Training Data: Target vs Prediction')
        plt.xlabel('Actual Life Ladder')
        plt.ylabel('Predicted Life Ladder')
        st.subheader("Training and Testing : Target vs Prediction ")
        # 6 - Plot scatter plot of target vs prediction for training data
        plt.figure(figsize=(8, 6))
        plt.scatter(y_train, train_predictions, color='blue', alpha=0.5)
        plt.title('Training Data: Target vs Prediction')
        plt.xlabel('Actual Life Ladder')
        plt.ylabel('Predicted Life Ladder')
        plt.grid(True)
        st.pyplot(plt)  # Use st.pyplot() instead of plt.show()
        # Plot scatter plot of target vs prediction for testing data
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, test_predictions, color='green', alpha=0.5)
        plt.title('Testing Data: Target vs Prediction')
        plt.xlabel('Actual Life Ladder')
        plt.ylabel('Predicted Life Ladder')
        plt.grid(True)
        st.pyplot(plt)  # Use st.pyplot() instead of plt.show()
        st.write("""
     The initial creation of the decision tree model resulted in an R² score of 1, indicating a perfect fit. The three most important features within the model were GDP log per capita, healthy life expectancy at birth, and social support.
                 To address overfitting in the Decision Tree Model, several steps were taken:
                 **Limiting Maximum Depth**: 
                 The maximum depth of the decision tree was set to 5. This restricts the number of levels in the tree, simplifying the model and reducing overfitting.
                 - **Training and Evaluation**: The Decision Tree Model was trained on the imputed training data and evaluated on both the training and testing sets. R² scores were calculated to assess the model's performance in capturing the variance in the target variable.
                 - **Visualiztion of the Decision Tree**: The decision tree was visualized using the `plot_tree` function from the `sklearn.tree` module. This visualization provides insights into the structure of the decision tree and how it makes predictions, aiding in understanding its behavior and potential areas of improvement.
                 Overall, these adjustments help in mitigating overfitting and improving the generalization performance of the Decision Tree Model.
""")
        st.subheader("Adjusting Overfitting ")
        from sklearn.impute import SimpleImputer

        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        # Initialize a Decision Tree Regressor with a maximum depth
        max_depth = 5  # Set the maximum depth of the decision tree
        decision_tree_model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        # Train the Decision Tree Model
        decision_tree_model.fit(X_train_imputed, y_train)
        # Make predictions on the training and testing sets
        train_predictions = decision_tree_model.predict(X_train_imputed)
        test_predictions = decision_tree_model.predict(X_test_imputed)
        # Calculate R² for training and testing data
        train_r2 = r2_score(y_train, train_predictions)
        test_r2 = r2_score(y_test, test_predictions)
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        # Calculate MAE, MSE, and RMSE for training and testing data
        train_mae = mean_absolute_error(y_train, train_predictions)
        test_mae = mean_absolute_error(y_test, test_predictions)
        train_mse = mean_squared_error(y_train, train_predictions)
        test_mse = mean_squared_error(y_test, test_predictions)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        st.write("Training R²:", train_r2)
        st.write("Testing R²:", test_r2)
        st.write("Training MAE:", train_mae)
        st.write("Testing MAE:", test_mae)
        st.write("Training MSE:", train_mse)
        st.write("Testing MSE:", test_mse)
        st.write("Training RMSE:", train_rmse)
        st.write("Testing RMSE:", test_rmse)
        import streamlit as st
        import matplotlib.pyplot as plt
        from sklearn.tree import DecisionTreeRegressor, plot_tree

        # Plot the decision tree
        plt.figure(figsize=(12, 6))
        plot_tree(decision_tree_model, filled=True, feature_names=X.columns)
        st.pyplot(plt)
        if st.button("Toggle Decision Tree Model"):
            st.session_state.dt_button = not st.session_state.dt_button
        if st.session_state.dt_button:
            decision_tree_model()

    if st.button("Linear Regression Model "):
        label_encoder = LabelEncoder()
        data['Regional indicator'] = label_encoder.fit_transform(data['Regional indicator'])
        data['Country name'] = label_encoder.fit_transform(data['Country name'])
        X = data.drop(['Life Ladder'], axis=1)
        y = data['Life Ladder']
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Fit the linear regression model
        linear_reg_model = LinearRegression()
        linear_reg_model.fit(X_train, y_train)
        # Predict on training and testing sets
        y_train_pred = linear_reg_model.predict(X_train)
        y_test_pred = linear_reg_model.predict(X_test)
        # Calculate metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        mse = mean_squared_error(y_test, y_test_pred)
        rmse = mean_squared_error(y_test, y_test_pred, squared=False)
        st.subheader("Metrics Results ")
        st.write("Linear Regression Model Training Metrics:")
        st.write("R²:", train_r2)
        st.write("MAE:", mae)
        st.write("MSE:", mse)
        st.write("RMSE:", rmse)
        st.write("Linear Regression Model Testing Metrics:")
        st.write("R²:", test_r2)
        st.write("MAE:", mae)
        st.write("MSE:", mse)
        st.write("RMSE:", rmse)
        # Plot feature importance (coefficients)
        st.subheader("Features Importance ")
        feature_importance = pd.Series(linear_reg_model.coef_, index=X.columns)
        feature_importance_sorted = feature_importance.sort_values(ascending=False)
        plt.figure(figsize=(10, 8))  # Increase the figure size for better visibility
        fig, ax = plt.subplots(figsize=(10, 8))  # Create a matplotlib figure and axis
        sns.barplot(x=feature_importance_sorted.values, y=feature_importance_sorted.index, palette='viridis', ax=ax)
        ax.set_title('Feature Importance (Absolute Coefficients) for Linear Regression')
        ax.set_xlabel('Absolute Coefficient Value')
        ax.set_ylabel('Feature')
        ax.tick_params(axis='x', rotation=45)  # Rotate the x-axis labels for better readability
        st.pyplot(fig)  # Display the plot in Streamlit using st.pyplot()
        # Create scatter plot for training data
        st.subheader("Training and Testing : Target vs Prediction ")
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.scatter(y_train, y_train_pred, color='blue', label='Actual vs Predicted')
        plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', linestyle='--',
                 label='Ideal Line')
        plt.title('Linear Regression: Training Data - Actual vs Predicted')
        plt.xlabel('Actual Life Ladder')
        plt.ylabel('Predicted Life Ladder')
        plt.legend()
        st.pyplot(fig)  # Display the plot in Streamlit using st.pyplot()
        # Create scatter plot for testing data
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.scatter(y_test, y_test_pred, color='green', label='Actual vs Predicted')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--',
                 label='Ideal Line')
        plt.title('Linear Regression: Testing Data - Actual vs Predicted')
        plt.xlabel('Actual Life Ladder')
        plt.ylabel('Predicted Life Ladder')
        plt.legend()
        st.pyplot(fig)  # Display the plot in Streamlit using st.pyplot()
        st.write("Conclusion on Linear Regression Model:")
        st.write(
            "Training R² of 0.769 and Testing R² of 0.733 suggest that the model explains approximately 76.9% and 73.3% of the variance in the target variable, respectively. These values are relatively high, indicating a good fit of the model to the data.")
        st.write(
            "Overall, these metrics indicate that the linear regression model performs reasonably well in predicting the Life Ladder score based on the provided features. However, there is still room for improvement, especially considering potential complexities and nuances in the data that may not be captured by a linear model.")

    # Buttons for individual models
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    # Ensure Streamlit session state
    if st.button("Gradiant Boosting Model "):
        # Define target variable (y) and features (X)
        y = data['Life Ladder']
        X = data[['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth',
                  'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Define the parameter grid
        param_grid = {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
        # Initialize the Gradient Boosting Regressor
        gbr = GradientBoostingRegressor(random_state=42)
        # Initialize Grid Search with cross-validation
        grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error',
                                   n_jobs=-1)
        # Fit Grid Search to the data
        grid_search.fit(X_train, y_train)
        # Best parameter found by Grid Search
        best_max_depth = grid_search.best_params_['max_depth']
        st.write(f"Best max_depth: {best_max_depth}")
        # Train the Gradient Boosting Regressor with the best max_depth
        gbr = GradientBoostingRegressor(max_depth=best_max_depth, random_state=42)
        gbr.fit(X_train, y_train)
        # Predictions
        y_train_pred_gbr = gbr.predict(X_train)
        y_test_pred_gbr = gbr.predict(X_test)
        # Metrics for Gradient Boosting Regressor

        st.markdown("### Metrics Results")
        r2_train_gbr = r2_score(y_train, y_train_pred_gbr)
        st.write("R² Train = 0.987827")
        r2_test_gbr = r2_score(y_test, y_test_pred_gbr)
        st.write("R² Test = 0.850765")
        mae_test_gbr = mean_absolute_error(y_test, y_test_pred_gbr)
        st.write("MAE Test = 0.326058")
        mae_train_gbr = mean_absolute_error(y_train, y_train_pred_gbr)
        st.write("MAE Train = 0.091736")
        mse_test_gbr = mean_squared_error(y_test, y_test_pred_gbr)
        st.write("MSE Test = 0.186756")
        mse_train_gbr = mean_squared_error(y_train, y_train_pred_gbr)
        st.write("MSE Train = 0.015003")
        rmse_test_gbr = np.sqrt(mse_test_gbr)
        st.write("RMSE Test = 0.432153")
        rmse_train_gbr = np.sqrt(mse_train_gbr)
        st.write("RMSE Train = 0.122488")
        # Feature importances from Gradient Boosting Regressor
        st.markdown("### Feature Importance")
        feature_importances = gbr.feature_importances_
        # Plotting feature importances
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        sns.barplot(x=feature_importances, y=X.columns, ax=ax1)
        ax1.set_title('Feature Importances from Gradient Boosting Regressor')
        ax1.set_xlabel('Importance')
        ax1.set_ylabel('Feature')
        st.pyplot(fig1)

        # Scatter plot of target vs predictions for Gradient Boosting Regressor
        st.markdown("### Training and Testing")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.scatter(y_test, y_test_pred_gbr, alpha=0.5)
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
        ax2.set_title('Gradient Boosting Regressor: Target vs Predictions')
        ax2.set_xlabel('Actual')
        ax2.set_ylabel('Predicted')
        st.pyplot(fig2)

        # Display model metrics
        st.markdown("""
         <div style="font-size: 14px;">
         <b>Gradient Boosting:</b><br>
         - <b>R² Train and Test</b> The Gradient Boosting model has a higher R² value for both training (0.9878) and testing (0.8508) compared to the LassoCV model (training: 0.7423, testing: 0.7055). This indicates that the Gradient Boosting model explains a higher proportion of the variance in the target variable both in training and testing sets. The Gradient Boosting model's R² values show a small decrease from training to testing, suggesting it generalizes well to unseen data.<br>
         - <b>MAE Train and Test:</b>  The Gradient Boosting model has a significantly lower MAE for both training (0.0917) and testing (0.3261) compared to the LassoCV model (training: 0.4446, testing: 0.4728). This indicates that the Gradient Boosting model's predictions are, on average, closer to the actual values.<br>
         - <b>MSE and RMSE :</b> The Gradient Boosting model also has a much lower MSE for both training (0.0150) and testing (0.1868) compared to the LassoCV model (training: 0.3176, testing: 0.3685). Lower MSE values indicate better model performance in terms of the average squared difference between the predicted and actual values.Similarly, the Gradient Boosting model has a lower RMSE for both training (0.1225) and testing (0.4322) compared to the LassoCV model (training: 0.5636, testing: 0.6070). Lower RMSE values indicate that the Gradient Boosting model has better overall predictive accuracy.<br>
         - <b>Conclusion :</b>Overall Performance: The Gradient Boosting model outperforms the LassoCV model across all metrics (R², MAE, MSE, RMSE) for both training and testing datasets. It has higher R² values and lower error metrics, indicating more accurate and reliable predictions. Generalization: The Gradient Boosting model shows good generalization from training to testing data, as indicated by the relatively small decrease in R² and moderate increase in error metrics. Prediction Accuracy: The significantly lower MAE, MSE, and RMSE values for the Gradient Boosting model suggest it is more precise in predicting the target variable compared to the LassoCV model.
         </div>
         """, unsafe_allow_html=True)

    if st.button("LassoCV Model"):
        from sklearn.linear_model import LassoCV
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.datasets import make_regression

        # Define target variable (y) and features (X)
        y = data['Life Ladder']
        X = data[['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth',
                  'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']]
        from sklearn.model_selection import train_test_split

        # Split data into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        from sklearn.linear_model import LassoCV

        # Training the LassoCV model
        lasso = LassoCV(cv=5, random_state=42)
        lasso.fit(X_train, y_train)
        # Predictions
        y_train_pred_lasso = lasso.predict(X_train)
        y_test_pred_lasso = lasso.predict(X_test)
        from sklearn.metrics import r2_score
        from sklearn.metrics import mean_absolute_error
        from sklearn.metrics import mean_squared_error

        # Metrics for LassoCV
        r2_train_lasso = r2_score(y_train, y_train_pred_lasso)
        r2_test_lasso = r2_score(y_test, y_test_pred_lasso)
        mae_test_lasso = mean_absolute_error(y_test, y_test_pred_lasso)
        mae_train_lasso = mean_absolute_error(y_train, y_train_pred_lasso)
        mse_test_lasso = mean_squared_error(y_test, y_test_pred_lasso)
        mse_train_lasso = mean_squared_error(y_train, y_train_pred_lasso)
        rmse_train_lasso = np.sqrt(mse_train_lasso)
        rmse_test_lasso = np.sqrt(mse_test_lasso)
        # Feature Importance
        st.subheader('Feature Importance from LassoCV Model')
        feature_importances = np.abs(lasso.coef_)
        fig_feat, ax_feat = plt.subplots(figsize=(10, 6))
        sns.barplot(x=feature_importances, y=X.columns, ax=ax_feat)
        ax_feat.set_title('Feature Importances from LassoCV Model')
        ax_feat.set_xlabel('Importance')
        ax_feat.set_ylabel('Feature')
        plt.tight_layout()
        st.pyplot(fig_feat)
        # Plotting the results
        st.subheader('LassoCV: Target vs Predictions')
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_test_pred_lasso, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
        ax.set_title('LassoCV: Target vs Predictions')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        plt.tight_layout()
        st.pyplot(fig)
        st.subheader('Metrics Results ')
        st.write('R² Train:', r2_train_lasso)
        st.write('R² Test:', r2_test_lasso)
        st.write('MAE Train:', mae_train_lasso)
        st.write('MAE Test:', mae_test_lasso)
        st.write('MSE Train:', mse_train_lasso)
        st.write('MSE Test:', mse_test_lasso)
        st.write('RMSE Train:', rmse_train_lasso)
        st.write('RMSE Test:', rmse_test_lasso)
        st.subheader('Conclusion ')
        st.write(
            "R² Train (0.742295): This indicates that the LassoCV model explains approximately 74.23% of the variance in the training data. This is significantly lower than Gradient Boosting, suggesting that LassoCV is not capturing the relationship as well.")
        st.write(
            "R² Test (0.705542): This indicates that the model explains approximately 70.55% of the variance in the test data. Although this is slightly lower than the training R², it is close, indicating reasonable generalization.")
        st.write(
            "MAE Test(0.472798): The mean absolute error is higher compared to Gradient Boosting, suggesting that on average, the model's predictions are off by about 0.473 units.")
        st.write(
            "MSE Test(0.368493) and RMSE Test(0.432153): These metrics are higher than those for Gradient Boosting, indicating that LassoCV has worse predictive performance.")
        st.write(
            "MSE Train(0.317628) and RMSE Train(0.607036): These metrics are higher than those for Gradient Boosting, indicating that LassoCV has worse predictive performance.")

        st.write(
            "Overall, these metrics indicate that the LassoCV model performs reasonably well in predicting the Life Ladder score based on the provided features. However, there is still room for improvement, especially considering potential complexities and nuances in the data that may not be captured by a linear model.")

    if st.button("Random Forest"):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

        y = data['Life Ladder']
        x = data[['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth',
                  'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        # Train the Random Forest model
        rf_model = RandomForestRegressor(random_state=42)
        rf_model.fit(x_train, y_train)
        # Predictions
        y_train_pred_rf = rf_model.predict(x_train)
        y_test_pred_rf = rf_model.predict(x_test)
        # Metrics for Random Forest
        r2_train_rf = r2_score(y_train, y_train_pred_rf)
        r2_test_rf = r2_score(y_test, y_test_pred_rf)
        mae_rf = mean_absolute_error(y_test, y_test_pred_rf)
        mse_rf = mean_squared_error(y_test, y_test_pred_rf)
        rmse_rf = np.sqrt(mse_rf)
        # Plotting the results - Target vs Predictions
        st.subheader('Random Forest: Target vs Predictions')
        fig_rf, ax_rf = plt.subplots()
        ax_rf.scatter(y_test, y_test_pred_rf, alpha=0.5)
        ax_rf.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
        ax_rf.set_title('Random Forest: Target vs Predictions')
        ax_rf.set_xlabel('Actual')
        ax_rf.set_ylabel('Predicted')
        plt.tight_layout()
        st.pyplot(fig_rf)
        # Feature Importance
        st.subheader('Feature Importance from Random Forest Model')
        feature_importances_rf = rf_model.feature_importances_
        fig_feat_rf, ax_feat_rf = plt.subplots(figsize=(10, 6))
        sns.barplot(x=feature_importances_rf, y=x.columns, ax=ax_feat_rf)
        ax_feat_rf.set_title('Feature Importances from Random Forest Model')
        ax_feat_rf.set_xlabel('Importance')
        ax_feat_rf.set_ylabel('Feature')
        plt.tight_layout()
        st.pyplot(fig_feat_rf)
        # Conclusion
        st.subheader('Metrics')
        st.write(f"R² Train: {r2_train_rf:.4f}")
        st.write(f"R² Test: {r2_test_rf:.4f}")
        st.write(f"MAE: {mae_rf:.4f}")
        st.write(f"MSE: {mse_rf:.4f}")
        st.write(f"RMSE: {rmse_rf:.4f}")
        st.subheader('Conclusion')
        st.write(
            "Overall, these metrics suggest that the Random Forest model performs well in predicting the Life Ladder score based on the provided features. The R² values indicate that the model explains a significant portion of the variance in both training and test datasets. Additionally, the feature importance plot highlights which features have the most impact on the predictions.")

    if st.button("Ridge"):
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

        # Assuming 'data' is your dataset
        y = data['Life Ladder']
        x = data[['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth',
                  'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']]
        # Split data into train/test sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        # Train the Ridge model
        ridge_model = Ridge(alpha=1.0)  # You can adjust the alpha (regularization strength) as needed
        ridge_model.fit(x_train, y_train)
        # Predictions
        y_train_pred_ridge = ridge_model.predict(x_train)
        y_test_pred_ridge = ridge_model.predict(x_test)
        # Metrics for Ridge
        r2_train_ridge = r2_score(y_train, y_train_pred_ridge)
        r2_test_ridge = r2_score(y_test, y_test_pred_ridge)
        mae_ridge = mean_absolute_error(y_test, y_test_pred_ridge)
        mse_ridge = mean_squared_error(y_test, y_test_pred_ridge)
        rmse_ridge = np.sqrt(mse_ridge)
        # Plotting the results - Target vs Predictions
        st.subheader('Ridge Regression: Target vs Predictions')
        fig_ridge, ax_ridge = plt.subplots()
        ax_ridge.scatter(y_test, y_test_pred_ridge, alpha=0.5)
        ax_ridge.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
        ax_ridge.set_title('Ridge Regression: Target vs Predictions')
        ax_ridge.set_xlabel('Actual')
        ax_ridge.set_ylabel('Predicted')
        plt.tight_layout()
        st.pyplot(fig_ridge)
        # Feature Importance (Ridge doesn't have feature_importances_, so we skip this section)
        # # Conclusion
        st.subheader('Metrics')
        st.write(f"R² Train: {r2_train_ridge:.4f}")
        st.write(f"R² Test: {r2_test_ridge:.4f}")
        st.write(f"MAE: {mae_ridge:.4f}")
        st.write(f"MSE: {mse_ridge:.4f}")
        st.write(f"RMSE: {rmse_ridge:.4f}")
        st.subheader('Conclusion')
        st.write(
            "Overall, these metrics suggest that the Ridge regression model performs reasonably well in predicting the Life Ladder score based on the provided features. The R² values indicate that the model explains a significant portion of the variance in both training and test datasets.")


    # Function to display the conclusion and table
    def display_conclusion_and_table():
        data = {
            "Model Name": ["Linear Regression", "Decision Tree", "Gradient Boosting", "LassoCV", "Random Forest",
                           "Ridge"],
            "Parameters": ["Default", "MaxDepth=7", "MaxDepth=7", "Alpha = 1", "MaxDepth=7", "Alpha=1"],
            "R2 Train": [0.7694, 0.8342, 0.9878, 0.7423, 0.9192, 0.9029],
            "R2 Test": [0.7328, 0.765, 0.850765, 0.7328, 0.8533, 0.8647],
            "MAE Train": [0.445, 0.3523, None, None, 0.2441, 0.2627],
            "MAE Test": [0.445, 0.4196, 0.3261, 0.4728, 0.327, 0.3048],
            "MSE Train": [0.3344, 0.2044, None, None, 0.0996, 0.1197],
            "MSE Test": [0.3344, 0.2935, 0.186756, 0.3685, 0.1836, 0.1693],
            "RMSE Train": [0.5782, 0.4521, None, None, 0.3155, 0.346],
            "RMSE Test": [0.5782, 0.5417, 0.432153, 0.607, 0.4285, 0.4115]
        }
        # Convert the data to DataFrame
        df = pd.DataFrame(data)
        # Adjust the index to start from 1
        df.index = pd.RangeIndex(start=1, stop=len(df) + 1, step=1)
        # Display the conclusion
        st.write("**Conclusion:**")
        st.write(
            "Considering the data in the comparative chart above, we can observe that for the test set the **Ridge Model** created the best result throughout all metrics of all models:")
        st.write(
            "It accurately explained 86.47% of the variance of the test set and also minimizes the error metrics to 0.4114 RMSE and 0.3048 MAE.")
        st.write("This indicates that the Ridge model performs the best among the considered models.")
        # Filter out None values which are causing empty rows
        df_filtered = df.dropna(how='all')
        # Manually format Ridge model metrics to be bold
        ridge_idx = df_filtered[df_filtered['Model Name'] == 'Ridge'].index
        styled_df = df_filtered.style.applymap(
            lambda x: 'font-weight: bold' if df_filtered.loc[ridge_idx, :].isin([x]).any().any() else '',
            subset=pd.IndexSlice[ridge_idx, :]
        )
        # Display the table
        st.write("**Comparative Table of Model Metrics:**")
        st.write(styled_df)


    if __name__ == "__main__":
        # Button to display conclusion and table
        if st.button("Conclusion"):
            display_conclusion_and_table()

if page == pages[4]:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    y = data['Life Ladder']
    X = data[['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth',
              'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']]
    # Split data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train the Ridge model
    ridge_model = Ridge(alpha=1.0)  # You can adjust the alpha (regularization strength) as needed
    ridge_model.fit(X_train, y_train)
    # Obtain feature coefficients
    feature_names = X.columns
    coefficients = ridge_model.coef_
    # Display feature coefficients
    st.subheader('Feature Coefficients from Ridge Regression Model')
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    st.write(coef_df)
    # Optionally, you can visualize the coefficients
    st.subheader('Visualization of Feature Coefficients')
    fig_coef, ax_coef = plt.subplots(figsize=(10, 6))
    sns.barplot(x=coefficients, y=feature_names, ax=ax_coef)
    ax_coef.set_title('Feature Coefficients from Ridge Regression Model')
    ax_coef.set_xlabel('Coefficient')
    ax_coef.set_ylabel('Feature')
    plt.tight_layout()
    st.pyplot(fig_coef)
    # Metrics for Ridge
    y_train_pred_ridge = ridge_model.predict(X_train)
    y_test_pred_ridge = ridge_model.predict(X_test)
    r2_train_ridge = r2_score(y_train, y_train_pred_ridge)
    r2_test_ridge = r2_score(y_test, y_test_pred_ridge)
    mae_ridge = mean_absolute_error(y_test, y_test_pred_ridge)
    mse_ridge = mean_squared_error(y_test, y_test_pred_ridge)
    rmse_ridge = np.sqrt(mse_ridge)
    # Conclusion and Policy Implications

    st.write(
        f"These coefficients suggest that higher values of 'Log GDP per capita', 'Social support', 'Freedom to make life choices', and 'Generosity', and a positive correlation with happiness.")
    st.write(
        f"Policy makers can use these insights to focus on improving economic conditions, social support systems, and freedom for individuals, which are crucial factors contributing to happiness in societies.")
    st.write(
        f"Further analysis and policy interventions can aim to enhance these factors to promote overall well-being and happiness across different countries.")
