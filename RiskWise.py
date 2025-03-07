import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import seaborn as sns

# ==========================
# User Controls
# ==========================
YEAR = 2024
GENERAL_GRAPHS = False

# ==========================
# Read N Data
# ==========================

# Get the directory of the current Python script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the relative path to the Excel file
relative_path = "DataMaster.xlsx"  # Adjust based on the file's location relative to the script

# Construct the full file path and read the Excel file
file_path = os.path.join(script_dir, relative_path)
df = pd.read_excel(file_path)

# Filter the data for the specified year
df_current_yr = df[df['Year'] == YEAR]

# ==========================
# Graphing Function
# ==========================
def f_plot_y_by_x_and_z(df, y="Yield (t/ha)", x="Previous Land Use", z=None):
    """
    Plots a bar chart of a specified dependent variable (y) against an independent variable (x), 
    optionally faceted by a third categorical variable (z).

    Parameters:
    ----------
    df : pandas.DataFrame
        The dataset containing the required columns.
    y : str, optional
        The dependent variable to be plotted on the y-axis (default is "Yield (t/ha)").
    x : str, optional
        The independent variable to be plotted on the x-axis (default is "Previous Land Use").
    z : str, optional
        An optional categorical variable to create multiple subplots (default is None).
        If None, a single plot is generated.

    Functionality:
    -------------
    - If `z` is provided and exists in the DataFrame, the function creates separate subplots for each unique value of `z`.
    - If `z` is None or not found in the DataFrame, a single plot is created using all data.
    - Bars are grouped by "Treatment" with a predefined color scheme.
    - Error bars (standard deviation) are included.
    - The function removes unnecessary plot borders for better visualization.
    - A shared y-axis is used for better comparison across subplots.
    - A legend is placed in the upper right corner.

    Returns:
    -------
    None
        Displays the generated plots.

    Example Usage:
    -------------
    f_plot_y_by_x_and_z(df_current_yr, y="Yield (t/ha)", x="Previous Land Use", z="Current Land Use")
    """
    # Get unique keys for subplots
    # Check if z exists in the DataFrame
    if z in df_current_yr.columns:
        subplot_categories = df[z].unique()
    else:
        # If "Current Land Use" doesn't exist, use a single subplot
        subplot_categories = [None]

    # Create subplots with independent x-axes
    fig, axes = plt.subplots(1, len(subplot_categories), figsize=(3 * len(subplot_categories), 3), sharey=True)  # Only share y-axis

    # Ensure axes is iterable even if there's only one subplot
    if len(subplot_categories) == 1:
        axes = [axes]

    # Define the colors and order for treatments
    colors = ['#d3d3d3', '#a9a9a9', '#696969']
    treatment_order = ['N', 'L', 'H']

    # Plot each land use separately
    for ax, subplot_category in zip(axes, subplot_categories):
        if subplot_category == None:
            subset = df  # No filtering, use the entire dataset
            # ax.set_title("All Land Uses")
        else:
            subset = df[df[z] == subplot_category]  # Filter data per subplot
            ax.set_title(subplot_category)  # Title per subplot
        sns.barplot(data=subset, x=x, y=y, hue="Treatment", hue_order=treatment_order, palette=colors, ax=ax, edgecolor='black', errorbar="sd", errwidth=1, capsize=0.1)

        ax.set_xlabel(x)  # X-axis label
        ax.set_ylabel(y)  # Y-axis label (shared)
        # ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for readability

        # Remove the box around the graph
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Remove legend from individual subplots
        ax.get_legend().remove()

    # Add a legend to the last subplot
    handles, labels = axes[-1].get_legend_handles_labels()
    # fig.legend(handles, labels, title="Treatment", loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=len(treatment_order))
    fig.legend(handles, labels, title="Treatment", loc="upper right")

    # Improve layout
    plt.tight_layout()

    # Show the plot
    plt.show()


# ==========================
# PART A: Current Year Summary
# ==========================

## ---------------------------------
## A.1 General Graphs
## ---------------------------------
if GENERAL_GRAPHS:
    f_plot_y_by_x_and_z(df_current_yr, y="Yield (t/ha)", x="Current Land Use", z=None)
    f_plot_y_by_x_and_z(df_current_yr, y="Yield (t/ha)", x="Previous Land Use", z="Current Land Use")
    f_plot_y_by_x_and_z(df_current_yr, y="Protein (%)", x="Previous Land Use", z="Current Land Use")
    f_plot_y_by_x_and_z(df_current_yr, y="Screenings (%)", x="Previous Land Use", z="Current Land Use")
    f_plot_y_by_x_and_z(df_current_yr, y="NUE (yield Nf)", x="Previous Land Use", z="Current Land Use")

## ---------------------------------
## A.2 Gross Margin Calculation
## ---------------------------------

# Nitrogen price scenarios
nitrogen_price = {
    "Low": 1200 * 0.7,  # 30% decrease
    "Standard": 1200,
    "High": 1200 * 1.3  # 30% increase
}

# Grain price and quality standards
grain_prices = {
    "Wheat": {
        "APW": {"Price": 375, "Protein Requirement": 10.5, "Max Screenings": 5},
        "ASW": {"Price": 362, "Protein Requirement": 0, "Max Screenings": 5},
        "Feed": {"Price": 335, "Protein Requirement": 0, "Max Screenings": None},
    },
    "Canola": {
        "CAN1": {"Price": 750, "Protein Requirement": None, "Max Screenings": None},
    },
    "Barley": {
        "Malt": {"Price": 341, "Protein Requirement": 9, "Max Screenings": 7},
        "Feed": {"Price": 310, "Protein Requirement": 0, "Max Screenings": 60},
    },
    "Lupin": {
        "Lup1": {"Price": 450, "Protein Requirement": None, "Max Screenings": None},
    }
}

# Variable costs excluding nitrogen
variable_costs = {
    "Wheat": {"Other Fertiliser": 80.00, "Chem": 150.00, "Labour": 30.00, "Machinery": 60.00, "Total": 320.00},
    "Barley": {"Other Fertiliser": 80.00, "Chem": 150.00, "Labour": 30.00, "Machinery": 60.00, "Total": 320.00},
    "Oats": {"Other Fertiliser": 80.00, "Chem": 100.00, "Labour": 30.00, "Machinery": 60.00, "Total": 270.00},
    "Canola": {"Other Fertiliser": 80.00, "Chem": 170.00, "Labour": 30.00, "Machinery": 60.00, "Total": 340.00},
    "Lupin": {"Other Fertiliser": 80.00, "Chem": 150.00, "Labour": 30.00, "Machinery": 60.00, "Total": 320.00},
}


def select_grain_price(crop, protein, screenings):
    """Selects the appropriate grain price based on crop type, protein, and screenings."""
    if crop in grain_prices:
        for grade, specs in grain_prices[crop].items():
            if (specs["Protein Requirement"] is None or protein >= specs["Protein Requirement"]) and \
               (specs["Max Screenings"] is None or screenings <= specs["Max Screenings"]):
                return specs["Price"]
    return 0  # Default if no price found

# Calculate gross margin
def calculate_gross_margin(row, nitrogen_scenario="Standard"):
    crop = row["Current Land Use"]
    grain_price = select_grain_price(crop, row["Protein (%)"], row["Screenings (%)"])
    nitrogen_cost = row["N Rate (kg N/ha)"] * (nitrogen_price[nitrogen_scenario] / 1000)  # Convert to $/ha
    variable_cost = variable_costs[crop]["Total"]
    revenue = row["Yield (t/ha)"] * grain_price
    gross_margin = revenue - (nitrogen_cost + variable_cost)
    return grain_price, gross_margin

# Apply the function to each row and add new columns
df_current_yr[["Grain Price ($/t)", "Gross Margin ($/ha)"]] = df_current_yr.apply(lambda row: pd.Series(calculate_gross_margin(row)), axis=1)

# create a pivot table
pivot_table = round(df_current_yr.pivot_table(index=["Rotation", "Current Land Use"], columns="Treatment", values="Gross Margin ($/ha)", aggfunc="mean"),0)
print(pivot_table)

# plot gm by n and rot
# f_plot_y_by_x_and_z(df_current_yr, y="Gross Margin ($/ha)", x="Previous Land Use", z="Current Land Use")


## ---------------------------------
## A.2 GHG Calculation
## ---------------------------------


# ==========================
# PART B: All Years - Statistical Analysis
# ==========================

## ---------------------------------
## B.1 Correlation Analysis
## ---------------------------------

# Convert categorical variables to numeric codes for correlation analysis
df['Spring rainfall decile'] = df['Spring rainfall decile'].map({'Poor': 0, 'Good': 1})
df['GS rainfall decile'] = df['GS rainfall decile'].map({'Poor': 0, 'Good': 1})
df['Previous Land Use'] = pd.Categorical(df['Previous Land Use']).codes
df['Current Land Use'] = pd.Categorical(df['Current Land Use']).codes

# Correlation matrix between numeric variables
correlation_matrix = df[['N Rate (kg N/ha)', 'Yield (t/ha)', 'Protein (%)', 'Screenings (%)',
                         'Spring rainfall decile', 'GS rainfall decile', 'Previous Land Use', 'Current Land Use']].corr()

print("Correlation Matrix:")
print(correlation_matrix)

# Visualizing the relationships between N Rate, Protein, and Screenings
plt.figure(figsize=(12, 6))
sns.pairplot(df[['N Rate (kg N/ha)', 'Protein (%)', 'Screenings (%)', 'Yield (t/ha)']])
plt.suptitle('Pairplot of N Rate, Protein, Screenings, and Yield', y=1.02)
plt.show()

## ---------------------------------
## B.2 Linear Regression Analysis
## ---------------------------------

# Perform Linear Regression to predict Protein and Screenings from other variables
# Predictor variables (independent)
X = df[['N Rate (kg N/ha)', 'Yield (t/ha)', 'Spring rainfall decile', 'GS rainfall decile', 
        'Previous Land Use', 'Current Land Use']]

# Target variables (dependent)
y_protein = df['Protein (%)']
y_screenings = df['Screenings (%)']

# Fit linear regression model for Protein
model_protein = LinearRegression()
model_protein.fit(X, y_protein)
protein_pred = model_protein.predict(X)

# Fit linear regression model for Screenings
model_screenings = LinearRegression()
model_screenings.fit(X, y_screenings)
screenings_pred = model_screenings.predict(X)

# Print the coefficients for each feature
print("Protein Model Coefficients:")
print(model_protein.coef_)
print("Screenings Model Coefficients:")
print(model_screenings.coef_)

# Summary of results
summary = pd.DataFrame({
    'Feature': ['N Rate (kg N/ha)', 'Yield (t/ha)', 'Spring rainfall decile', 'GS rainfall decile',
                'Previous Land Use', 'Current Land Use'],
    'Protein Coefficients': model_protein.coef_,
    'Screenings Coefficients': model_screenings.coef_
})
print("\nSummary of Features and Coefficients:")
print(summary)






# Define independent variables (categorical variables must be treated as categorical)
df_current_yr = pd.get_dummies(df_current_yr, columns=["Previous Land Use", "Current Land Use"], drop_first=True)

# Define dependent variable (Yield) and independent variables
X = df_current_yr[["N Rate (kg N/ha)"] + [col for col in df_current_yr.columns if "Previous Land Use" in col or "Current Land Use" in col]]
y = df_current_yr["Yield (t/ha)"]

# Add constant term for regression
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X).fit()

# Print results
print(model.summary())



