import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import seaborn as sns

# Import economic inputs from config.py
from config import nitrogen_price, grain_prices, variable_costs, emission_coefficients, emission_farm_specific

# ==========================
# User Controls
# ==========================
YEAR = 2024
GENERAL_GRAPHS = False
CAN_LEACH = True #Does your crop get enough rainfall or irrigation to drain through the soil profile, i.e. typically above 600mm

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
df[["Grain Price ($/t)", "Gross Margin ($/ha)"]] = df.apply(lambda row: pd.Series(calculate_gross_margin(row)), axis=1)

# create a pivot table - only show current yr
df_current_yr = df[df['Year'] == YEAR] #update df_current to include the gm calcs
pivot_table = round(df_current_yr.pivot_table(index=["Rotation", "Current Land Use"], columns="Treatment", values="Gross Margin ($/ha)", aggfunc="mean"),0)
print(pivot_table)

# plot gm by n and rot
# f_plot_y_by_x_and_z(df_current_yr, y="Gross Margin ($/ha)", x="Previous Land Use", z="Current Land Use")


## ---------------------------------
## A.2 GHG Calculation
## ---------------------------------

def f_n2o_leach_runoff(N, FracWET, FracLEACH):
    '''
    Calculate the nitrous oxide production from leaching and runoff of nitrogen.

    Nitrous oxide production from runoff and leaching (N): N = (F + U) x FracWET x FracLEACH x EF x Cg

    :param N: Nitrogen
    :param ef: Emission factor (EF) (Gg N2O-N/GgN)
    :param FracWET: fraction of N available for leaching and runoff
    :param FracLEACH: fraction of N lost through leaching and runoff
    :return:
    '''
    Cg = emission_coefficients['i_cf_n2o']  # 44/28 - weight conversion factor of Nitrogen (molecular weight 28) to Nitrous oxide (molecular weight 44)
    ef = emission_coefficients['i_ef_leach_runoff']  # emission factor for leaching and runoff of N.
    property_leach_factor = CAN_LEACH  # factor based on rainfall to scale leaching. Typically zones under 600mm annual rainfall dont leach.

    n2o = N * FracWET * FracLEACH * property_leach_factor * ef * Cg

    return n2o


def f_n2o_atmospheric_deposition(N, ef, FracGASM):
    '''
    Calculate the nitrous oxide production from atmospheric deposition due to ammonia released from volatilization
    which increases nitrogen in the nitrogen cycle and therefore increase nitrogen deposition which
    produces some n2o when interacts with the earth.

    Nitrous oxide production from atmospheric deposition: N x FracGASM x EF x Cg

    :param N: Nitrogen
    :param ef: Emission factor (EF) (Gg N2O-N/GgN)
    :param FracGasm: fraction of N volatilised
    :return:
    '''
    Cg = emission_coefficients['i_cf_n2o']  # 44/28 - weight conversion factor of Nitrogen (molecular weight 28) to Nitrous oxide (molecular weight 44)

    n2o = N * FracGASM * ef * Cg

    return n2o


def f_crop_residue_n2o_nir(crop, residue_dm, F, decay_before_burning):
    '''
    Nitrous oxide and methane emissions from crop residues:

        1. the combined nitrification-denitrification process that
           occurs on the nitrogen returned to soil from residues.
        2. Burning of crop residues.
        3. runoff and leaching of nitrogen returned to soil from residues.

    These parameters are hooked up to both the residue production at harvest (+ve) and consumption (-ve) decision variables.
    The AFO equation is a simplified version of the NIR formula below
    because the decision variables are already represented in dry matter and account for removal.

    Mass of N in crop residues returned to soil: M = (P x Rag x (1- F - FFOD) x DM x NCag) +(P x Rag x Rbg x DM x NCbg)

        - P = annual production of crop
        - Rag = residue to crop ratio
        - Rbg = below ground-residue to above ground residue ratio
        - DM = dry matter content
        - NCa = nitrogen content of above-ground crop residue
        - NCb = nitrogen content of below-ground crop residue
        - F= fraction of crop residue that is burnt
        - FFOD = fraction of the crop residue that is removed

    The mass of fuel burnt (M): M = P x R x S x DM x Z x F

        - P = annual production of crop
        - R = residue to crop ratio
        - S = fraction of crop residue remaining at burning
        - DM = dry matter content
        - Z = burning efficiency for residue from crop
        - F = fraction of the annual production of crop that is burnt


    Nitrous oxide production from nitrification-denitrification process (E)	E = M x EF x Cg

    Nitrous oxide production from leaching and runoff (E)	E = M x FracWET x FracLEACH x EF x Cg


    :param residue_dm: dry matter mass of residue decision variable.
    :param F: fraction of crop residue that is burnt (ha burnt/ha harvested).
    :param decay_before_burning: fraction of crop residue that is decayed before burning time.
    :return: Nitrous oxide production from nitrification-denitrification process and nitrous oxide production from leaching and runoff.
    '''
    ##inputs
    Rbg_k = emission_coefficients['i_Rbg'] #below ground-residue to above ground residue ratio
    CCa = emission_coefficients['i_CCa'] #carbon mass fraction in crop residue
    NCa = emission_coefficients['i_NCa'] #nitrogen content of above-ground crop residue
    NCb = emission_coefficients['i_NCb'] #nitrogen content of below-ground crop residue
    Cg_n2o = emission_coefficients['i_cf_n2o']  # 44/28 - weight conversion factor of Nitrogen (molecular weight 28) to Nitrous oxide (molecular weight 44)
    Cg_ch4 = emission_coefficients['i_cf_ch4']  # 16/12 - weight conversion factor of Carbon to Methane
    EF = emission_coefficients['i_ef_residue'] #emision factor for break down of N from residue.
    EF_n2o_burning = emission_coefficients['i_ef_n2o_burning'] #emision factor for n2o for burning residue.
    EF_ch4_burning = emission_coefficients['i_ef_ch4_burning'] #emision factor for ch4 for burning residue.
    FracWET = emission_coefficients['i_FracWET_residue'] #fraction of N available for leaching and runoff
    FracLEACH = emission_coefficients['i_FracLEACH_residue'] #fraction of N lost through leaching and runoff
    Z = emission_coefficients['i_Z'] #burning efficiency for residue from crop (fuel burnt/fuel load)
    n2o_gwp_factor = emission_coefficients['i_n2o_gwp_factor']
    ch4_gwp_factor = emission_coefficients['i_ch4_gwp_factor']
    
    
    ##the formulas used here are slightly different to NIR because we are accounting for decay between harvest and burning

    ##The mass of fuel burnt per tonne of residue at the time of the dv (ie harvest or grazing)
    M_burn = F * Z * residue_dm * decay_before_burning

    ##The mass of N in above and below ground crop residues returned to soils (M).
    ## note, it is correct to multiply fraction burnt with both the production and consumption dv's because the input is fraction of stubble burnt after grazing
    M = ((residue_dm - M_burn) * NCa) + (residue_dm * Rbg_k[crop] * NCb) * (residue_dm>0) #last bit is to make it so that below ground residue is not included in the consumption call

    ##Nitrous oxide production from nitrification-denitrification process
    n2o_residues = M * EF * Cg_n2o

    ##Nitrous oxide production from leaching and runoff
    n2o_leach = f_n2o_leach_runoff(M, FracWET, FracLEACH)

    ##Nitrous oxide production from burning
    n2o_burning = M_burn * NCa * EF_n2o_burning * Cg_n2o

    ##Methane production from burning
    ch4_burning = M_burn * CCa * EF_ch4_burning * Cg_ch4
    
    ##total co2e
    co2e_residue = ((n2o_burning + n2o_leach + n2o_residues) * n2o_gwp_factor
                   + ch4_burning * ch4_gwp_factor)

    return co2e_residue


def f_fuel_emissions(diesel_used):
    '''
    co2, n2o and ch4 emissions from fuel combustion. Assumption in AFO is that all equipment is diesel.

    For some reason in this function, ef also converts to co2e.

    :param diesel_used: L of diesel used by one unit of a given decision variable.
    :return:
    '''

    co2e_ef_diesel_co2 = emission_coefficients['i_ef_diesel_co2']  # Scope 1 Emission Factor CO2-e / L
    co2e_ef_diesel_ch4 = emission_coefficients['i_ef_diesel_ch4']  # Scope 1 Emission Factor CO2-e / L
    co2e_ef_diesel_n2o = emission_coefficients['i_ef_diesel_n2o']  # Scope 1 Emission Factor CO2-e / L

    ##co2e from co2
    co2_fuel_co2e = diesel_used * co2e_ef_diesel_co2

    ##co2e from ch4
    ch4_fuel_co2e = diesel_used * co2e_ef_diesel_ch4

    ##co2e from n2o
    n2o_fuel_co2e = diesel_used * co2e_ef_diesel_n2o

    return co2_fuel_co2e + ch4_fuel_co2e + n2o_fuel_co2e


def f_fert_emissions(nitrogen_applied, propn_urea, lime_applied):
    '''
    Calculates GHG emissions linked to fertiliser applied to rotation activities, using the methods documented
    in the National Greenhouse Gas Inventory Report.

    Emissions are from several exchanges:

        1. the combined nitrification-denitrification process that occurs on the nitrogen in soil.
        2. atmospheric deposition due to ammonia released from the volatilization of fert which increases
           nitrogen in the nitrogen cycle and therefore increase nitrogen deposition which produces some n2o when interacts with the earth.
        3. runoff and leaching of nitrogen.
        4. urea hydrolysis: Urea applied to the soil reacts with water and the soil enzyme urease and is rapidly
           converted to ammonium and bicarbonate.
        5. Liming hydrolysis: The lime dissolves to form calcium, bicarbonate, and hydroxide ions.


    :return: fert co2e kg/ha
    '''
    ef_fert = emission_coefficients['i_ef_fert']
    n2o_gwp_factor = emission_coefficients['i_n2o_gwp_factor']

    ##nitrification
    Cg = emission_coefficients['i_cf_n2o']  # 44/28 - weight conversion factor of Nitrogen (molecular weight 28) to Nitrous oxide (molecular weight 44)
    n2o_fert = nitrogen_applied * ef_fert * Cg

    ##leaching and runoff
    FracWET = emission_coefficients['i_FracWET_fert'] #fraction of N available for leaching and runoff
    FracLEACH = emission_coefficients['i_FracLEACH_fert'] #fraction of N lost through leaching and runoff
    n2o_leach = f_n2o_leach_runoff(nitrogen_applied, FracWET, FracLEACH)

    ##atmospheric
    FracGASM = emission_coefficients['i_FracGASM_fert'] #fraction of animal waste N volatilised
    n2o_atmospheric_deposition = f_n2o_atmospheric_deposition(nitrogen_applied, ef_fert, FracGASM)

    ##urea hydrolysis
    Cg_co2 = emission_coefficients['i_cf_co2']  # 44/12 - weight conversion factor of carbon (molecular weight 12) to carbon dioxide (molecular weight 44)
    ef_urea = emission_coefficients['i_ef_urea']
    urea_applied = nitrogen_applied * propn_urea / 0.46
    co2_urea_application = urea_applied * ef_urea * Cg_co2

    ##lime hydrolysis
    ef_limestone = emission_coefficients['i_ef_limestone']
    ef_dolomite = emission_coefficients['i_ef_dolomite']
    FracLime = emission_coefficients['i_FracLime']
    purity_limestone = emission_coefficients['i_purity_limestone']
    purity_dolomite = emission_coefficients['i_purity_dolomite']
    co2_lime_application = ((lime_applied * FracLime * purity_limestone * ef_limestone)
                            + (lime_applied * (1-FracLime) * purity_dolomite * ef_dolomite)) * Cg_co2

    ##total co2e
    co2e_fert = ((n2o_fert + n2o_leach + n2o_atmospheric_deposition) * n2o_gwp_factor
                   + co2_urea_application + co2_lime_application)

    return co2e_fert



# all the nitrgen is urea except the nitrogen from MacroPro which is ammonium.
df['propn_urea'] = (df['N Rate (kg N/ha)'] - emission_farm_specific["N_MacroPro"])/ df['N Rate (kg N/ha)']

# Function to calculate residue left after harvest
def calculate_residue(row):
    crop = row["Current Land Use"]
    yield_ton = row["Yield (t/ha)"]
    if crop in emission_farm_specific["harvest_index"]:
        total_biomass = yield_ton / emission_farm_specific["harvest_index"][crop]
        residue = total_biomass - yield_ton
        return residue * 1000  # Convert to kg/ha
    else:
        return None  # Return None if crop type is not found

# Apply the function to calculate residue
df["Residue (kg/ha)"] = df.apply(calculate_residue, axis=1)

df["co2e_residue"] = df.apply(lambda row: pd.Series(f_crop_residue_n2o_nir(row["Current Land Use"], row["Residue (kg/ha)"], emission_farm_specific["F"], emission_farm_specific["decay_before_burning"])), axis=1)
df["co2e_fuel"] = f_fuel_emissions(emission_farm_specific["diesel_used"])
df["co2e_fert"] = df.apply(lambda row: pd.Series(f_fert_emissions(row["N Rate (kg N/ha)"], row["propn_urea"], emission_farm_specific["lime_applied"])), axis=1)
df["co2e_total"] = (df["co2e_residue"] + df["co2e_fuel"] + df["co2e_fert"])

df["emission_intensity"] = df["co2e_total"]/1000 / df["Yield (t/ha)"] #calc emission intensity

#display a table for current yr
df_current_yr = df[df['Year'] == YEAR]
# Pivot table to summarize emissions by Rotation and Crop for each Treatment
df_summary = df_current_yr.pivot_table(index=["Rotation", "Current Land Use"], 
                            columns="Treatment", 
                            values=["co2e_total", "emission_intensity"], 
                            aggfunc="mean").reset_index()

# Flatten multi-level column names
df_summary.columns = ["Rotation", "Crop"] + [f"{col[1]} {col[0]}" for col in df_summary.columns[2:]]

# Creating a formatted table with total emissions and emission intensity in brackets
df_summary["H"] = df_summary.apply(lambda row: f"{round(row['H co2e_total'], 0)} ({round(row['H emission_intensity'], 2)})", axis=1)
df_summary["L"] = df_summary.apply(lambda row: f"{round(row['L co2e_total'], 0)} ({round(row['L emission_intensity'], 2)})", axis=1)
df_summary["N"] = df_summary.apply(lambda row: f"{round(row['N co2e_total'], 0)} ({round(row['N emission_intensity'], 2)})", axis=1)

# Selecting and renaming necessary columns
df_summary = df_summary[["Rotation", "Crop", "H", "L", "N"]]

# Display the formatted table
print(df_summary)
df_summary.to_excel("Output/emissions_summary.xlsx", index=False)


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



