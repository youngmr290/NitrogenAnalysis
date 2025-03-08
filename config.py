# ==========================
# Analysis assumptions and parameters
# ==========================

# Nitrogen Price Scenarios
nitrogen_price = {
    "Low": 1200 * 0.7,
    "Standard": 1200,
    "High": 1200 * 1.3
}

# Grain Price and Quality Standards
grain_prices = {
    "Wheat": {"APW": {"Price": 375, "Protein Requirement": 10.5, "Max Screenings": 5},
              "ASW": {"Price": 362, "Protein Requirement": 0, "Max Screenings": 5},
              "Feed": {"Price": 335, "Protein Requirement": 0, "Max Screenings": None}},
    "Canola": {"CAN1": {"Price": 750, "Protein Requirement": None, "Max Screenings": None}},
    "Barley": {"Malt": {"Price": 341, "Protein Requirement": 9, "Max Screenings": 7},
               "Feed": {"Price": 310, "Protein Requirement": 0, "Max Screenings": 60}},
    "Lupin": {"Lup1": {"Price": 450, "Protein Requirement": None, "Max Screenings": None}}
}

# Variable Costs Excluding Nitrogen
variable_costs = {
    "Wheat": {"Total": 320.00},
    "Barley": {"Total": 320.00},
    "Oats": {"Total": 270.00},
    "Canola": {"Total": 340.00},
    "Lupin": {"Total": 320.00}
}


# ghg emissions
emission_farm_specific = {
    "N_MacroPro": 10, #10 kg/ha of N come from MacroPro for each crop.
    "lime_applied": 500, #assuming 2t/ha every 4 years
    "diesel_used": 17, #assume 17L/ha for all crops
    "harvest_index": {"Wheat": 0.42, "Barley": 0.44, "Canola": 0.2, "Lupin": 0.3}, #harvest index for each crop
    "F": 0.1, #fraction of crop residue that is burnt (ha burnt/ha harvested).
    "decay_before_burning": 0.25 #fraction of crop residue that is decayed before burning time.
}    
    


emission_coefficients = {

    # 🔹 Greenhouse Gas Global Warming Potentials (GWPs)
    'i_ch4_gwp_factor': 28,  # Global warming potential factor for methane (CH4)
    'i_co2_gwp_factor': 1,  # Global warming potential factor for carbon dioxide (CO2)
    'i_n2o_gwp_factor': 265,  # Global warming potential factor for nitrous oxide (N2O)

    # 🔹 Conversion Factors
    'i_cf_ch4': 1.3333333333333333,  # Conversion factor for CH4 (16/12)
    'i_cf_n2o': 1.5714285714285714,  # Conversion factor for N2O (44/28)
    'i_cf_co2': 3.6666666666666665,  # Conversion factor for CO2 (44/12)

    # 🔹 Emission Factors
    'i_ef_residue': 0.01,  # Emission factor for nitrogen release from residue decomposition
    'i_ef_n2o_burning': 0.0076,  # Emission factor for nitrous oxide (N2O) from burning residue
    'i_ef_ch4_burning': 0.0035,  # Emission factor for methane (CH4) from burning residue
    'i_ef_leach_runoff': 0.011,  # Emission factor for nitrogen lost via leaching and runoff
    'i_ef_fert': 0.0005,  # Emission factor for fertilizer application
    'i_ef_urea': 0.2,  # Emission factor for urea hydrolysis
    'i_ef_limestone': 0.12,  # Emission factor for limestone application
    'i_ef_dolomite': 0.13, #Dolomite Emission Factor (CO2-C/C)
    'i_ef_diesel_co2': 2.7,  # Emission factor for CO2 emissions from diesel combustion
    'i_ef_diesel_ch4': 0.002123,  # Emission factor for CH4 emissions from diesel combustion
    'i_ef_diesel_n2o': 0.01351,  # Emission factor for N2O emissions from diesel combustion

    # 🔹 Residue and Crop Properties
    'i_CCa': 0.4,  # Carbon mass fraction in crop residue
    'i_NCa': 0.006,  # Nitrogen content of above-ground crop residue
    'i_NCb': 0.01,  # Nitrogen content of below-ground crop residue
    'i_Rbg': {'Barley': 0.32, 'Wheat': 0.29, 'Canola': 0.33, 'Lupin': 0.51},  # Below ground-residue to above ground residue ratio
    
    # 🔹 Fraction of Nitrogen Available for Loss Processes
    'i_FracGASM_fert': 0.11,  # Fraction of fertilizer nitrogen volatilized as ammonia
    'i_FracLEACH_fert': 0.24,  # Fraction of applied fertilizer nitrogen lost through leaching
    'i_FracLEACH_residue': 0.24,  # Fraction of residue nitrogen lost through leaching
    'i_FracLime': 1,  # Fraction of lime carbon released as CO2
    'i_purity_limestone': 0.9,
    'i_purity_dolomite': 0.95,
    'i_FracWET_fert': 0.223,  # Fraction of fertilizer nitrogen subject to wetland conditions
    'i_FracWET_residue': 1,  # Fraction of residue nitrogen subject to wetland conditions

    # 🔹 Burning Efficiency Factor
    'i_Z': 0.96  # Burning efficiency for crop residue (fraction of fuel burnt)
}
