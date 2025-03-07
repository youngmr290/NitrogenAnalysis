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
