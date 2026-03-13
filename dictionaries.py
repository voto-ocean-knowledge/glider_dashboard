```python
# Import necessary libraries
import cmocean.cm
import numpy as np

# Define a dictionary mapping variable names to their corresponding color maps
def get_color_maps() -> dict:
    """
    Returns a dictionary mapping variable names to their corresponding color maps.
    
    :return: A dictionary where keys are variable names and values are color maps.
    """
    return {
        "conservative_temperature": cmocean.cm.thermal,
        "potential_density": cmocean.cm.dense,
        "temperature": cmocean.cm.thermal,
        "salinity": cmocean.cm.haline,
        "backscatter": cmocean.cm.turbid,
        "backscatter_scaled": cmocean.cm.turbid,
        "cdom": cmocean.cm.matter,
        "fdom": cmocean.cm.haline,
        "chlorophyll": cmocean.cm.algae,
        "phycocyanin": cmocean.cm.algae,
        "phycocyanin_tridente": cmocean.cm.algae,
        "oxygen_concentration": cmocean.cm.amp,
        "N2": cmocean.cm.balance,
        "spice": cmocean.cm.matter,
        "temperature_oxygen": cmocean.cm.thermal,
        "turbidity": cmocean.cm.turbid,
        "profile_num": "hsv",
        "methane_concentration": cmocean.cm.thermal,
        "methane_raw_concentration": cmocean.cm.thermal,
        "longitude": "hsv",
        "latitude": "hsv",
        "u": cmocean.cm.balance,
        "v": cmocean.cm.balance,
    }

# Define a dictionary mapping variable names to their corresponding ranges
def get_variable_ranges() -> dict:
    """
    Returns a dictionary mapping variable names to their corresponding ranges.
    
    :return: A dictionary where keys are variable names and values are tuples representing the lower and upper bounds of the range.
    """
    return {
        "conservative_temperature": (-2, 25),
        "potential_density": (1000, 1035),
        "temperature": (-2, 25),
        "salinity": (0, 35),
        "backscatter": (0, 100),
        "backscatter_scaled": (0, 5e-3),
        "cdom": (0, 2),
        "fdom": (0, 2),
        "chlorophyll": (0, 1.5),
        "phycocyanin": (-1, 3),
        "phycocyanin_tridente": (-1, 3),
        "oxygen_concentration": (0, 400),
        "N2": (0, 1e-3),
        "spice": (0, 5),
        "temperature_oxygen": (-2, 25),
        "turbidity": (0, 1),
        "profile_num": (0, 10000),
        "methane_concentration": (0, 10),
        "methane_raw_concentration": (0, 10),
        "longitude": (0, 360),
        "latitude": (-180, 180),
        "time": (np.datetime64("2020-01-01"), np.datetime64("2026-01-01")),
        "downwelling_PAR": (0, 1),
    }

# Define a dictionary mapping variable names to their corresponding units
def get_units() -> dict:
    """
    Returns a dictionary mapping variable names to their corresponding units.
    
    :return: A dictionary where keys are variable names and values are strings representing the units.
    """
    return {
        "conservative_temperature": "⁰C",
        "potential_density": "kg/m³",
        "temperature": "⁰C",
        "salinity": "g/kg",
        "backscatter": "m⁻¹sr⁻¹",
        "backscatter_scaled": "m⁻¹sr⁻¹",
        "cdom": "mg/m³",
        "fdom": "ppb Quinine Sulfate Equivalent",
        "chlorophyll": "mg/m³",
        "phycocyanin": "mg/m³",
        "phycocyanin_tridente": "mg/m³",
        "oxygen_concentration": "mmol/m³",
        "N2": "s⁻¹",
        "spice": "",
        "temperature_oxygen": "",
        "turbidity": "NTU",
        "profile_num": "",
        "methane_concentration": "mg/m³",
        "methane_raw_concentration": "",
        "longitude": "°E",
        "latitude": "°N",
        "time": "[time]",
        "u": "[m/s]",
        "v": "[m/s]",
        "speed_through_water": "[m/s]",
        "shear_E_mean": "[s-1]",
        "shear_N_mean": "[s-1]",
    }

# Define a list of SAMBA observatories
SAMBA_observatories = [
    "Bornholm Basin",
    "Eastern Gotland",
    "Western Gotland",
    "Skagerrak, Kattegat",
    "Åland Sea",
]

# Example usage:
color_maps = get_color_maps()
variable_ranges = get_variable_ranges()
units = get_units()

print(color_maps)
print(variable_ranges)
print(units)
```