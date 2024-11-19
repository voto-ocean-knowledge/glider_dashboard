import cmocean.cm


cmap_dict = dict(
    conservative_temperature=cmocean.cm.thermal,
    potential_density=cmocean.cm.dense,
    temperature=cmocean.cm.thermal,
    salinity=cmocean.cm.haline,
    backscatter=cmocean.cm.turbid,
    backscatter_scaled=cmocean.cm.turbid,
    cdom=cmocean.cm.matter,
    fdom=cmocean.cm.haline,
    chlorophyll=cmocean.cm.algae,
    phycocyanin=cmocean.cm.algae,
    phycocyanin_tridente=cmocean.cm.algae,
    oxygen_concentration=cmocean.cm.amp,
    N2=cmocean.cm.balance,  # cmocean.cm.amp,
    spice=cmocean.cm.matter,
    temperature_oxygen=cmocean.cm.thermal,
    turbidity=cmocean.cm.turbid,
    profile_num=cmocean.cm.haline,
    methane_concentration=cmocean.cm.thermal,
    methane_raw_concentration=cmocean.cm.thermal,
    longitude='hsv',
    latitude='hsv',
)


units_dict = dict(
    conservative_temperature="⁰C",
    potential_density="kg/m³",
    temperature="⁰C",
    salinity="g/kg",
    backscatter="m⁻¹sr⁻¹",
    backscatter_scaled="m⁻¹sr⁻¹",
    cdom="mg/m³",
    fdom="ppb Quinine Sulfate Equivalent",
    chlorophyll="mg/m³",
    phycocyanin="mg/m³",
    phycocyanin_tridente="mg/m³",
    oxygen_concentration="mmol/m³",
    N2="s⁻¹",  # cmocean.cm.amp,
    spice="",
    temperature_oxygen="",
    turbidity="NTU",
    profile_num="",
    methane_concentration="mg/m³",
    methane_raw_concentration="",
    longitude='°E',
    latitude='°N',
)


SAMBA_observatories = [
    "Bornholm Basin",
    "Eastern Gotland",
    "Western Gotland",
    "Skagerrak, Kattegat",
    "Åland Sea",
]
