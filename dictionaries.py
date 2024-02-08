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
    oxygen_concentration=cmocean.cm.amp,
    N2=cmocean.cm.balance,  # cmocean.cm.amp,
    spice=cmocean.cm.matter,
    temperature_oxygen=cmocean.cm.thermal,
    turbidity=cmocean.cm.turbid,
    profile_num=cmocean.cm.haline,
    methane_concentration=cmocean.cm.thermal,
    methane_raw_concentration=cmocean.cm.thermal
)
