# Getting started
Welcome to Glider Dashboard. This browser tools is designed to enable fast initial exploration and interpretation of Glider data. It supports rapid generation of contour-plots, scatter-plots (correlations) and a single profile view. Glider Dashboard is free and Open Source. We welcome feedback, bug reports and contributions.  

## 1. Data sources
Glider Dashboard is written to sync regularly with ERDDAP Glider Databases. In order to keep the load and costs for external GDAC maintainers low, we cache downloaded data on our own Glider Dashboard server. Many thousand datasets from the IOOS Glider ERDDAP, VOTO ERDDAP and others are available yet. In case you collect your own Glider datasets, the easiest way to get them into the Glider Dashboard is to upload the data to a Glider GDAC.

## 2. Data selection
Datasets can be selected by clicking on the "Choose dataset(s)" menu. You can either filter by one or multiple DatasetIDs, the data providers institution, or an observatory. After making a valid selection, data should show up in a section-plot over time (colored scatter plot), where it can be explored interactively. 

## 4. Annotations and metadata
Basic annotations (the ID, start and end of each mission) can be activated by ticking the box "show mission name and start" below the section plot. More extensive metadata about the current data can be activated by ticking the "show metadata" box. Here you can also find links to the presented datasets with extensive metadata information. 

## 5. Interacting with the  section plots
### Basic Navigation
The presented plots are interactive. By hovering the mouse over any point, the variable value at that point will be displayed. By clicking at any point, the profile located closest to the mouse click will be presented below the section plots. You can zoom into the data by using the mouse-wheel, pan by holding the left mouse button and drag, or use any of the tools presented above the section plot (e.g. box zoom, save). By long pressing the tools, often more options become available (e.g. vertical paning or box-zoom in y-direction only). 
### Add additional variables
By default, the temperature will be visualised in the section plots. However, any variable that is included as a column in the dataset can be by added to the visualisation. To do so, open the "Section plot options" menu by clicking on it. Add additional variables by clicking in the "variable" input field. You can speed up your search by tiping the first letter on your keyboard to filter the results. 
### Change the colorbar scaling
Some variables (e.g. Photosyntetic Active Radiation PAR) span many orders of magnitude and are therefore easier to interpret on a logarithmic color scale. Other variables (for example salinity) can have large (depth-)gradients that cover small scale variation. For an irregular colorbar with histogram equalisation, activate the "eq-hist" scaling. This can be usefull to see small property variations within the mixed layer. 
### 2D Data aggregation
Usually, each pixel in the section plots consists of multiple datapoints, that are aggregated with the mean() function to present the data in a smooth and thruthfull way. To instead see the variability (std) behind each datapoints, activate the "std" button. 
### Contour lines overlay
The section plots can be overlayed with contour lines. To use the primary variable of the plot for the contour lines, choose the option "same as above" (e.g. temmperature section with temperature contour lines). Alternatively, choose another variable as overlay (e.g. temperature section with density contour lines).
## 5. Add linked scatter (or profile) plots
To add a scatter plot of the data visualised in the section plot, click on the menu "Linked (scatter-) plots" and activate the "Show scatter diagram" toggle. A scatter plot will appear. The scatter plot mirrors the selection of data within the viewport of the section plot. For example, if you zoom into the deep section of the section plot, then also a linked TS-diagram would only show datapoints from the depth. By default, the scatter diagram is colored by the amount (density) of datapoints at each x/y location. However, the scatter plot can alternatively be colored by any variable that can be selected in "Colour scatterplot by". 
### Custom scatter plots
Additional to the predefined TS-diagrams and profile diagrams, a custom selection of variables for the x-axis and y-axis can be chosen by clicking the "custom" button. Possible use cases are spatial plots (x=longitude, y=latitude), potential density instead of depth on the y-axis or having the profile number (instead of time) on the x-axis. The custom scatter plots are also useful to get an idea of any kind of correlation between too variables.
### Enable linked selections
Ticking the box "enable linked selections" adds a new tool to the toolbar in the upper left. The "box select" tool lets you draw a selection in either plot, which will be reflected in all other plots. For example if you draw a small frame around a few profiles in the section plot, the respective profiles will automatically be highlighted in the linked TS-diagram or any other active scatter plot.
