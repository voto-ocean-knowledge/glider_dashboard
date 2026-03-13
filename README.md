# Glider Explorer
================

## Overview
--------

Glider Explorer is a Python program designed to visualize large amounts of glider data interactively. This application allows users to explore and analyze glider data in a user-friendly and intuitive manner.

## Requirements
------------

* Python 3.x
* Panel library for interactive visualization
* Other dependencies as specified in `requirements.txt`

## Installation
------------

To install the required dependencies, run the following command in your terminal:

```bash
pip install -r requirements.txt
```

## Running the Application
-------------------------

To serve the application in a multi-user environment, use the following command:

```bash
panel serve glider_explorer.py
```

This will start the application in a separate process, ensuring that each user has their own session state.

Alternatively, if you prefer to serve the application from within the script using the `pn.serve()` function, you can do so by running:

```bash
python glider_explorer.py
```

However, please note that this approach will share the session state across users, which may lead to unexpected behavior and irritation.

## Usage
-----

Once the application is running, you can interact with it by:

* Navigating through the visualization using the toolbar and menus
* Selecting different data sets and parameters to explore
* Customizing the visualization to suit your needs

## Contributing
------------

If you'd like to contribute to the development of Glider Explorer, please fork the repository and submit a pull request with your changes. We welcome feedback and suggestions for improving the application.

## License
-------

Glider Explorer is released under the [MIT License](https://opensource.org/licenses/MIT).