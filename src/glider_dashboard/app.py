"""Panel application script.

Served by ``glider-dashboard serve``, which runs ``panel serve`` on this file.
Panel executes it once per browser session; the data catalog is cached per
process (see :func:`glider_dashboard.catalog.get_catalog`), so the session
state stays separate for every user while the data is loaded only once.

The config file is taken from the ``GLIDER_DASHBOARD_CONFIG`` environment
variable (set by the CLI), see :mod:`glider_dashboard.config`.
"""

import logging

import panel as pn

from glider_dashboard.catalog import get_catalog
from glider_dashboard.dashboard import GliderDashboard, create_metadata_table

logger = logging.getLogger("glider_dashboard.app")


def exception_handler(ex):
    logger.error("Error", exc_info=ex)
    if dashboard.data_in_view is None:
        pn.state.notifications.error(
            "Please proceed by selecting one or more datasets to display",
            duration=10000,
        )
    else:
        pn.state.notifications.error("Please complete/change input parameters", duration=10000)


pn.extension(
    "plotly",
    "mathjax",
    "tabulator",
    throttled=True,
    template="fast",
    loading_indicator=True,
    exception_handler=exception_handler,
    notifications=True,
    nthreads=0,
)

catalog = get_catalog()
dashboard = GliderDashboard(catalog)


def home():
    pn.Column(
        dashboard.create_app_instance(), "[Open Metadata tables](?page=page1)"
    ).servable(title="Glider Dashboard")


def page1():
    pn.Column(
        create_metadata_table(catalog), " [Return to data visualisation](?page=home)"
    ).servable(title="Metadata tables")


PAGES = {"home": home, "page1": page1}

page_name = pn.state.session_args.get("page", [b"home"])[0].decode("utf8")
PAGES.get(page_name, home)()
