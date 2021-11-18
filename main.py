import dash
import dash_bootstrap_components as dbc
from dash_viz import main_dash

external = [dbc.themes.SLATE]
app = dash.Dash(__name__, external_stylesheets=external)
server = app.server

app = main_dash(app)

if __name__ == '__main__':
    app.run_server()
