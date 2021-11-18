import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from generation_images import function_generate_image


def main_dash(app: dash) -> dash:
    app.layout = html.Div([
        dbc.Card([
            dbc.CardHeader("Images generator", style={'textAlign': 'center', 'fontSize': 40}),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Choose the images you want to create : ", html_for="id_model_selector",
                                  style={'textAlign': 'center', 'fontSize': 20}),
                        dcc.Dropdown(
                            options=[
                                {'label': 'Car', 'value': 'cars'},
                                {'label': 'Portrait', 'value': 'portrait'},
                                {'label': 'Horse', 'value': 'horses'}
                            ],
                            value='cars',
                            id='id_model_selector',
                            style={
                                'textAlign': 'center', 'width': '70%'
                            }
                        )
                    ], width={"size": 4, "offset": 1}),
                    dbc.Col([
                        dbc.Label("Number of images : ", id='id_label_slider', html_for="id_slider",
                                  style={'textAlign': 'center', 'fontSize': 20}),
                        dbc.Input(type="range", id="id_slider", min=2, max=10, step=2, value=4,
                                  style={'width': '80%'}, className="form-group"),
                    ], width=4),
                    dbc.Col([
                        dbc.Button("Generate", className="btn btn-primary",
                                   id='id_generate_button', size="lg", style={'margin-top': '25%'})
                    ], width=1)
                ], align='center', className="g-0"),
                html.Br(),
                dbc.Row([
                    dbc.Col([
                        dbc.Spinner(children=[dcc.Graph(id="id_img")], spinnerClassName="spinner",
                                    type=None)
                    ], width={"size": 10, "offset": 1})
                ], align='stretch')

            ])
        ], className=["h-100 d-inline-block", "w-100 p-3"])
    ])

    @app.callback(Output('id_label_slider', 'children'),
                  [Input('id_slider', 'value')])
    def updateslidervalue(value_slider: str):
        return "Number of images : " + str(value_slider)

    @app.callback(Output('id_img', 'figure'),
                  [Input('id_generate_button', 'n_clicks')],
                  [State('id_model_selector', 'value'), State('id_slider', 'value')])
    def showimages(n_clicks: int, value_dropdown: str, value_slider: int):
        return function_generate_image(value_dropdown, value_slider)

    return app
