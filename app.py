import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import webbrowser

from services.logic import ExplorationLogic
from tools.query_registry import available_query_names

app = Dash(__name__, external_stylesheets=[dbc.themes.VAPOR])
logic = ExplorationLogic()

# --- Componentes de la UI ---

db_tab = dbc.Card(
    dbc.CardBody([
        dbc.Row([
            dbc.Col(html.Label('Base Query'), width=12),
            dbc.Col(dcc.Dropdown(
                id='base-query-dropdown',
                options=[{'label': name, 'value': name} for name in available_query_names()],
                value=next(iter(available_query_names()), None)
            ), width=12),
        ]),
        dbc.Row([
            dbc.Col(html.Label('Additional Population Sources'), width=12),
            dbc.Col(dbc.Input(id='population-sources-input', type='text'), width=12),
        ]),
        dbc.Row([
            dbc.Col(html.Label('Chord Type'), width=12),
            dbc.Col(dcc.Dropdown(
                id='chord-type-dropdown',
                options=[{'label': t, 'value': t} for t in ['A', 'B', 'C']],
                value='B'
            ), width=12),
        ]),
    ])
)

gen_tab = dbc.Card(
    dbc.CardBody([
        dbc.Row([
            dbc.Col(html.Label('Pitch Classes (e.g., 0,2,4,5,7,9,11)'), width=12),
            dbc.Col(dbc.Input(id='pitch-classes-input', type='text', value='0,2,4,5,7,9,11'), width=12),
        ]),
        dbc.Row([
            dbc.Col(html.Label('Octave Range'), width=6),
            dbc.Col(html.Label('Cardinalities'), width=6),
        ]),
        dbc.Row([
            dbc.Col(dbc.Input(id='octave-min-input', type='number', value=4), width=3),
            dbc.Col(dbc.Input(id='octave-max-input', type='number', value=5), width=3),
            dbc.Col(dbc.Input(id='cardinalities-input', type='text', value='3'), width=6),
        ]),
        dbc.Checklist(
            id='edge-pc0-checkbox',
            options=[{'label': 'Include edge pc0', 'value': 'True'}],
            inline=True,
            switch=True,
        ),
        html.Br(),
        dbc.Button('Generate Chords', id='generate-chords-button', color="primary", className="d-grid gap-2 col-6 mx-auto")
    ])
)

# --- Layout de la App ---

app.layout = dbc.Container([
    dcc.Store(id='population-data-store'),
    dbc.Row(dbc.Col(html.H1('Exploration Studio'), width=12)),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('Population Source'),
                dbc.CardBody(
                    dcc.Tabs(id='population-source-tabs', value='tab-gen', children=[
                        dcc.Tab(label='From Database', value='tab-db', children=db_tab),
                        dcc.Tab(label='Combinatorial', value='tab-gen', children=gen_tab),
                    ])
                ),
            ]),
            html.Br(),
            dbc.Card([
                dbc.CardHeader('Shared Filters'),
                dbc.CardBody([
                    html.P("Future shared filters will be placed here.", className="card-text"),
                ])
            ]),
        ], width=4),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader('Exploration Studio'),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(html.Label('Reduction Algorithm'), width=6),
                        dbc.Col(html.Label('Dissonance Model'), width=6),
                    ]),
                    dbc.Row([
                        dbc.Col(dcc.Dropdown(id='reduction-dropdown', options=[{'label': v, 'value': v} for v in ['MDS', 'UMAP']], value='MDS'), width=6),
                        dbc.Col(dcc.Dropdown(id='model-dropdown', options=[{'label': v, 'value': v} for v in ['Sethares', 'Euler']], value='Sethares'), width=6),
                    ]),
                    dbc.Row([
                        dbc.Col(html.Label('Distance Metric'), width=6),
                        dbc.Col(html.Label('Ponderation'), width=6),
                    ]),
                    dbc.Row([
                        dbc.Col(dcc.Dropdown(id='metric-dropdown', options=[{'label': v, 'value': v} for v in ['euclidean', 'cosine']], value='euclidean'), width=6),
                        dbc.Col(dcc.Dropdown(id='ponderation-dropdown', options=[{'label': v, 'value': v} for v in ['ninguna', 'consonancia']], value='ninguna'), width=6),
                    ]),
                    html.Label('Comparison Seeds'),
                    dbc.Input(id='comparison-seeds-input', type='text', value='42'),
                    html.Label('Output Directory'),
                    dbc.Input(id='output-dir-input', type='text', value='outputs/exploration_studio'),
                    html.Br(),
                    dbc.Checklist(
                        id='transpose-checkbox',
                        options=[{'label': 'Enable Transposition', 'value': 'True'}],
                        inline=True,
                        switch=True,
                    ),
                    dbc.Input(id='transpose-steps-input', type='text', value='0-11', placeholder='e.g., 0-11 or 1,3,5'),
                    html.Br(),
                    dbc.Button('Generate Report', id='generate-report-button', color="success", className="d-grid gap-2"),
                    html.Div(id='report-status', style={'marginTop': '10px'}),
                ])
            ]),
        ], width=8),
    ]),
    dbc.Row(dbc.Col(
        dash_table.DataTable(
            id='population-table',
            page_size=15,
            style_table={'overflowX': 'auto'},
            style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
            style_cell={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'},
        ),
        width=12
    ), style={'marginTop': '20px'}),
], fluid=True)


# --- Callbacks ---

@app.callback(
    Output('population-data-store', 'data'),
    Input('generate-chords-button', 'n_clicks'),
    State('pitch-classes-input', 'value'),
    State('octave-min-input', 'value'),
    State('octave-max-input', 'value'),
    State('cardinalities-input', 'value'),
    State('edge-pc0-checkbox', 'value')
)
def update_population_data(n_clicks, pitch_classes_str, o_min, o_max, cardinalities_str, edge_pc0):
    if n_clicks is None:
        return []
    generation_params = {
        'pitch_classes': {int(pc.strip()) for pc in pitch_classes_str.split(',') if pc.strip()},
        'o_min': o_min, 'o_max': o_max,
        'N': [int(n.strip()) for n in cardinalities_str.split(',') if n.strip()],
        'edge_pc0': bool(edge_pc0)
    }
    generated_chords = logic.generate_chords(generation_params)
    return [{'midi': c.midi, **c.meta} for c in generated_chords] if generated_chords else []

@app.callback(
    Output('population-table', 'data'),
    Output('population-table', 'columns'),
    Input('population-data-store', 'data')
)
def update_population_table_view(records):
    if not records:
        return [], []
    df = pd.DataFrame(records)
    for col in ['midi', 'octave_vector', 'pc_tuple_canon0', 'struct_id', 'chroma01']:
        if col in df.columns:
            df[col] = df[col].astype(str)
    display_cols = ["midi", "n", "span", "pc_mask", "pc_tuple_canon0", "struct_id"]
    final_cols = [col for col in display_cols if col in df.columns]
    return df[final_cols].to_dict('records'), [{"name": i, "id": i} for i in final_cols]

@app.callback(
    Output('report-status', 'children'),
    Input('generate-report-button', 'n_clicks'),
    State('population-data-store', 'data'),
    State('reduction-dropdown', 'value'),
    State('model-dropdown', 'value'),
    State('metric-dropdown', 'value'),
    State('ponderation-dropdown', 'value'),
    State('comparison-seeds-input', 'value'),
    State('output-dir-input', 'value'),
    State('transpose-checkbox', 'value'),
    State('transpose-steps-input', 'value')
)
def run_report_callback(n_clicks, records, reduction, model, metric, ponderation, seeds, output_dir, transpose_enabled, transpose_steps):
    if n_clicks is None or not records:
        return "Waiting for population to generate report."

    df_override = pd.DataFrame(records)

    # Placeholder for transposition logic
    if transpose_enabled:
        print(f"Transposition enabled with steps: {transpose_steps}")
        # Here you would add the logic to transpose the chords in df_override

    report_params = {
        "output_dir": output_dir, "reduction": reduction, "model": model,
        "metric": metric, "ponderation": ponderation, "comparison_seeds": seeds,
        "generator_settings": {"label": "Exploration Studio Generation"}
    }
    result = logic.run_report(report_params, df_override=df_override)
    output_path = result['output_dir']
    report_url = f"file://{output_path.resolve()}/scatter.html"
    return html.A(f"Report generated at: {output_path}", href=report_url, target="_blank")

if __name__ == '__main__':
    app.run(debug=True)
