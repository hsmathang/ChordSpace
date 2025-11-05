from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import webbrowser

from services.logic import ExplorationLogic
from tools.query_registry import available_query_names

app = Dash(__name__)
logic = ExplorationLogic()

app.layout = html.Div([
    html.H1(children='Exploration Studio', title='Title'),
    dcc.Store(id='population-data-store'),
    html.Div(className='panels', children=[
        html.Div(id='panel-population-source', className='panel', children=[
            html.H2('Population Source'),
            dcc.Tabs(id='population-source-tabs', value='tab-gen', children=[
                dcc.Tab(label='Load from Database', value='tab-db', children=[
                    html.Div([
                        html.Label('Base Query'),
                        dcc.Dropdown(
                            id='base-query-dropdown',
                            options=[{'label': name, 'value': name} for name in available_query_names()],
                            value=next(iter(available_query_names()), None)
                        ),
                        html.Label('Additional Population Sources'),
                        dcc.Input(id='population-sources-input', type='text'),
                        html.Label('Chord Type'),
                        dcc.Dropdown(
                            id='chord-type-dropdown',
                            options=[
                                {'label': 'A', 'value': 'A'},
                                {'label': 'B', 'value': 'B'},
                                {'label': 'C', 'value': 'C'}
                            ],
                            value='B'
                        )
                    ])
                ]),
                dcc.Tab(label='Combinatorial Generation', value='tab-gen', children=[
                    html.Div([
                        html.Label('Pitch Classes (comma-separated: 0,2,4,5,7,9,11)'),
                        dcc.Input(id='pitch-classes-input', type='text', value='0,2,4,5,7,9,11'),
                        html.Label('Octave Range'),
                        dcc.Input(id='octave-min-input', type='number', value=4),
                        dcc.Input(id='octave-max-input', type='number', value=5),
                        html.Label('Cardinalities (comma-separated: 3,4)'),
                        dcc.Input(id='cardinalities-input', type='text', value='3'),
                        dcc.Checklist(
                            id='edge-pc0-checkbox',
                            options=[{'label': 'Include edge pc0', 'value': 'True'}]
                        ),
                        html.Button('Generate Chords', id='generate-chords-button')
                    ])
                ])
            ]),
            html.Hr(),
            html.H3('Generated Population'),
            dash_table.DataTable(id='population-table')
        ]),
        html.Div(id='panel-shared-filters', className='panel', children=[
            html.H2('Shared Filters'),
            html.P('Content for Shared Filters panel...')
        ]),
        html.Div(id='panel-exploration-studio', className='panel', children=[
            html.H2('Exploration Studio'),
            html.Label('Reduction Algorithm'),
            dcc.Dropdown(id='reduction-dropdown', options=[{'label': 'MDS', 'value': 'MDS'}, {'label': 'UMAP', 'value': 'UMAP'}], value='MDS'),
            html.Label('Dissonance Model'),
            dcc.Dropdown(id='model-dropdown', options=[{'label': 'Sethares', 'value': 'Sethares'}, {'label': 'Euler', 'value': 'Euler'}], value='Sethares'),
            html.Label('Distance Metric'),
            dcc.Dropdown(id='metric-dropdown', options=[{'label': 'euclidean', 'value': 'euclidean'}, {'label': 'cosine', 'value': 'cosine'}], value='euclidean'),
            html.Label('Ponderation'),
            dcc.Dropdown(id='ponderation-dropdown', options=[{'label': 'ninguna', 'value': 'ninguna'}, {'label': 'consonancia', 'value': 'consonancia'}], value='ninguna'),
            html.Label('Comparison Seeds'),
            dcc.Input(id='comparison-seeds-input', type='text', value='42'),
            html.Label('Output Directory'),
            dcc.Input(id='output-dir-input', type='text', value='outputs/exploration_studio'),
            html.Button('Generate Report', id='generate-report-button'),
            html.Div(id='report-status')
        ])
    ])
])

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
        'pitch_classes': {int(pc.strip()) for pc in pitch_classes_str.split(',')},
        'o_min': o_min,
        'o_max': o_max,
        'N': [int(n.strip()) for n in cardinalities_str.split(',')],
        'edge_pc0': bool(edge_pc0)
    }

    generated_chords = logic.generate_chords(generation_params)

    if not generated_chords:
        return []

    records = [{'midi': c.midi, **c.meta} for c in generated_chords]
    return records

@app.callback(
    Output('population-table', 'data'),
    Input('population-data-store', 'data')
)
def update_population_table_view(records):
    if not records:
        return []
    # Display a subset of columns for readability
    df = pd.DataFrame(records)
    display_cols = ['midi', 'n', 'span', 'pc_mask']
    return df[display_cols].to_dict('records')

@app.callback(
    Output('report-status', 'children'),
    Input('generate-report-button', 'n_clicks'),
    State('population-data-store', 'data'),
    State('reduction-dropdown', 'value'),
    State('model-dropdown', 'value'),
    State('metric-dropdown', 'value'),
    State('ponderation-dropdown', 'value'),
    State('comparison-seeds-input', 'value'),
    State('output-dir-input', 'value')
)
def run_report_callback(n_clicks, records, reduction, model, metric, ponderation, seeds, output_dir):
    if n_clicks is None or not records:
        return "Waiting for population to generate report."

    df_override = pd.DataFrame(records)

    report_params = {
        "output_dir": output_dir,
        "reduction": reduction,
        "model": model,
        "metric": metric,
        "ponderation": ponderation,
        "comparison_seeds": seeds,
        "generator_settings": {"label": "Exploration Studio Generation"}
    }

    result = logic.run_report(report_params, df_override=df_override)
    output_path = result['output_dir']
    report_url = f"file://{output_path.resolve()}/scatter.html"

    # This will not automatically open a browser tab from the server.
    # It provides a clickable link for the user.
    return html.A(f"Report generated at: {output_path}", href=report_url, target="_blank")

if __name__ == '__main__':
    app.run(debug=True)
