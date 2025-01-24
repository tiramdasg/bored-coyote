import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
from dash.dependencies import Input, Output, State
from datetime import datetime
import requests
import json
import plotly.graph_objs as go


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}])

#input card for two keywords
first_card = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Write down two words", className="card-title"),
            dbc.Input(id="input-text1", placeholder="Type first keyword...", type="text", className="pb-2"),
            dbc.Input(id="input-text2", placeholder="Type second keyword...", type="text", className="pb-2 mt-2"),
            dbc.Button("Analyze", id="analyze-button", color="dark", className="mt-3"),
        ]
    )
)

# overview card
second_card = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Overview", className="card-title"),
            html.P("This card has some text content, but not much else"),
        ]
    )
)

#result card
third_card = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Result", className="card-title"),
            dbc.Row([
                dbc.Col(html.Div(id='output-div', className='text-center'), width=12)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='sentiment-graph'), width=12)
            ]),
        ]
    ),
    className="my-3"
)

#history card with DataTable
forth_card = dbc.Card(
    dbc.CardBody(
        dbc.Row([
            dbc.Col(html.Div([
                html.H5("Recent Analyses", className='text-center'),
                dash_table.DataTable(
                    id='history-table',
                    columns=[
                        {'name': 'Keyword', 'id': 'keyword'},
                        {'name': 'positive', 'id': 'positive'},
                        {'name': 'negative', 'id': 'negative'},
                    ],
                    data=[],  # Initialize with empty data
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    }
                )
            ]), width=12)
        ])
    ), className="my-3"
)

#Biden and Trump timeline card
fifth_card = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Biden and Trump", className="card-title"),
            dbc.Row([
                dbc.Col(dcc.Graph(id='line-graph'), width=12)
            ])
        ]
    ),
    className="my-3"
)

#layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Team A4: Federated Sentiment Analysis on Reddit Comment Data", className='text-center py-4 mb-0', style={'background-color': 'white', 'color': '#00C1D4'}), width=12)
    ]),
    dbc.Row([
        dbc.Col(first_card, width=6),
        dbc.Col(second_card, width=6),
    ]),
    dbc.Row([
        dbc.Col(third_card, width=6),
        dbc.Col(forth_card, width=6),
    ]),
    dbc.Row([
        dbc.Col(fifth_card, width=12),
    ]),
    dbc.Row([
        dbc.Col([
            html.H5("Manage Keywords"),
            dcc.Dropdown(
                id='keyword-dropdown',
                options=[],
                placeholder="Select keywords...",
                multi=True
            )
        ], width=6),
    ]),

    dcc.Interval(id='interval-component', interval=5*1000, n_intervals=0),  #add interval component
    dcc.Store(id='history-store', data=""),  # Initialize with empty data
    dcc.Store(id='time-series-store', data=[]),
    dcc.Store(id='keywords-store', data=[])
], fluid=True)

#merged callback to analyze sentiment and update the sentiment graph
@app.callback(
    [Output('output-div', 'children'),
     Output('sentiment-graph', 'figure'),
     Output('history-store', 'data'),
     Output('keyword-dropdown', 'options'),
     Output('keywords-store', 'data')],
    [Input('analyze-button', 'n_clicks'),
     Input('keyword-dropdown', 'value')],
    [State('input-text1', 'value'),
     State('input-text2', 'value'),
     State('history-store', 'data'),
     State('keywords-store', 'data')]
)
def analyze_sentiment(n_clicks, selected_keywords, keyword1, keyword2, history_json, keywords_data):
    ctx = dash.callback_context
    triggered_input = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_input == 'analyze-button':
        if n_clicks is None or not keyword1 or not keyword2:
            return "", {'data': [], 'layout': {}}, history_json, [], keywords_data

        # Deserialize history from JSON
        try:
            history = json.loads(history_json)
        except json.JSONDecodeError:
            history = []

        results = []
        try:
            for keyword in [keyword1, keyword2]:
                api_url = "http://localhost:8082/predict_keywords"
                response = requests.post(api_url, json={"text": keyword})
                if response.status_code == 200:
                    result = response.json()
                    data = json.loads(result)
                    positive_percentage = round(data[0] * 100)
                    negative_percentage = round((1 - data[0]) * 100)

                    result = {
                        "positive": positive_percentage,
                        "negative": negative_percentage
                    }
                    positive = result.get("positive")
                    negative = result.get("negative")
                else:
                    return [html.P(f"API error: {response.status_code} - {response.text}")], {'data': [], 'layout': {}}, history_json, [], keywords_data

                results.append({
                    'keyword': keyword,
                    'positive': positive,
                    'negative': negative,
                })

                history.append({
                    'keyword': keyword,
                    'positive': positive,
                    'negative': negative
                })

            history_json = json.dumps(history)

            result_text = [html.H4("Sentiment Analysis Results")]
            for result in results:
                result_text.extend([
                    html.P(f"Keyword: {result['keyword']}"),
                    html.P(f"positive: {result['positive']}"),
                    html.P(f"negative: {result['negative']}"),
                    html.Hr()
                ])

            sentiment_fig = {
                'data': [
                    {'x': ['positive', 'negative'],
                     'y': [results[0]['positive'], results[0]['negative']],
                     'type': 'bar',
                     'name': f'Sentiment: {results[0]["keyword"]}'},
                    {'x': ['positive', 'negative'],
                     'y': [results[1]['positive'], results[1]['negative']],
                     'type': 'bar',
                     'name': f'Sentiment: {results[1]["keyword"]}'}
                ],
                'layout': {
                    'title': 'Sentiment Analysis Visualization'
                }
            }

            keywords_data.extend([keyword1, keyword2])
            keyword_options = [{'label': kw, 'value': kw} for kw in keywords_data]

            return result_text, sentiment_fig, history_json, keyword_options, keywords_data

        except Exception as e:
            return [html.P(f"An error occurred: {str(e)}")], {'data': [], 'layout': {}}, history_json, [], keywords_data

    elif triggered_input == 'keyword-dropdown':
        if not selected_keywords:
            return dash.no_update, {'data': [], 'layout': {}}, dash.no_update, dash.no_update, dash.no_update

        history = json.loads(history_json)
        selected_data = [entry for entry in history if entry['keyword'] in selected_keywords]

        if not selected_data:
            return dash.no_update, {'data': [], 'layout': {}}, dash.no_update, dash.no_update, dash.no_update

        sentiment_fig = {
            'data': [
                {'x': ['positive', 'negative'], 'y': [entry['positive'], entry['negative']], 'type': 'bar',
                 'name': f'Sentiment: {entry["keyword"]}'} for entry in selected_data
            ],
            'layout': {
                'title': 'Sentiment Analysis Visualization'
            }
        }

        return dash.no_update, sentiment_fig, dash.no_update, dash.no_update, keywords_data

    else:
        return dash.no_update, {'data': [], 'layout': {}}, dash.no_update, dash.no_update, dash.no_update

# Callback to update the history dispaly
@app.callback(
    Output('history-table', 'data'),
    [Input('history-store', 'data')]
)
def update_history(history_json):
    # Deserialize history from JSON
    try:
        history = json.loads(history_json)
    except json.JSONDecodeError:
        history = []

    return history

# Callback to update the line graph every 5 seconds
@app.callback(
    [Output('line-graph', 'figure'),
     Output('time-series-store', 'data')],
    [Input('interval-component', 'n_intervals')],
    [State('time-series-store', 'data')]
)

def update_line_graph(n_intervals, time_series):
    try:
        api_url_series = "http://localhost:8082/fetch_time_series"
        response_series = requests.get(api_url_series)
        data_series = json.loads(response_series.json())[0]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        time_series.append({
            'timestamp': timestamp,
            'biden': data_series['biden_sentiment'],
            'trump': data_series['trump_sentiment'],
        })

        # time_series_json = json.dumps(time_series)

        timestamps = [item['timestamp'] for item in time_series]
        biden_sentiments = [item['biden'] for item in time_series]
        trump_sentiments = [item['trump'] for item in time_series]

        line_fig = {
            'data': [
                {'x': timestamps, 'y': biden_sentiments, 'type': 'line', 'name': 'Biden'},
                {'x': timestamps, 'y': trump_sentiments, 'type': 'line', 'name': 'Trump'},
            ],
            'layout': {
                'title': 'Sentiment Analysis Over Time',
                'xaxis': {'title': 'Analysis Index'},
                'yaxis': {'title': 'Value'}
            }
        }

        return line_fig, time_series

    except Exception as e:
        return {'data': [], 'layout': {'title': f"An error occurred: {str(e)}"}}



if __name__ == '__main__':
    app.run_server(debug=True, port=8001)
