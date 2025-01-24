import logging

import plotly
import dash
import pytz
from dash import dcc
from dash import dash_table
from dash import html
import plotly.express as px
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, ALL
from datetime import datetime
import pandas as pd
import requests
import json
import os.path
import os

import flask


def create_dash_app(requests_pathname_prefix):
    logging.basicConfig(level=10)

    # loading (older) data
    records_path = os.getcwd() + "/" + "records.txt"
    keywords_path = os.getcwd() + "/" + "tracked_keywords.txt"
    if os.path.exists(records_path):
        with open(records_path, "r") as f:
            read_history = json.load(f)
    else:
        read_history = []

    if os.path.exists(keywords_path):
        with open(keywords_path, "r") as f:
            read_keywords = json.load(f)
    else:
        read_keywords = []

    with open("world.json", "r") as f:
        interesting_countries = json.load(f)

    # defining the app layout
    server = flask.Flask(__name__)
    server.secret_key = os.environ.get('secret_key', 'secret')
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                    server=server,
                    requests_pathname_prefix=requests_pathname_prefix,
                    serve_locally=False,
                    meta_tags=[{'name': 'viewport',
                                'content': 'width=device-width, initial-scale=1.0'}])
    app.scripts.config.serve_locally = False
    dcc._js_dist[0]['external_url'] = 'https://cdn.plot.ly/plotly-basic-latest.min.js'

    input_title = html.H6("Supply Two Words", className="card-title ")
    input_fields_row = dbc.Row([
        dbc.Col([
            dbc.Input(id="input-text1", placeholder="1st Word...", type="text", className="pb-2")
        ]),
        dbc.Col([
            dbc.Input(id="input-text2", placeholder="2nd Word...", type="text",
                      className="pb-2")
        ])
    ], justify='center')

    buttons_row = dbc.Row([
        dbc.Button("Analyze", id="analyze-button", color="dark", className="mt-3 mx-2"),
        dbc.Button("Track", id="track-button", color="dark", className="mt-3 mx-2"),
    ], justify='center')
    # the base graph
    sentiments = [0.2, 1.0]
    bugn = px.colors.sequential.BuGn

    bar = go.Bar(x=['Word A', 'Word B'], y=sentiments)
    bar.marker.cmin = 0.0
    bar.marker.cmax = 1.0
    bar.marker.color = sentiments
    bar.marker.colorscale = bugn

    fig = go.Figure(data=[bar])

    fig.update_layout(xaxis_title='',
                      yaxis_title='Sentiment Score', bargap=0.2,  # Gap between bars
                      plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background,
                      autosize=True,
                      )

    warning_div = html.Div(id='warning-message', className='text-danger')
    result_graph = dcc.Graph(id='sentiment-graph', figure=fig, style={"height": '28vh'})

    input_col = dbc.Col(children=[input_title, input_fields_row, warning_div, buttons_row], width=4)
    graph_col = dbc.Col(children=[result_graph], width=8)
    history_row = dbc.Row([
        dbc.Col(html.Div([
            html.H6("Recent Analyses"),
            dash_table.DataTable(
                id='history-table',
                columns=[
                    {'name': 'Timestamp', 'id': 'timestamp'},
                    {'name': 'Keyword', 'id': 'keyword'},
                    {'name': 'positive', 'id': 'positive'},
                    {'name': 'negative', 'id': 'negative'},
                ],
                data=read_history,
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                column_selectable="single",
                page_action="native",
                page_current=0,
                page_size=8,
            )
        ]))
    ])

    input_card_body = dbc.CardBody(children=[dbc.Row([input_col, graph_col, history_row])])

    input_card = dbc.Card(input_card_body, style={'height': '70vh'})

    saot_fig = go.Figure()

    # Update layout of the starting figure
    saot_fig.update_layout(
        title='Sentiment Analysis Over Time',
        xaxis_title='Timestamp',
        yaxis_title='Sentiment Value',
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
        autosize=True,
        yaxis=dict(
            range=[0, 1],  # Set y-axis range from 0 to 1
            tickformat=".1f"  # Optional: Format ticks to one decimal place
        )
    )
    # Keywords timeline
    sentiment_timeline_card = dbc.Card(
        dbc.CardBody(
            [dbc.Row([html.H6("Tracked Keywords"),
                      html.Div(id='word-pool', children=[
                          html.Ul(id='word-pool-list', children=[], style={'display': 'flex', 'flex-wrap': 'wrap'})])
                      ]),
             html.H6("Keyword Sentiment over Time", className="card-title"),
             dbc.Row([
                 dbc.Col(dcc.Graph(id='line-graph', figure=saot_fig))
             ])
             ]
        ),
        className="my-3"
    )

    # History card with DataTable

    main_sentiment_analysis_tab = dcc.Tab(
        id='dash-tab',
        label="Sentiment Analysis Dashboard",
        value='main-tab',
        children=[dbc.Container([
            dbc.Row([
                # Left column (40% width)
                dbc.Col([
                    # Input card (top left, at most 25% width)
                    dbc.Row([
                        dbc.Col(input_card, width=12, align="center"),
                        # dbc.Col(result_card)
                    ], justify='center'),
                    # History card (below input, fills remaining space)
                    # dbc.Row([
                    #     dbc.Col(history_card, align="center")
                    # ], justify='center')
                ], width=5, style={'height': '70vh'}),  # Adjust width for 40% of the screen

                # Right column (57% width)
                dbc.Col([
                    dbc.Row([
                        dbc.Col(sentiment_timeline_card)  # No width specified to fill remaining space
                    ], style={'height': '100%', 'display': 'flex', 'flexDirection': 'column'})
                ], width=7, style={'height': '70vh'})
            ], style={'height': '100%'})
        ], fluid=True)])

    original_timestamp = int(datetime.now().timestamp())

    world_map_sentiment_tab = dcc.Tab(label="World Map",
                                      value='worldmap-tab',
                                      id='worldmap-tab',
                                      children=[dbc.Container([
                                          dbc.Row([
                                              dbc.Col(dcc.Graph(id='world-map', style={"height": "33vw"}))
                                          ]),
                                          dbc.Row([
                                              dbc.Col(
                                                  dcc.Slider(
                                                      id='timestamp-slider',
                                                      min=original_timestamp,
                                                      max=original_timestamp,
                                                      step=None,
                                                      marks=None,
                                                      value=original_timestamp,
                                                  ),
                                                  width={"size": 6, "offset": 3}
                                              ),
                                          ])
                                      ], fluid=True),
                                          html.Div(id='output-world-data')],
                                      style={"height": "100%", "overflow": "auto"}
                                      )

    app.layout = html.Div(children=[html.Div([
        html.H1("Team A4: Federated Sentiment Analysis on Reddit Comment Data", className='text-center py-4 mb-0',
                style={'background-color': 'white', 'color': '#00C1D4'})
    ]),
        dcc.Tabs(id="sentiment-tabs", value='main-tab',
                 children=[main_sentiment_analysis_tab, world_map_sentiment_tab]),
        dcc.Interval(id='interval-component', interval=5 * 1000, n_intervals=0),  # Add Interval component
        dcc.Interval(id='interval-component2', interval=2000, n_intervals=0),  # Add Interval component
        dcc.Store(id='history-store', data=read_history),
        dcc.Store(id='time-series-store', data=[]),
        dcc.Store(id='world-map-store', data=[]),
        dcc.Store(id='bt-series-store', data=[]),
        dcc.Store(id='tracked-keywords-store', data=read_keywords), ])

    ####################################################################################################################
    # defining utility functions
    def map_ts_to_str(key, ts):
        if key == "timestamp":
            return datetime.fromtimestamp(int(ts), tz=pytz.timezone('Europe/Berlin')).strftime('%Y-%m-%d %H:%M:%S')
        else:
            return ts

    def map_to_field(record_list, field):
        return list(map(lambda x: x[field], record_list))


    paths = {
        "time_series": "time_series_pth.json",
        "bt_series": "bt_series_pth.json",
        "world_map": "world_map_pth.json"
    }
    def stash_data(time_series_json, bt_series_json, world_map_json):
        logging.info("loading data to file")
        # Define paths for each JSON data
        nonlocal paths
        # Iterate over each JSON data and save to its corresponding file
        for key, data in zip(paths.keys(), [time_series_json, bt_series_json, world_map_json]):
            try:
                with open(os.getcwd() + "/" + paths[key], 'w') as f:
                    json.dump(data, f)
            except Exception as e:
                logging.exception(e)


    def conditional_load_data():
        nonlocal paths

        loaded_data = []
        for key, path in paths.items():
            try:
                with open(os.getcwd() + "/" + path, 'r') as f:
                    data = json.load(f)
                    loaded_data.append(data)

            except Exception:
                loaded_data.append([])

        return loaded_data

    ####################################################################################################################
    # defining the callbacks
    @app.callback(
        [Output('word-pool-list', 'children'),
         Output('tracked-keywords-store', 'data'),
         Output('warning-message', 'children')],
        [Input('track-button', 'n_clicks'),
         Input({'type': 'word-button', 'index': ALL}, 'n_clicks')],
        [State('input-text1', 'value'),
         State('input-text2', 'value'),
         State('word-pool-list', 'children'),
         State('tracked-keywords-store', 'data')
         ]
    )
    def update_word_pool(n_clicks, word_button_n, new_word1, new_word2, word_pool, tracked_keywords):
        max_keywords = 7
        ctx = dash.callback_context

        if tracked_keywords:
            with open(keywords_path, "w") as file:
                json.dump(tracked_keywords, file)
            if not {child['props']['children'] for child in word_pool}:
                word_pool.extend([dbc.Button(nw, id={'type': 'word-button', 'index': nw},
                                             color='secondary', className='m-1')
                                  for nw in tracked_keywords])
        if not ctx.triggered:
            return word_pool, tracked_keywords, ""

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if 'track-button' in button_id:
            existing_words = {child['props']['children'].strip() for child in word_pool}
            new_words = [dbc.Button(nw.strip(), id={'type': 'word-button', 'index': nw},
                                    color='secondary', className='m-1')
                         for nw in [new_word1, new_word2] if nw and nw not in existing_words]
            if len(existing_words) + len(new_words) > max_keywords:
                return word_pool, tracked_keywords, 'Too many keywords tracked'
            word_pool.extend(new_words)
            tracked_keywords.extend([nw.strip() for nw in [new_word1, new_word2] if nw and nw not in existing_words])
        else:
            button_index = json.loads(button_id)['index']
            # extracting "name" of the button
            tracked_keywords = [tkw for tkw in tracked_keywords if tkw != button_index]

            word_pool = [button for button in word_pool if button['props']['id']['index'] != button_index]

        return word_pool, tracked_keywords, ""

    @app.callback(
        [
            Output('timestamp-slider', 'min'),
            Output('timestamp-slider', 'max'),
            Output('timestamp-slider', 'marks')],
        [Input('world-map-store', 'data'),
         Input('worldmap-tab', 'value'),
         Input('timestamp-slider', 'min'),
         Input('timestamp-slider', 'max'),
         Input('timestamp-slider', 'marks')],
        [State('world-map-store', 'data')]
    )
    def update_slider(n, tab, min_val, max_val, marks, world_map_json):
        # logging.info(n, tab, min_val, max_val, marks, world_map_json)
        if tab != "worldmap-tab":
            return min_val, max_val, marks
        if not world_map_json:
            return min_val, max_val, marks

        min_value = min(map(lambda entry: entry['timestamp'], world_map_json))
        max_value = max(map(lambda entry: entry['timestamp'], world_map_json))

        timestamps = list(map(lambda entry: entry['timestamp'], world_map_json))
        unique_ts = []
        last = -1
        for ts in timestamps:
            if last == ts:
                continue
            else:
                last = ts
                unique_ts += [ts]
        if len(unique_ts) > 25:
            marks = None
        else:
            marks = {
                i: {'label': f"{datetime.fromtimestamp(i, tz=pytz.timezone('Europe/Berlin')).strftime('%Y-%m-%d %H:%M:%S') if isinstance(i, int) else i}"}
                for i in unique_ts}
        return min_value, max_value, marks

    @app.callback(
        Output('world-map', 'figure'),
        [Input('world-map-store', 'data'),
         Input('timestamp-slider', 'value')],
    )
    def update_world_map(world_map_json, selected_time_stamp):

        # getting closest value to timestamp in the data
        if world_map_json:
            selected_time_stamp = min(world_map_json, key=lambda entry: abs(entry["timestamp"] - selected_time_stamp))[
                "timestamp"]

            filtered_wmj = list(filter(lambda entry: entry["timestamp"] == selected_time_stamp, world_map_json))

            # Create a plotly express scatter map

            fig = px.choropleth(pd.DataFrame.from_records(filtered_wmj, columns=["country", "value", "timestamp"]),
                                locations="country",
                                locationmode="country names",
                                color="value",
                                range_color=[0, 1],
                                hover_name="country",
                                projection="natural earth",
                                color_continuous_scale="BuGn")

            fig.update_layout(title="",
                              geo=dict(showframe=False, showcoastlines=False, projection_type='natural earth'))

            return fig

    # Callback to analyze sentiment on button click
    @app.callback(
        [Output('sentiment-graph', 'figure'),
         Output('history-store', 'data')],
        [Input('analyze-button', 'n_clicks')],
        [State('input-text1', 'value'),
         State('input-text2', 'value'),
         State('history-store', 'data'),
         State('sentiment-graph', 'figure'), ]
    )
    def analyze_sentiment(n_clicks, keyword1, keyword2, history_json, sentiment_graph):
        if n_clicks is None or not keyword1 or not keyword2:
            return sentiment_graph, history_json

        timestamp = int(datetime.now().timestamp())
        keywords = [keyword1, keyword2]
        try:
            api_url = "http://federated_sentiment_aggregator:8082/predict_keywords"
            response = requests.post(api_url, json={"text": keywords})
            if response.status_code == 200:
                result = response.json()
                data = json.loads(result)
                results = [{
                    'timestamp': timestamp,
                    'keyword': keyword,
                    'positive': round(data[idx] * 100),
                    'negative': round((1 - data[idx]) * 100),
                    'value': data[idx]
                } for idx, keyword in enumerate(keywords)]

                history_json += results
                history_json = history_json[:50]

            else:
                return [html.P(f"API error: {response.status_code} - {response.text}")], sentiment_graph

            bar = go.Bar(
                x=[result["keyword"] for result in results],
                y=[result["value"] for result in results],
            )
            bar.marker.cmin = 0.0
            bar.marker.cmax = 1.0
            bar.marker.color = [result["value"] for result in results]
            bar.marker.colorscale = bugn

            # Create the figure
            sentiment_fig = go.Figure(data=[bar])

            # Update layout
            sentiment_fig.update_layout(
                xaxis_title='',
                yaxis_title='Sentiment Score',
                bargap=0.2,  # Gap between bars
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
                autosize=True,
                yaxis=dict(
                    range=[0, 1]
                )
            )

            # cast figure to string, then interprete string as dict
            return json.loads(sentiment_fig.to_json()), history_json

        except Exception as e:
            return {'data': [], 'layout': {}}, history_json

    # Callback to update the line graph every few seconds
    @app.callback(
        [Output('line-graph', 'figure'),
         Output('time-series-store', 'data'),
         Output('bt-series-store', 'data'),
         Output('world-map-store', 'data'),
         ],
        [Input('interval-component2', 'n_intervals'),
         ],
        [State('time-series-store', 'data'),
         State('world-map-store', 'data'),
         State('bt-series-store', 'data'),
         State('tracked-keywords-store', 'data'), ]
    )
    def update_line_graph(n_intervals, time_series_json, world_map_json, bt_series_json, tracked_keywords):
        timestamp = int(datetime.now().timestamp())

        if not tracked_keywords:
            with open(keywords_path, "r") as file:
                tracked_keywords = json.load(file)

        if not time_series_json and not bt_series_json:
            logging.info("loading data from file")
            time_series_json, bt_series_json, world_map_json = conditional_load_data()
            logging.info((len(time_series_json), len(bt_series_json), len(world_map_json)))

        if world_map_json:
            world_map_json = [entry for entry in world_map_json if
                              entry["timestamp"] > (timestamp - 3600)]  # only data of the last hours is interesting

        # Fetch new data from API for countries
        api_url = "http://federated_sentiment_aggregator:8082/predict_keywords"

        response_countries = requests.post(api_url, json={"text": interesting_countries})

        if response_countries.status_code == 200:
            data_countries = json.loads(response_countries.json())
            res = [{
                'timestamp': timestamp,
                'country': interesting_countries[idx],
                'value': data_countries[idx]
            } for idx in range(len(interesting_countries))]
            length = len(res)
            # only update, if new data
            if not world_map_json or not all(x == y for x, y in zip(map_to_field(res, 'value'),
                                              map_to_field(world_map_json[:length], 'value'))):
                world_map_json += res

        # Fetch new data from API for trump v biden
        api_url_series = "http://federated_sentiment_aggregator:8082/fetch_time_series"
        response_series = requests.get(api_url_series)
        unique_timestamps = set(map_to_field(bt_series_json, 'timestamp'))
        if response_series.status_code == 200:
            data_series = json.loads(response_series.json())
            logging.info(f"length of data series (bt_series): {len(data_series)}")
            if data_series:
                new_entry = [{
                    'timestamp': datetime.fromtimestamp(entry['timestamp'], tz=pytz.timezone('Europe/Berlin')).strftime("%Y-%m-%d %H:%M:%S"),
                    'biden': entry['biden_sentiment'],
                    'trump': entry['trump_sentiment']
                } for idx, entry in enumerate(data_series)]

                new_entry = list(filter(lambda x: x['timestamp'] not in unique_timestamps, new_entry))
                new_entry.sort(key=lambda x: x['timestamp'], reverse=True)
                bt_series_json += new_entry

        fig_data = [
            {'x': map_to_field(bt_series_json, 'timestamp'),
             'y': map_to_field(bt_series_json, 'biden'),
             'type': 'scatter',
             'mode': 'lines+markers', 'name': 'Sentiment: Biden'},
            {'x': map_to_field(bt_series_json, 'timestamp'),
             'y': map_to_field(bt_series_json, 'trump'),
             'type': 'scatter',
             'mode': 'lines+markers', 'name': 'Sentiment: Trump'}
        ]

        # user-defined keywords

        api_url = "http://federated_sentiment_aggregator:8082/predict_keywords"
        if tracked_keywords:
            response = requests.post(api_url, json={"text": tracked_keywords})
            if response.status_code == 200:
                result = response.json()
                data = json.loads(result)
                entries = [{
                    'timestamp': datetime.fromtimestamp(timestamp, tz=pytz.timezone('Europe/Berlin')).strftime("%Y-%m-%d %H:%M:%S"),
                    'keyword': tracked_keywords[idx],
                    'fvalue': data[idx],
                } for idx in range(len(tracked_keywords)) if tracked_keywords[idx].lower() not in ["trump", "biden"]]

                time_series_json += entries

        for keyword in tracked_keywords:
            keyword_data = list(filter(lambda x: x['keyword'] == keyword, time_series_json))
            keyword_data.sort(key=lambda x: x['timestamp'])
            fig_data.append({
                'x': map_to_field(keyword_data, 'timestamp'),
                'y': map_to_field(keyword_data, 'fvalue'),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': f'Sentiment: {keyword}'
            })
        # add traces to plot
        tl_fig = go.Figure()
        for trace in fig_data:
            tl_fig.add_trace(go.Scatter(x=trace['x'], y=trace['y'], mode=trace['mode'], name=trace['name']))
        # make it so that the plot doesnt change appearance
        tl_fig.update_layout(
            title='Sentiment Analysis Over Time',
            xaxis_title='Timestamp',
            yaxis_title='Sentiment Value',
            plot_bgcolor='rgba(0,0,0,0)',
            autosize=True,
            yaxis=dict(
                range=[0, 1],
                tickformat=".1f"
            )
        )
        stash_data(time_series_json, bt_series_json, world_map_json)
        # same trick as in other plot, return all outputs
        return json.loads(tl_fig.to_json()), time_series_json, bt_series_json, world_map_json

    # Callback to update the DataTable data from the store
    @app.callback(
        Output('history-table', 'data'),
        [Input('history-store', 'data')]
    )
    def update_table_data(store_data):
        if store_data:
            with open(records_path, "w") as file:
                json.dump(store_data, file)
        store_data.sort(key=lambda x: x['timestamp'], reverse=True)

        store_data = list(map(lambda x: {k: map_ts_to_str(k, v) for k, v in x.items()}, store_data))
        return store_data

    return app
