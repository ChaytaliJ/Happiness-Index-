import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dash
from dash import dcc, html, Input, Output
import plotly.express as px  # Import Plotly Express for scatter plot visualization
import joblib
from flask import Flask, send_file


# Initialize the Flask app
server = Flask(__name__)

# Load the datasets
df = pd.read_csv('C:\\Users\\Chaytu\\OneDrive\\Desktop\\DAV\\happiness_index_report.csv')
df1 = pd.read_csv('C:\\Users\\Chaytu\\OneDrive\\Desktop\\DAV\\world_happiness_corruption.csv')

# Load the trained model
model = joblib.load('happiness_prediction_model.pkl')

# Initialize the Dash app
app = dash.Dash(__name__,server=server,suppress_callback_exceptions=True)

# Define the layout of the dashboard page
app.layout = html.Div(
    children=[
        html.H1("Happiness Index Analysis Dashboard"),
        html.Div([
              dcc.Link('Happiness Index', href='/happiness-info'),
            dcc.Graph(
                id='happiness-line-chart',
                figure={
                    'data': [
                        {'x': df['Year'], 'y': df.groupby('Year')['Index'].mean(), 'type': 'line', 'name': 'Happiness Index'},
                    ],
                    'layout': {
                        'title': 'Average Happiness Index Over the Years',
                        'xaxis': {'title': 'Year'},
                        'yaxis': {'title': 'Average Happiness Index'},
                        'plot_bgcolor': '#f0f0f0',
                        'paper_bgcolor': '#f8f8f8',
                    }
                }
            ),
            dcc.Graph(
                id='top-10-happiest-countries',
                figure={
                    'data': [
                        {'x': df['Country'], 'y': df['Index'], 'type': 'bar', 'name': 'Happiness Index'},
                    ],
                    'layout': {
                        'title': 'Top 10 Happiest Countries',
                        'xaxis': {'title': 'Country'},
                        'yaxis': {'title': 'Happiness Index'},
                        'plot_bgcolor': '#f0f0f0',
                        'paper_bgcolor': '#f8f8f8',
                    }
                }
            ),
            html.Div([
                dcc.Dropdown(
                    id='country-dropdown',
                    options=[{'label': country, 'value': country} for country in df['Country'].unique()],
                    value=df['Country'].unique()[0],
                    style={'width': '50%'}
                ),
                dcc.Graph(
                    id='happiness-index-trend',
                ),
                
                # Dropdowns to select parameters for correlation graph
                html.H3("Relations between two variables"),
                dcc.Dropdown(
                    id='param1-dropdown',
                    options=[
                        {'label': 'GDP per Capita', 'value': 'gdp_per_capita'},
                        {'label': 'Family', 'value': 'family'},
                        {'label': 'Health', 'value': 'health'},
                        {'label': 'Freedom', 'value': 'freedom'},
                        {'label': 'Generosity', 'value': 'generosity'},
                        {'label': 'Government Trust', 'value': 'govtrust'},
                        {'label': 'Dystopia Residual', 'value': 'dystopia_residual'},
                        # Add more options for other parameters
                    ],
                    value='gdp_per_capita'  # Default value
                ),
                dcc.Dropdown(
                    id='param2-dropdown',
                    options=[
                        {'label': 'GDP per Capita', 'value': 'gdp_per_capita'},
                        {'label': 'Family', 'value': 'family'},
                        {'label': 'Health', 'value': 'health'},
                        {'label': 'Freedom', 'value': 'freedom'},
                        {'label': 'Generosity', 'value': 'generosity'},
                        {'label': 'Government Trust', 'value': 'govtrust'},
                        {'label': 'Dystopia Residual', 'value': 'dystopia_residual'},
                        # Add more options for other parameters
                    ],
                    value='health'  # Default value
                ),
                dcc.Graph(
                    id='correlation-graph',
                ),
                
                # Inputs and button for predicting happiness score
              html.Div([
                    html.H3("Predict Happiness Score on user input"),
                    html.Div([
                        html.Label("GDP per Capita", style={'margin-right': '10px'}),
                        dcc.Input(id='gdp-input', type='number', value=1.48),
                    ], style={'margin-bottom': '10px'}),
                    html.Div([
                        html.Label("Family", style={'margin-right': '10px'}),
                        dcc.Input(id='family-input', type='number', value=1.46),
                    ], style={'margin-bottom': '10px'}),
                    html.Div([
                        html.Label("Health", style={'margin-right': '10px'}),
                        dcc.Input(id='health-input', type='number', value=0.81),
                    ], style={'margin-bottom': '10px'}),
                    html.Div([
                        html.Label("Freedom", style={'margin-right': '10px'}),
                        dcc.Input(id='freedom-input', type='number', value=0.57),
                    ], style={'margin-bottom': '10px'}),
                    html.Div([
                        html.Label("Generosity", style={'margin-right': '10px'}),
                        dcc.Input(id='generosity-input', type='number', value=0.32),
                    ], style={'margin-bottom': '10px'}),
                    html.Div([
                        html.Label("Government Trust", style={'margin-right': '10px'}),
                        dcc.Input(id='govtrust-input', type='number', value=0.22),
                    ], style={'margin-bottom': '10px'}),
                    html.Div([
                        html.Label("Dystopia Residual", style={'margin-right': '10px'}),
                        dcc.Input(id='dystopia-input', type='number', value=2.14),
                    ], style={'margin-bottom': '10px'}),
                    html.Button('Predict', id='predict-button', n_clicks=0, style={'margin-top': '10px'}),
                    html.Div(id='prediction-output', style={'margin-top': '10px'})
                ], 
                    style={'padding': '20px', 'border': '1px solid #ddd', 'border-radius': '5px', 'background-color': '#f9f9f9'}),
               html.Div([
                html.H3("World Happiness Map"),
                dcc.Graph(
                    id='world-happiness-map'
                )
                ],
                style={'padding-top': '20px'}),
            
                
                # Dropdowns to select outlier type and parameter
                # html.Div([
                #     html.H3("Outlier Analysis"),
                #     dcc.Dropdown(
                #         id='outlier-selector',
                #         options=[
                #             {'label': 'High Outliers', 'value': 'high'},
                #             {'label': 'Low Outliers', 'value': 'low'},
                #             {'label': 'Both', 'value': 'both'},
                #         ],
                #         value='high',  # Default value
                #         style={'width': '50%'}
                #     ),
                #     dcc.Dropdown(
                #         id='outlier-param-selector',
                #         options=[
                #             {'label': 'GDP per Capita', 'value': 'gdp_per_capita'},
                #             {'label': 'Family', 'value': 'family'},
                #             {'label': 'Health', 'value': 'health'},
                #             {'label': 'Freedom', 'value': 'freedom'},
                #             {'label': 'Generosity', 'value': 'generosity'},
                #             {'label': 'Government Trust', 'value': 'govtrust'},
                #             {'label': 'Dystopia Residual', 'value': 'dystopia_residual'},
                #         ],
                #         value='happiness_score',  # Default value
                #         style={'width': '50%'}
                #     ),
                #     dcc.Graph(
                #         id='outlier-graph'
                #     ),
                # ])
                
        
    ], style={'padding-top': '50px'})  # Add padding after top 10 happiest countries graph
        ])
    ],
    style={"padding": "50px"}
    )

# Define callback to update happiness index trend graph based on selected country
@app.callback(
    Output('happiness-index-trend', 'figure'),
    [Input('country-dropdown', 'value')]
)
def update_happiness_index_trend(selected_country):
    country_df = df[df['Country'] == selected_country]
    fig = {
        'data': [
            {'x': country_df['Year'], 'y': country_df['Index'], 'type': 'line', 'name': 'Happiness Index'},
        ],
        'layout': {
            'title': f'Happiness Index Trend for {selected_country}',
            'xaxis': {'title': 'Year'},
            'yaxis': {'title': 'Happiness Index'},
            'plot_bgcolor': '#f0f0f0',
            'paper_bgcolor': '#f8f8f8',
        }
    }
    return fig

# Define callback to update correlation graph based on selected parameters
@app.callback(
    Output('correlation-graph', 'figure'),
    [Input('param1-dropdown', 'value'),
     Input('param2-dropdown', 'value')]
)
def update_correlation_graph(param1, param2):
    # Filter dataframe to include only the selected parameters
    df_subset = df1[[param1, param2]]
    # Drop rows with missing values
    df_subset.dropna(inplace=True)
    
    # Plot a scatter plot to visualize the correlation
    fig = px.scatter(df_subset, x=param1, y=param2, title=f'Correlation between {param1} and {param2}')
    
    return fig



# # Define callback to update outlier graph based on selected outlier type
# @app.callback(
#     Output('outlier-graph', 'figure'),
#     [Input('outlier-selector', 'value'),
#      Input('outlier-param-selector', 'value')]
# )
# def update_outlier_graph(outlier_type, outlier_param):
#     # Filter dataframe based on selected outlier type
#     if outlier_type == 'high':
#         df_outliers = df1[df1[outlier_param] > df1[outlier_param].quantile(0.75)]
#     elif outlier_type == 'low':
#         df_outliers = df1[df1[outlier_param] < df1[outlier_param].quantile(0.25)]
#     else:  # Both
#         high_outliers = df1[df1[outlier_param] > df1[outlier_param].quantile(0.75)]
#         low_outliers = df1[df1[outlier_param] < df1[outlier_param].quantile(0.25)]
#         df_outliers = pd.concat([high_outliers, low_outliers])
    
#     # Plot a scatter plot to visualize the outliers
#     fig = px.scatter(df_outliers, x='Country', y=outlier_param, title=f'{outlier_type.capitalize()} Outliers for {outlier_param}')
    
#     return fig



# Define callback to update top 10 happiest countries bar graph
@app.callback(
    Output('top-10-happiest-countries', 'figure'),
    [Input('happiness-line-chart', 'clickData')]  # Trigger the callback on clicking the line chart
)
def update_top_10_happiest_countries(clickData):
    # Check if any data is clicked on the line chart
    if clickData is not None:
        year = clickData['points'][0]['x']  # Get the clicked year
        top_10_countries = df[df['Year'] == year].nlargest(10, 'Index')  # Filter to include top 10 happiest countries for the selected year
    else:
        latest_year = df['Year'].max()  # If no data clicked, use the latest year
        top_10_countries = df[df['Year'] == latest_year].nlargest(10, 'Index')  # Filter to include top 10 happiest countries for the latest year

    # Create the bar graph figure
    fig = {
        'data': [
            {'x': top_10_countries['Country'], 'y': top_10_countries['Index'], 'type': 'bar', 'name': 'Happiness Index'},
        ],
        'layout': {
            'title': 'Top 10 Happiest Countries',
            'xaxis': {'title': 'Country'},
            'yaxis': {'title': 'Happiness Index'},
            'plot_bgcolor': '#f0f0f0',
            'paper_bgcolor': '#f8f8f8',
        }
    }
    return fig

# Define callback to display content on the next page
@app.callback(
    Output('content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/happiness-info':
        # Define the layout for the next page
        return html.Div([
            html.H2("Additional Information"),
            html.P("This page contains additional information about happiness."),
            # Add more content as needed
        ])
    else:
        return ''  # Return empty content if the URL doesn't match any page

# Serve the happiness_info.html file when /happiness-info route is accessed
@server.route("/happiness-info")
def serve_happiness_info():
    return send_file('happiness_index.html')

@app.server.route('/happiness-info')
def render_happiness_info():
    return flask.redirect('/happiness-index.html')

# Define callback to predict happiness score based on user input
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [
        Input('gdp-input', 'value'),
        Input('family-input', 'value'),
        Input('health-input', 'value'),
        Input('freedom-input', 'value'),
        Input('generosity-input', 'value'),
        Input('govtrust-input', 'value'),
        Input('dystopia-input', 'value'),
    ]
)
def predict_happiness_score(n_clicks, gdp, family, health, freedom, generosity, govtrust, dystopia):
    if n_clicks > 0:
        input_data = [[gdp, family, health, freedom, generosity, govtrust, dystopia]]
        predicted_happiness_score = model.predict(input_data)
        return f'Predicted Happiness Score: {predicted_happiness_score[0]:.2f}'


# Define callback to update world map based on happiness index
@app.callback(
    Output('world-happiness-map', 'figure'),
    [Input('happiness-line-chart', 'figure')]
)  
def update_world_map(happiness_line_chart):
    fig = px.choropleth(df1, 
                        locations='Country',
                        locationmode='country names',
                        color='happiness_score',
                        hover_name='Country',
                        title='World Happiness Map',
                        color_continuous_scale=px.colors.sequential.RdBu)
    fig.update_layout(plot_bgcolor='#f0f0f0', paper_bgcolor='#f8f8f8')
    return fig


@app.callback(
    Output('happiness-details', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/happiness-info':
        selected_country = 'India'  # Default value
        selected_year = 2020  # Default value
        happiness_index = 5.5  # Default value
        return dcc.Location(
            pathname='/happiness-info',
            id='happiness-info-url',
            refresh=True,
            search=f'?country={selected_country}&year={selected_year}&happiness_index={happiness_index}'
        )
    else:
        return ''



# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)

