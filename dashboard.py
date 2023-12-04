import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from statsmodels.tsa.arima.model import ARIMA  # Corrected import
import base64
from io import BytesIO
from plotly import graph_objs as go  # Add this line
from statsmodels.tsa.statespace.sarimax import SARIMAX


# Sample DataFrame
df = pd.read_csv(r'C:\Users\user\Desktop\PROJECT PROGRAMMING\DASH\SLA Details.csv')
df['Acceptance Date'] = pd.to_datetime(df['Acceptance Date'])

# Get unique 'Destination Office' values for dropdown
destination_offices = df['Destination Office'].unique().tolist()

# Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1('Parcel Volume Dashboard'),

    dcc.DatePickerRange(
        id='date-picker-range',
        start_date=df['Acceptance Date'].min(),
        end_date=df['Acceptance Date'].max(),
        display_format='YYYY-MM-DD'
    ),

    dcc.Dropdown(
        id='destination-office-dropdown',
        options=[{'label': office, 'value': office} for office in destination_offices],
        value=destination_offices[0],
        multi=False,
        placeholder='Select Destination Office'
    ),

    dcc.Graph(id='consignment-count-chart'),
    dcc.Graph(id='status-distribution-chart'),
    dcc.Graph(id='forecast-chart')
])

# Callbacks (update charts based on user interactions)

@app.callback(
    Output('consignment-count-chart', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('destination-office-dropdown', 'value')]
)
def update_consignment_count_chart(start_date, end_date, selected_office):
    filtered_df = df[(df['Acceptance Date'] >= start_date) & (df['Acceptance Date'] <= end_date)]
    if selected_office:
        filtered_df = filtered_df[filtered_df['Destination Office'] == selected_office]

    consignment_counts = filtered_df.groupby('Acceptance Date').size().reset_index(name='Consignment Count')

    fig = px.line(consignment_counts, x='Acceptance Date', y='Consignment Count', title='Consignment Count by Acceptance Date')
    return fig


@app.callback(
    Output('forecast-chart', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('destination-office-dropdown', 'value')]
)
def update_forecast_chart(start_date, end_date, selected_office):
    filtered_df = df[(df['Acceptance Date'] >= start_date) & (df['Acceptance Date'] <= end_date)]
    if selected_office:
        filtered_df = filtered_df[filtered_df['Destination Office'] == selected_office]

    consignment_counts = filtered_df.groupby('Acceptance Date').size().reset_index(name='Consignment Count')

    # Handle missing values in 'Acceptance Date'
    consignment_counts = consignment_counts.dropna(subset=['Acceptance Date'])

    # Forecast using SARIMA
    forecast_steps = 30  # Adjust the number of forecast steps as needed
    order = (1, 0, 1)  # Non-seasonal order: (p, d, q)
    seasonal_order = (1, 1, 1, 30)  # Seasonal order: (P, D, Q, s)

    model = SARIMAX(consignment_counts['Consignment Count'], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    forecast_values = model_fit.get_forecast(steps=forecast_steps)
    
    # Extract forecast values and confidence intervals
    forecast_values_mean = forecast_values.predicted_mean
    forecast_values_ci = forecast_values.conf_int()

    # Generate forecast dates
    forecast_index = pd.date_range(start=consignment_counts['Acceptance Date'].max(), periods=forecast_steps + 1, freq='D')[1:]

    # Create a DataFrame for the forecast
    forecast_df = pd.DataFrame({'Acceptance Date': forecast_index, 'Consignment Count': forecast_values_mean})

    # Combine original data and forecast data
    combined_df = pd.concat([consignment_counts, forecast_df])

    # Create the figure
    fig = go.Figure()

    # Plot the historical data
    fig.add_trace(go.Scatter(x=combined_df['Acceptance Date'], y=combined_df['Consignment Count'], mode='lines', name='Actual'))

    # Plot the forecast data with shaded confidence intervals
    fig.add_trace(go.Scatter(x=forecast_df['Acceptance Date'], y=forecast_values_mean,
                             mode='lines', name='Forecast', line=dict(color='blue')))

    # Shaded confidence intervals
    fig.add_trace(go.Scatter(x=forecast_df['Acceptance Date'], y=forecast_values_ci.iloc[:, 0],
                             mode='lines', name='Lower Bound', line=dict(color='orange'), fill='tonexty'))
    
    fig.add_trace(go.Scatter(x=forecast_df['Acceptance Date'], y=forecast_values_ci.iloc[:, 1],
                             mode='lines', name='Upper Bound', line=dict(color='red'), fill='tonexty'))

    # Set layout and labels
    fig.update_layout(title='Consignment Count Forecast', xaxis_title='Acceptance Date', yaxis_title='Consignment Count')

    return fig



@app.callback(
    Output('status-distribution-chart', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('destination-office-dropdown', 'value')]
)
def update_status_distribution_chart(start_date, end_date, selected_office):
    filtered_df = df[(df['Acceptance Date'] >= start_date) & (df['Acceptance Date'] <= end_date)]
    if selected_office:
        filtered_df = filtered_df[filtered_df['Destination Office'] == selected_office]
        filtered_df['Status'] = filtered_df['Status'].astype(str)
    fig = px.pie(filtered_df, names='Status', title='Status Distribution')
    return fig




if __name__ == '__main__':
    app.run_server(debug=True)
