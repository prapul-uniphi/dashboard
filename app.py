import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('chargebee_4.csv', low_memory=False)
df1 = df.groupby(['operator', 'query_id'], as_index = False).agg('size')
df1.cross = pd.crosstab(index=df['operator'], columns=df1['size'])
freqcount=df1.cross.sort_values(by = df1.cross.columns.tolist(), ascending = False)
tabscan_df = df[df['operator'] == 'TableScanOperator']
cols = ['query_id',

        'thread_duration_max', 'cost_percent_str_Values', 'parquet_task_cost_percent_Values',
        'parquet_reading_cost_percent_Values', 'read_io_time_percent_Values', 'filtering_cost_percent_Values',
        'open_time_percent_Values',

        'files', 'partitions', 'tasks', 'total_row_groups', 'parallelism', 'skipped_row_groups', 'row_count_in',
        'row_count_out', 'num_chunks', 'task_rowsInCount_0_50_75_90_100', 'RowsInPerThread_max',
        'InputRowsPerThread_0_50_75_90_100', 'read_io_count', 'read_io_bytes', 'seek_io_count'
         ]
tabscan_df = tabscan_df[cols]
tabscan_df = tabscan_df.dropna(axis = 1, how = 'all')
col_ind = np.where(tabscan_df.isnull().sum() > 1000)[0]
# print(col_ind)
tabscan_df = tabscan_df.drop(tabscan_df.columns[col_ind],axis = 1)
tabscan_df = tabscan_df.dropna()
# print(len(tabscan_df))
tabscan_df = tabscan_df.reset_index(drop = True)

anom_files = np.where((tabscan_df['files'] < 1) | (tabscan_df['tasks'] < 1) |
                      (tabscan_df['row_count_in'] < 1))[0].tolist()
# print(len(anom_files))
tabscan_df.loc[anom_files, 'anomaly_type'] = 'files/tasks/rows = 0'
# print(tabscan_df['anomaly_type'].value_counts())
# tabscan_df.iloc[anom_files]
anom_tasksVtrg = np.where(tabscan_df['tasks'] > tabscan_df['total_row_groups'])[0].tolist()
tabscan_df.loc[anom_tasksVtrg, 'anomaly_type'] = 'tasks > total_row_groups'
anom_tasksVpll = np.where(tabscan_df['tasks'] > tabscan_df['parallelism'])[0].tolist()
# print(len(anom_tasksVpll))

tabscan_df['anomaly_type'].value_counts()

anom_tasksVpll = np.where(tabscan_df['tasks'] > tabscan_df['parallelism'])[0].tolist()
# print(len(anom_tasksVpll))
anom_tasksVpll = [i for i in anom_tasksVpll if i not in anom_tasksVtrg]
# print(len(anom_tasksVpll))
tabscan_df.loc[anom_tasksVpll, 'anomaly_type'] = 'tasks > parallelism'
tabscan_df['anomaly_type'].value_counts()

join_df = df[df['operator'] == 'JoinOperator']

cols = ['query_id', 'thread_duration_max',
       'row_count_in', 'Join_build_row_count_in', 'row_count_out', 'RowsInPerThread_max',
        'InputRowsPerThread_0_50_75_90_100', 'num_chunks', 'join_type']
join_df = join_df[cols]

join_df['join_type'] = join_df['join_type'].astype('category')
join_df['join_type'] = join_df['join_type'].cat.codes
join_df = join_df.dropna(axis = 1, how = 'all')
col_ind = np.where(join_df.isnull().sum() > 1000)[0]
join_df = join_df.drop(join_df.columns[col_ind],axis = 1)
join_df = join_df.dropna()
join_df = join_df.reset_index(drop = True)
join_df['anomaly_type'] = None
anom_files = np.where(( (join_df['thread_duration_max'] == 0) & (join_df['row_count_in'] > 0) &
             (join_df['Join_build_row_count_in'] > 0) ))[0].tolist()
# print(len(anom_files))
join_df.loc[anom_files, 'anomaly_type'] = 'duration = 0 & probe,build rows > 0'
# print(join_df['anomaly_type'].value_counts())
anom_files = np.where((join_df['row_count_in'] < join_df['Join_build_row_count_in']))[0].tolist()
# print(len(anom_files))
join_df.loc[anom_files, 'anomaly_type'] = 'Probe rows < Build rows'
# print(join_df['anomaly_type'].value_counts())

from dash import Dash, dcc, html
import dash_table
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output

value_counts_df = pd.DataFrame(df['operator'].value_counts().reset_index())
value_counts_df.columns = ['Operator', 'Count']

# Filter the DataFrame to get data for "Sink Operator" entries
sink_operator_data = df[df['operator'] == 'SinkOperator']
# Create a bar plot for the count of "Hour" occurrences using Plotly Express
fig_hour = px.bar(
    sink_operator_data.groupby('Hour').size().reset_index(name='count'),  # Count occurrences for each Hour
    x='Hour',  # X-axis: Capitalized "Hour"
    y='count',  # Y-axis: Count of occurrences
    title='Hourly Data for Sink Operator',  # Chart title
    labels={'Hour': 'Hour', 'count': 'Count'},  # Label customization
)
fig_hour.update_layout(
    xaxis_title='Hour',
    yaxis_title='Count',
    xaxis=dict(showline=True, showgrid=False, showticklabels=True, linecolor='black', linewidth=2),
    yaxis=dict(showline=True, showgrid=False, showticklabels=True, linecolor='black', linewidth=2),
    plot_bgcolor='#F2F2F2',  # Plot background color
    paper_bgcolor='white',   # Paper background color
    font=dict(family='Arial', size=14, color='black'),
)
# Create a bar plot for the count of "Month" occurrences using Plotly Express
fig_month = px.bar(
    sink_operator_data.groupby('Month').size().reset_index(name='count'),  # Count occurrences for each Month
    x='Month',  # X-axis: Capitalized "Month"
    y='count',  # Y-axis: Count of occurrences
    title='Monthly Data for Sink Operator',  # Chart title
    labels={'Month': 'Month', 'count': 'Count'},  # Label customization
)
fig_month.update_layout(
    xaxis_title='Month',
    yaxis_title='Count',
    xaxis=dict(showline=True, showgrid=False, showticklabels=True, linecolor='black', linewidth=2),
    yaxis=dict(showline=True, showgrid=False, showticklabels=True, linecolor='black', linewidth=2),
    plot_bgcolor='#F2F2F2',  # Plot background color
    paper_bgcolor='white',   # Paper background color
    font=dict(family='Arial', size=14, color='black'),
)

# Create a Dash app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)

# Define the layout for the Operators window
operators_layout = html.Div([
    html.H2('Operators', style={'text-align': 'center'}),
    dcc.RadioItems(
        id='operator-type',
        options=[
            {'label': 'Table Scan', 'value': 'tablescan'},
            {'label': 'Join', 'value': 'join'}
        ],
        value='tablescan',  # Default value
        style={'text-align': 'center', 'margin-bottom': '20px'}
    ),
    dcc.Tabs(id='operator-tabs', value='A', children=[
        dcc.Tab(label='Independent/Input metrics:', value='A'),
        dcc.Tab(label='Dependent/Output metrics:', value='B'),
        dcc.Tab(label='Correlation', value='C'),
        dcc.Tab(label='Anomaly Types - Rule based', value='D'),
        dcc.Tab(label='Anomaly Types - Model based', value='E'),
    ]),
    html.Div(id='operator-content')
])

# Define the layout for the new window with tabs
new_window_layout = html.Div([
    html.H2('Sink Operator Data', style={'text-align': 'center'}),
    dcc.Tabs([
        dcc.Tab(label='Hourly Data', children=[
            dcc.Graph(figure=fig_hour),  # Add the "Hour" bar plot to the "Hourly Data" tab
        ]),
        dcc.Tab(label='Monthly Data', children=[
            dcc.Graph(figure=fig_month),  # Add the "Month" bar plot to the "Monthly Data" tab
        ]),
    ]),
])

# Define the main layout
app.layout = html.Div([
    html.H1('CHARGBEE DATA', style={'text-align': 'center'}),
    dcc.Tabs([
        dcc.Tab(label='Operator Count', children=[
            dash_table.DataTable(
                id='table',
                columns=[{'name': col, 'id': col} for col in value_counts_df.columns],
                data=value_counts_df.to_dict('records'),
                style_table={
                    'textAlign': 'left',
                    'margin': 'auto',
                    'width': '50%',
                    'border': '1px solid #ddd',
                    'borderCollapse': 'collapse',
                },
                style_header={
                    'backgroundColor': '#007BFF',
                    'color': 'white',
                    'fontWeight': 'bold',
                },
                style_cell={
                    'textAlign': 'left',
                    'border': '1px solid #ddd',
                    'padding': '10px',
                    'backgroundColor': '#F2F2F2',
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#F9F9F9',
                    },
                ],
            )
        ]),
        dcc.Tab(label='Operator Frequency', children=[
            dash_table.DataTable(
                id='freqcount-table',
                columns=[{'name': str(col), 'id': str(col)} for col in freqcount.columns],
                data=freqcount.to_dict('records'),
                style_table={
                    'textAlign': 'left',
                    'margin': 'auto',
                    'width': '50%',
                    'border': '1px solid #ddd',
                    'borderCollapse': 'collapse',
                },
                style_header={
                    'backgroundColor': '#007BFF',
                    'color': 'white',
                    'fontWeight': 'bold',
                },
                style_cell={
                    'textAlign': 'left',
                    'border': '1px solid #ddd',
                    'padding': '10px',
                    'backgroundColor': '#F2F2F2',
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#F9F9F9',
                    },
                ],
            )
        ]),
        dcc.Tab(label='Operator Time', children=[new_window_layout]),  # Add the new window as a tab
        dcc.Tab(label='Operators', children=[operators_layout]),  # Add the Operators window as a tab
    ])
])

# Define callback to update operator content based on selected radio item and tab
@app.callback(
    Output('operator-content', 'children'),
    [Input('operator-type', 'value'),
     Input('operator-tabs', 'value')]
)
def update_operator_content(operator_type, tab_selected):
    operator_data = join_df

    if operator_type == 'tablescan':
        # Define your table data for 'tablescan' here
        operator_data = pd.DataFrame(tabscan_df)  # Define your data
    elif operator_type =='joinscan':
        # Define your table data for 'join' here
        operator_data = pd.DataFrame(join_df)  # Define your data

    if tab_selected == 'A':
        if operator_type == 'tablescan':
            operator_content = html.P(
                "‘files’, ‘partitions’, ‘tasks’, ‘total_row_groups’, ‘parallelism’, "
                "‘skipped_row_groups’, ‘row_count_in’, ‘row_count_out’, ‘num_chunks’, "
                "‘task_rowsInCount_0_50_75_90_100’, ‘RowsInPerThread_max’, ‘seek_io_count’, "
                "‘InputRowsPerThread_0_50_75_90_100’, ‘read_io_count’, ‘read_io_bytes’"
            )
        else:
            operator_content = html.P(
                "‘row_count_in’, ‘Join_build_row_count_in’, ‘row_count_out’, ‘RowsInPerThread_max’, "
                "‘InputRowsPerThread_0_50_75_90_100’, ‘num_chunks’, ‘join_type’ "
            )


    elif tab_selected =='B':
        if operator_type == 'tablescan':
            operator_content=html.P(
                "‘thread_duration_max’,‘cost_percent_str_Values’,‘parquet_task_cost_percent_Values’, "
                "‘parquet_reading_cost_percent_Values’,‘read_io_time_percent_Values’, " 
                "‘filtering_cost_percent_Values’,‘open_time_percent_Values’ ")

        else:
            operator_content=html.P(
            "‘thread_duration_max’")

    elif tab_selected == 'C':

        # Assuming you have already computed the operator_data DataFrame (tabscan_df)

        numeric_columns = operator_data.select_dtypes(include=[float, int]).columns
        correlation_matrix = operator_data[numeric_columns].corr()

        # Create a heatmap using Plotly Express
        correlation_fig = px.imshow(correlation_matrix,
                                    x=correlation_matrix.columns,
                                    y=correlation_matrix.columns,
                                    color_continuous_scale='icefire')
        correlation_fig.update_layout(
            title="Correlation Matrix (With Labels and Color Bar)",
            xaxis_title="Features",
            yaxis_title="Features",
            width=800,  # Adjust the width as needed
            height=800  # Adjust the height as needed
        )
        operator_content = dcc.Graph(figure=correlation_fig)



    elif tab_selected == 'D':
        # Define your anomaly_type_table for both 'tablescan' and 'joinscan' here
        anomaly_type_table = dash_table.DataTable(
            id='anomaly-type-table',
           columns=[
                    {'name': col, 'id': col} for col in operator_data['anomaly_type'].value_counts().reset_index().columns
                ],
            data=operator_data['anomaly_type'].value_counts().reset_index().to_dict('records'),
            # Define the rest of the styles and configurations
            style_table={
                'textAlign': 'left',
                'margin': 'auto',
                'width': '50%',
                'border': '1px solid #ddd',
                'borderCollapse': 'collapse',
            },
            style_header={
                'backgroundColor': '#007BFF',
                'color': 'white',
                'fontWeight': 'bold',
            },
            style_cell={
                'textAlign': 'left',
                'border': '1px solid #ddd',
                'padding': '10px',
                'backgroundColor': '#F2F2F2',
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#F9F9F9',
                },
            ],
        )
        operator_content = anomaly_type_table

    else:
        operator_content = html.Div()  # Placeholder for other tabs (E)

    return operator_content

if __name__ == '__main__':
    app.run_server(debug=True, port=8056)
