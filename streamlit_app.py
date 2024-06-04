from datetime import datetime, timedelta
import random
from collections import Counter
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

## Hour and Minute Probability Function
def hour_minute_prob(h, m):
    """
    Calculates the probability of visits for a given hour and minute.
    The peak hours are set to 0900 and 1700.
    """
    # Define the probability distribution for peak hours
    hour_prob = np.exp(-(h - 9)**2/(2*4)) + 1.2*np.exp(-(h - 17)**2/(2*4)) + 0.05
    minute_prob = 1 / 60  # Assuming uniform distribution for minutes
    return hour_prob * minute_prob

## Calculate Cumulative Probability Distribution (CPD)
def calc_cpd(prob, domain):
    """
    Calculates the cumulative probability distribution (CPD) for a given probability function and domain.
    """
    norm_prob = []
    for a in range(domain):
        h, m = divmod(a, 60)  # Get hour and minute from the index
        norm_prob.append(prob(h, m))
    norm_prob = np.array(norm_prob) / np.sum(norm_prob)
    cpd = np.cumsum(norm_prob)
    return cpd

## Sample from CPD
def sample_cpd(cpd, n_samples):
    """
    Samples from the cumulative probability distribution (CPD) to generate a specified number of samples.
    """
    samples = np.random.uniform(size=n_samples)
    return np.digitize(samples, cpd)

# Initializes the car distribution at the start
def distribute_cars(total_cars, total_stations, max_station_capacity):
    # Initialize car IDs
    car_ids = list(range(1, total_cars + 1))
    
    # Initialize the car_locations dictionary
    car_locations = {}
    
    # Distribute cars to stations
    for i, car_id in enumerate(car_ids):
        station_id = (i % total_stations) + 1  # Adjusted to start from 1 instead of 0
        
        # Check and assign cars to the station if below capacity
        if sum(1 for v in car_locations.values() if v['station_id'] == station_id) < max_station_capacity:
            car_locations[car_id] = {'station_id': station_id, 'charge_level': random.randint(20, 100)}
        else:
            # Find the next station with available capacity
            for j in range(1, total_stations + 1):  # Starts from 1
                if sum(1 for v in car_locations.values() if v['station_id'] == j) < max_station_capacity:
                    car_locations[car_id] = {'station_id': j, 'charge_level': random.randint(20, 100)}
                    break
    
    return car_locations

def generate_ev_stations(num_stations, max_distance=40, cluster_radius=5):
    positions = np.random.rand(num_stations, 2) * max_distance
    clustering = DBSCAN(eps=cluster_radius, min_samples=1, metric='euclidean')
    labels = clustering.fit_predict(positions)
    return positions, labels

def select_station(labels, selection_bias='linear', exclude_cluster=None):
    unique_clusters, counts = np.unique(labels, return_counts=True)
    
    if exclude_cluster is not None:
        mask = unique_clusters != exclude_cluster
        unique_clusters = unique_clusters[mask]
        counts = counts[mask]

    if selection_bias == 'linear':
        cluster_probabilities = np.linspace(1, 10, num=len(unique_clusters), dtype=float)
    elif selection_bias == 'size':
        cluster_probabilities = counts.astype(float)
    
    cluster_probabilities /= cluster_probabilities.sum()  # Normalize to form a valid probability distribution

    # Select a cluster based on the defined probabilities
    selected_cluster = np.random.choice(unique_clusters, p=cluster_probabilities)
    
    # Select a station within the chosen cluster randomly
    stations_in_cluster = np.where(labels == selected_cluster)[0] + 1  # +1 for 1-based index
    selected_station_id = np.random.choice(stations_in_cluster)
    
    return selected_station_id, selected_cluster

def select_stations_and_calculate_distance(labels, station_map):
    # Select start station
    start_station_id, start_cluster = select_station(labels, selection_bias='size')
    
    # Select end station, excluding the start station's cluster
    end_station_id, end_cluster = select_station(labels, selection_bias='size', exclude_cluster=start_cluster)
    try:
        # Assuming station_map is indexed appropriately or not indexed on specific columns
        distance = station_map[(station_map['start_station_id'] == start_station_id) & 
                               (station_map['end_station_id'] == end_station_id)]['station_distance'].values[0]
    except IndexError:
        # Handle cases where no distance is found
        distance = 10.0  # or set to some default value, or raise an error

    return start_station_id, end_station_id, distance

def calculate_all_distances(positions):
    # Assuming positions is a list or array where index corresponds to station_id - 1
    num_stations = len(positions)
    data = []
    for start_station_id in range(1, num_stations + 1):
        for end_station_id in range(1, num_stations + 1):
            if start_station_id != end_station_id:
                # Calculate the Euclidean distance
                distance = np.linalg.norm(positions[start_station_id-1] - positions[end_station_id-1])
                data.append({
                    "start_station_id": start_station_id,
                    "end_station_id": end_station_id,
                    "station_distance": distance
                })
    return pd.DataFrame(data, columns=["start_station_id", "end_station_id", "station_distance"])

def generate_trip_data(total_requests, total_cars, total_stations, max_station_capacity, start_date, station_map, labels, stations_upgrade=[]):
    # Constants
    num_customers = 50
    consumption_rate = 0.12  # kWh/km
    total_capacity = 30.0  # kWh
    total_range = 250.0  # km

    trip_history = []
    charge_status = []

    # Initialize Q-Pop tables
    car_q_pop = []
    park_q_pop = []
    
    # Car status dictionary to track the location and charge status
    cars = distribute_cars(total_cars, total_stations, max_station_capacity)
    
    # Station capacity initialization
    station_capacity = {i: max_station_capacity for i in range(1, total_stations + 1)}  # Can be dynamically adjusted later
    if len(stations_upgrade) == 0:
        pass
    else:
        for st in stations_upgrade:
            station_capacity[st] = 6

    # Create Station locations
    station_cluster_mapping = {station_id + 1: cluster for station_id, cluster in enumerate(labels)} # Store station cluster mapping

    # Generate trip request times
    hours_minutes = sample_cpd(calc_cpd(hour_minute_prob, 24 * 60), total_requests)
    start_times = sorted([start_date + timedelta(hours=int(hm // 60), minutes=int(hm % 60)) for hm in hours_minutes])
    for start_time in start_times:
        # Randomly select customer ID
        customer_id = np.random.randint(1, num_customers + 1)

        # Randomly select start and end station
        start_station_id, end_station_id, distance_stations = select_stations_and_calculate_distance(labels, station_map)
        # start_station_id = np.random.randint(1, total_stations + 1)
        # Check that cars are available in start station
        available_car_pickup = [car_id_ for car_id_, details in cars.items() if details['station_id'] == start_station_id]
        if len(available_car_pickup) == 0:
            # Add Q-Pop event for unmet car demand
            car_q_pop.append([customer_id, start_time, start_station_id, "unsuccessful"])
            continue
        else:
            car_q_pop.append([customer_id, start_time, start_station_id, "successful"])
            # Randomly select car from available list
            car_id = np.random.choice(available_car_pickup)

        # Randomly select end station from remaining stations
        # Count the station capacity for at this time
        station_counts = Counter(car['station_id'] for car in cars.values())
        # Find the remaining cap for this end station
        remaining_capacity_st = station_capacity[end_station_id] - station_counts[end_station_id]
        # print(f'remain: {remaining_capacity}, station: {end_station_id}')
        # print(station_counts)
        if remaining_capacity_st > 0:
            # There is available parking hence this trip is successful
            park_q_pop.append([customer_id, start_time, end_station_id, "successful"])
            # Update the car to the new location, dont care abt the charge lvl
            cars[car_id]['station_id'] = end_station_id
        else:
            # No parking available, car rental unsuccessful
            park_q_pop.append([customer_id, start_time, end_station_id, "unsuccessful"])
            continue

        # Simulate trip duration and distance
        trip_distance = random.uniform(distance_stations, 50.0)  # Random distance between stations and max 50 km
        
        # Calculate estimated trip duration based on average speed
        average_speed = 60.0
        estimated_duration_minutes = (trip_distance / average_speed) * 60
        margin_of_error = random.uniform(-10, 10) # Adding a random margin of error up to +/- 10 minutes
        actual_duration_minutes = max(0, estimated_duration_minutes + margin_of_error)  # Ensure duration is not negative
        trip_duration = timedelta(minutes=int(actual_duration_minutes))

        # Start and end time of the trip
        end_time = start_time + trip_duration
        
        # Update charge status for tracking
        entry_charge_level = cars[car_id]['charge_level']
        exit_charge_level = max(0, entry_charge_level - int(trip_distance * consumption_rate))  # Assume consumption rate
        
        # Revenue addition
        # Rate per minute
        rate_per_minute = 0.5
        # Calculate the duration in minutes
        duration_in_minutes = trip_duration.total_seconds() / 60
        # Calculate revenue for the trip
        revenue = duration_in_minutes * rate_per_minute

        # Trip history record
        trip_history.append({
            'start_datetime': start_time,
            'end_datetime': end_time,
            'start_station_id': start_station_id,
            'end_station_id': end_station_id,
            'distance_travelled': trip_distance,
            'station_distance': distance_stations,
            'revenue':revenue,
            'car_id': car_id,
            'customer_id': customer_id  # Random customer ID
        })
        
        # Car charge status record
        charge_status.append({
            'entry_datetime': start_time,
            'entry_charge_level': entry_charge_level,
            'exit_datetime': end_time,
            'exit_charge_level': exit_charge_level,
            'car_id': car_id,
            'station_id': start_station_id,
            'exit_station_id': end_station_id
        })
        
        # Update car status
        # cars[car_id]['station_id'] = end_station_id
        cars[car_id]['charge_level'] = exit_charge_level
    
    # Create DataFrames
    car_q_pop_df = pd.DataFrame(car_q_pop, columns=["customer_id", "event_creation_time", "station_id", "event_status"])
    park_q_pop_df = pd.DataFrame(park_q_pop, columns=["customer_id", "event_creation_time", "station_id", "event_status"])
    trip_data = pd.DataFrame(trip_history).sort_values(['start_datetime'])
    charge_data = pd.DataFrame(charge_status).sort_values(['entry_datetime'])
    return trip_data, charge_data, car_q_pop_df, park_q_pop_df, station_cluster_mapping

def ensure_positions(G, positions, seed=42):
    # Ensure all nodes have positions
    missing_nodes = [node for node in G.nodes() if node not in positions]
    if missing_nodes:
        # Generate positions for missing nodes only
        sub_layout = nx.spring_layout(G.subgraph(missing_nodes), seed=seed)
        positions.update(sub_layout)
    return positions

def create_graph(trip_data):
    G = nx.DiGraph()
    for index, row in trip_data.iterrows():
        G.add_edge(row['start_station_id'], row['end_station_id'], weight=row['station_distance'])
    return G

def compute_positions(G, seed=42):
    return nx.spring_layout(G, seed=seed)

def create_visual(G, positions, trip_data, park_q_data, upgraded_nodes=[], seed=42):
    np.random.seed(seed)
    # Ensure all nodes have positions before plotting
    positions = ensure_positions(G, positions, seed)

    unsuccessful_parking_counts = park_q_data[park_q_data['event_status'] == 'unsuccessful']['station_id'].value_counts().to_dict()
    T = create_graph(trip_data)
    in_degrees = {node: T.in_degree(node) for node in T.nodes()}  # Recalculate in-degrees
    missing_nodes = set(G.nodes()) - set(T.nodes())
    in_degrees.update({node: 0 for node in missing_nodes})

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = positions[edge[0]]
        x1, y1 = positions[edge[1]]
        edge_x.extend([x0, x1, None])  # Ensure to add 'None' to separate segments
        edge_y.extend([y0, y1, None])

    node_x = []
    node_y = []
    node_color = []
    node_size = []
    node_line_width = []
    node_line_color = []

    for node in G.nodes():
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        node_color.append(unsuccessful_parking_counts.get(node, 0))
        node_size.append(10 + in_degrees[node] * 5)  # Node size based on new in-degrees
        node_line_width.append(3 if node in upgraded_nodes else 0.5)
        node_line_color.append('black' if node in upgraded_nodes else '#888')

    # Edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Reds',
            size=node_size,
            color=node_color,
            line=dict(width=node_line_width, color=node_line_color),
            colorbar=dict(title='Unsuccessful Parking Attempts', xanchor='left', titleside='right')
        ),
        text=[f'Station {node}<br>Incoming trips: {in_degrees[node]}<br>Unsuccessful attempts: {unsuccessful_parking_counts.get(node, 0)}'
              for node in G.nodes()]
    )

        # Adding a subcaption below the main title
    annotations = [
        dict(
            xref='paper', yref='paper', x=0.39, y=1.04,
            xanchor='center', yanchor='top',
            text='Node size represents the frequency of customer parking at each station.',
            showarrow=False,
            font=dict(size=12, color="grey")
        )
    ]
    # Figure configuration
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
        title=['Network graph of EV stations' if not upgraded_nodes else f'Upgraded EV Stations {str(upgraded_nodes)}'][0],
        titlefont=dict(size=16),
        annotations=annotations,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=40, l=0, r=0, t=40),  # Adjust top margin if necessary
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    ))

    return fig

# Set the seed for reproducibility
seed = 1
np.random.seed(seed)

# Streamlit app
def main():
    # Load your image
    image_path = '/image/bluesglogo.png'
    logo = Image.open(image_path)

    # Add custom CSS to make the image circular
    st.markdown(
        """
        <style>
        .logo-img {
            border-radius: 50%;  # Makes the image circular
            width: auto;  # Adjust the width to fit your layout or keep it responsive
            height: 80px;  # Fixed height, adjust as necessary
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display the image alongside the title
    col1, col2 = st.columns([1, 8])
    with col1:
        st.image(logo, width=80, caption='BlueSG', classes="logo-img")  # Using custom classes
    with col2:
        st.title('EV Station Upgrade Analysis')

    with st.sidebar:
        st.header('Simulation Parameters')
        # Constants with sliders for interactivity
        TOTAL_CARS = st.slider('Total Cars', min_value=10, max_value=300, value=160, step=10)
        TOTAL_STATIONS = st.slider('Total Stations', min_value=10, max_value=100, value=50, step=5)
        TOTAL_REQUESTS = st.slider('Total Requests', min_value=50, max_value=200, value=100, step=50)
        upgrade_station_count = st.slider('Select number of stations to upgrade', 1, 20, 5)

        # Constants
        TOTAL_CARS = min(TOTAL_CARS, MAX_CHARGING_POINTS * TOTAL_STATIONS)
        MAX_CHARGING_POINTS = 4  # Maximum charging points in the future
        start_date = datetime(2024, 5, 1)

    with st.expander("Simulation Parameters", expanded=False):
        st.markdown("""
        **Time Frame**: 1 day  
        **Total cars**: {total_cars} (`{utilization:.2f}%` of maximum capacity based on cars and charging points.)
        **Total stations**: {total_stations}  
        **Total requests**: {total_requests} (indicates the number of rental requests. This does not mean there are 100 confirmed orders for rentals in the day but rather 100 attempts, where some orders may succeed or fail based on the availability of cars and parking.)
        **Max charging points**: {max_charging_points}  

        ---
        Many other assumptions are made, please refer to appendix
        """.format(
            total_cars=TOTAL_CARS,
            max_charging_points=MAX_CHARGING_POINTS,
            total_stations=TOTAL_STATIONS,
            total_requests=TOTAL_REQUESTS,
            utilization=(TOTAL_CARS / (MAX_CHARGING_POINTS * TOTAL_STATIONS)) * 100
        ))


    # Generate positions and labels for stations
    positions, labels = generate_ev_stations(TOTAL_STATIONS)
    station_map = calculate_all_distances(positions)

    # Simulate data without upgrades
    trip_data, charge_data, car_q_data, park_q_data, station_cluster_mapping = generate_trip_data(
        TOTAL_REQUESTS, TOTAL_CARS, TOTAL_STATIONS, MAX_CHARGING_POINTS, 
        start_date, station_map, labels, [])

    # Show basic data counts
    st.subheader("Simulate upgrade of key stations")
    # Analyze parking data for upgrades
    unsuccessful_parking_counts = park_q_data[park_q_data['event_status'] == 'unsuccessful']['station_id'].value_counts()
    successful_parking_counts = park_q_data[park_q_data['event_status'] == 'successful']['station_id'].value_counts()
    upgrade_scores = (unsuccessful_parking_counts * 1.5 + successful_parking_counts).fillna(0).sort_values(ascending=False)
    upgrade_scores.columns = ['Station ID', 'Score']
    # upgrade_station_count = st.slider('Select number of stations to upgrade', 1, 20, 5)
    stations_to_upgrade = list(upgrade_scores[:upgrade_station_count].index)
    st.markdown("""
These are the stations identified for upgrade, selected based on a combination of:
                - Urgency (Customer tried to choose this place to park but failed = revenue loss)
                - Popularity (Customers frequently choose this place to park)
""")
    st.write(upgrade_scores.head(upgrade_station_count))

    # Simulate data with upgrades
    trip_data_up, charge_data_up, car_q_data_up, park_q_data_up, station_cluster_mapping_up = generate_trip_data(
        TOTAL_REQUESTS, TOTAL_CARS, TOTAL_STATIONS, MAX_CHARGING_POINTS, 
        start_date, station_map, labels, stations_to_upgrade)

    # Visualization (Assuming `create_visual` is a function from your network_utils that returns a Plotly figure)
    st.subheader("Mapping out the fleet of stations")
    st.write("Bigger circles means more customers park there")
    st.write("Dark red circles indicate more failed attempts to park")
    G_initial = create_graph(trip_data)
    positions = compute_positions(G_initial, seed)
    fig = create_visual(G_initial, positions, trip_data, park_q_data, upgraded_nodes=[], seed=seed)
    st.plotly_chart(fig)

    fig_new = create_visual(G_initial, positions, trip_data_up, park_q_data_up, upgraded_nodes=stations_to_upgrade, seed=seed)
    st.plotly_chart(fig_new)

    st.subheader("Comparison Overview")
    completed_trips_o = len(trip_data)
    completed_trips_upgrade = len(trip_data_up)
    revenue_o = trip_data['revenue'].sum()
    revenue_upgrade = trip_data_up['revenue'].sum()
    unsuccessful_parking_count_o = len(park_q_data[park_q_data['event_status'] == 'unsuccessful'])
    unsuccessful_parking_count_upgrade = len(park_q_data_up[park_q_data_up['event_status'] == 'unsuccessful'])

    trips_completed_change = round((completed_trips_upgrade / completed_trips_o - 1) * 100, 2)
    revenue_change = round(((revenue_upgrade / revenue_o) - 1) * 100, 2)
    unsuccessful_parking_change = round((unsuccessful_parking_count_upgrade / unsuccessful_parking_count_o - 1) * 100, 2)
    temp_data = {
        'Metric': ['Trips Completed', 'Revenue Generated ($)', 'Unsuccessful Parking'],
        'Original': [completed_trips_o, revenue_o, unsuccessful_parking_count_o],
        'Upgraded': [completed_trips_upgrade, revenue_upgrade, unsuccessful_parking_count_upgrade],
        'Change (%)': [trips_completed_change, revenue_change, unsuccessful_parking_change]
    }
    results_df = pd.DataFrame(temp_data)
    st.write(results_df)


    cost_of_upgrade = 10000 * upgrade_station_count  # Upgrade cost only for the first year
    discount_rate = 0.05  # Annual discount rate
    daily_revenue = revenue_upgrade - revenue_o  # Daily revenue after upgrade, adjust accordingly
    days_per_year = 365  # Days in a year
    annual_revenues = [daily_revenue * days_per_year for _ in range(5)]  # Repeat the revenue for 5 years
    years = ['Year 0', 'Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5']
    npv_df = pd.DataFrame(index=['Revenue', 'Cost', 'Present Value'], columns=years)
    npv_df.loc['Revenue', 'Year 1':] = annual_revenues
    npv_df.loc['Cost', 'Year 0'] = -cost_of_upgrade
    npv_df.loc['Present Value', 'Year 1':] = [
        revenue / ((1 + discount_rate) ** i) for i, revenue in enumerate(annual_revenues, start=1)
    ]
    npv_df.loc['Present Value', 'Year 0'] = npv_df.loc['Cost', 'Year 0']
    # Calculate cumulative present value to determine payback period
    cumulative_pv = npv_df.loc['Present Value'].cumsum()
    payback_year = np.flatnonzero(cumulative_pv >= 0)[0] if np.any(cumulative_pv >= 0) else None
    # Compute exact year fraction for payback
    if payback_year is not None:
        if payback_year == 0:
            payback_period = 0
        else:
            cumulative_before_payback = cumulative_pv.iloc[payback_year - 1]
            revenue_needed = -cumulative_before_payback
            year_revenue = npv_df.loc['Present Value', f'Year {payback_year}']
            fraction_year = revenue_needed / year_revenue
            payback_period = payback_year + fraction_year - 1
        payback_msg = f"{payback_period:.1f} years"
    else:
        payback_msg = "Payback period not achievable within 5 years or infinite"

    st.subheader("Projected NPV (Very rough estimate)")
    st.markdown("""
                Assumes 10K per upgrade <br> 
                Estimated payback period: **{payback_msg}**
""".format(
            payback_msg=payback_msg
        ))
    st.write(npv_df)



    # Add custom CSS to place the footer at the bottom right of the app
    st.markdown("""
        <style>
        .footer {
            position: fixed;
            right: 10px;
            bottom: 10px;
            color: grey;
            text-align: right;
            font-size: small;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown('<div class="footer">Created by Avery Soh</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()