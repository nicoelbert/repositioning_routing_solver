from datetime import datetime
import pickle as pickle
import logging
import sys
import pandas as pd
from classes import classes as cl
import plotly.graph_objects as go
import csv

import networkx as nx

#######################################################################
def get_map(list_depots, list_plants, list_sites):
    import matplotlib.pyplot as plt

    list_depot_lon = []
    list_depot_lat = []

    list_plant_lon = []
    list_plant_lat = []

    list_sites_lon = []
    list_sites_lat = []

    for d in list_depots:
        list_depot_lon.append(d.lon)
        list_depot_lat.append(d.lat)

    for p in list_plants:
        list_plant_lon.append(p.lon)
        list_plant_lat.append(p.lat)

    for s in list_sites:
        list_sites_lon.append(s.lon)
        list_sites_lat.append(s.lat)

    plt.figure(figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')

    plt.scatter(list_sites_lon, list_sites_lat, marker='.', color='grey')
    plt.scatter(list_depot_lon, list_depot_lat, 100, color='blue')
    plt.scatter(list_plant_lon, list_plant_lat, 100, marker='X', color='green')
    plt.legend(['Sites','Depots','Plants'])
    plt.axes()
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.show()


#######################################################################
def get_tour_node_scatter(node_type: str, plot_name: str, color: str, tour: cl.Tour):
    node_x = []
    node_y = []
    for node in tour.routing_sequence:
        if node.node_type == node_type:
            x, y = node.lon, node.lat
            node_x.append(x)
            node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        name=plot_name,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            reversescale=True,
            color=color,
            size=10,
            line_width=2))
    return node_trace

######################################################################
def visualize_tour(tour: cl.Tour):

    if tour.total_tasks == 0:
        print("Tour empty")
        return None

    last_node = tour.routing_sequence[0]

    edge_x = []
    edge_y = []
    for node in tour.routing_sequence:
        x0, y0 = last_node.lon, last_node.lat
        x1, y1 = node.lon, node.lat
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        last_node = node

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        name='Route',
        mode='lines')

    node_trace_pi = get_tour_node_scatter('pickup_job', 'Pickups', 'black', tour)
    node_trace_do = get_tour_node_scatter('dropoff_job', 'Dropoffs', 'red', tour)
    node_trace_p = get_tour_node_scatter('plant', 'Plant/Depot', 'green', tour)
    # node_trace_d = get_tour_node_scatter('depot','Depot','blue',tour)

    node_adjacencies = []
    node_text = []

    fig = go.Figure(data=[edge_trace, node_trace_pi, node_trace_do, node_trace_p],
                    layout=go.Layout(
                        title=None,
                        titlefont_size=16,
                        showlegend=True,
                        legend_font={"size": 20},
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            showarrow=True,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=True, zeroline=True, showticklabels=True, gridcolor='lightgrey',
                                   title_text="Longitude", title_font={"size": 20}),
                        yaxis=dict(showgrid=True, zeroline=True, showticklabels=True, gridcolor='lightgrey',
                                   title_text="Latitude", title_font={"size": 20}),
                        paper_bgcolor='rgba(255,255,255,1)',
                        plot_bgcolor='rgba(255,255,255,1)')
                    )
    fig.show()


######################################################################

def get_time():
    # from https://www.programiz.com/python-programming/datetime/current-time
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return current_time


#######################################################################
def dict_to_csv(dict_to_write: dict, path: str, name: str):
    sys.setrecursionlimit(100000)
    file = path + name + ".csv"
    with open(file, 'w', newline='') as f:
        writer = csv.writer(f)
        for i in dict_to_write:
            writer.writerow(dict_to_write[i])

########################################################################
# adapted from https://stackoverflow.com/questions/39155206/nameerror-global-name-path-is-not-defined


def save_object(obj, filename) -> object:
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as input:
        loaded_object = pickle.load(input)
    return loaded_object


def quick_save(dict_objects: dict, path: str, prefix):
    print_log("Starting quicksave at {}".format(get_time()))
    sys.setrecursionlimit(100000)

    for object in dict_objects:
        # export all data objects
        save_object(object, path + '/quicksave/{}{}.pkl'.format(prefix,object))

    print_log("Done with persitation at {}".format(get_time()))

def persistate(dict_objects: dict, path: str, prefix):
    print_log("Starting persitation at {}".format(get_time()))
    sys.setrecursionlimit(100000)

    # if everything is handed over, create tour_df
    if 'dict_depot' in dict_objects:
        if 'list_days' in dict_objects:
            if 'dict_tours' in dict_objects:

                tour_cols = dict_objects['dict_tours']['Embsen'][17042].get_colums()

                # create df with objects and readable df for solution
                i = 0
                tour_df = pd.DataFrame([tour_cols])
                tour_df.columns = tour_cols
                tour_df_readable = pd.DataFrame([tour_cols])
                tour_df_readable.columns = tour_cols

                for depot in dict_objects['dict_depots']:
                    for day in dict_objects['list_days']:
                        t = dict_objects['dict_depots'][depot][day]
                        tour_df.loc[i] = t.get_all_values()
                        tour_df_readable.loc[i] = t.get_all_value_readable()
                        i += 1

                # export tabular tour data
                tour_df.to_csv(path + '/tour_df.csv')
                tour_df_readable.to_csv(path + '/tour_df_readable.csv')

    for object in dict_objects:
        # export all data objects
        save_object(dict_objects[object], path + '/' + prefix + '{}.pkl'.format(object))

    print_log("Done with persitation at {}".format(get_time()))


def initiate(dict_objects: dict, path: str,prefix: str):
    print_log("Starting initiation at {}".format(get_time()))

    for object in dict_objects:
        # load all data objects
        with open('{}/{}{}.pkl'.format(path,prefix,object), 'rb') as f:
            dict_objects[object] = pickle.load(f)
    return(dict_objects)

    print_log("Done with initiation at {}".format(get_time()))
########################################################################
def print_log(info: str, overwrite_inline=False):
    logging.info(info)
    if overwrite_inline:
        print(info + "- t: {}                                      ".format(get_time()),end='r')
    else:
        print(info+ "- t: {}".format(get_time()))


########################################################################
def day_navigation(day: int, index_delta: int, list_days: list):
    # to be able to navigate in the days
    day_index = list_days.index(day)
    new_index = day_index + index_delta

    return (list_days[new_index])

########################################################################
