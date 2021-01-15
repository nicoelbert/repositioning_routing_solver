import pandas as pd
from pulp import *
import math
from classes import classes as cl


#######################################################################

def get_distance(point1, point2):
    # source : https://www.kite.com/python/answers/how-to-find-the-distance-between-two-lat-long-coordinates-in-python
    R = 6373.0  # radius of the earth

    lat1 = math.radians(point1.lat)
    lon1 = math.radians(point1.lon)
    lat2 = math.radians(point2.lat)
    lon2 = math.radians(point2.lon)

    dlon = lon2 - lon1  # calc diff
    dlat = lat2 - lat1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2  # haversine formula

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance


#######################################################################
def get_proxdepot(site, plant, list_depots):
    distance_min = 10000000000000
    depot_min = ''
    for d in list_depots:
        distance_ges = get_distance(d, site) + get_distance(d, plant)
        if distance_ges < distance_min:
            distance_min = distance_ges
            depot_min = d
    return depot_min




#######################################################################
def routing(tour: cl.Tour):
    # create working nodes to level different object types
    d = cl.Worknode('depot', tour.depot.name, tour.depot)
    p = cl.Worknode('plant', tour.list_plants[0].name, tour.list_plants[0])  # only first plant is selected

    # format jobs to sites
    dropoff_nodes = []
    pickup_nodes = []
    #to resolve nodes later
    dict_node_job = {}

    i = 0
    for j in tour.list_dropoffs:
        wn_j = cl.Worknode('dropoff_job', j.name, j.site)
        dropoff_nodes.append(wn_j)
        dict_node_job[wn_j] = j
        i += 1

    i = 0
    for j in tour.list_pickups:
        wn_j = cl.Worknode('pickup_job'.format(i), j.name, j.site)
        pickup_nodes.append(wn_j)
        dict_node_job[wn_j] = j
        i += 1

    # combined lists
    site_nodes = dropoff_nodes + pickup_nodes
    all_end_nodes = site_nodes + [d] + [p]

    # create and fill distance dict
    distance = {}
    for n in all_end_nodes:
        distance[n] = {}

    for i in all_end_nodes:
        for j in all_end_nodes:
            distance[i][j] = get_distance(i, j)

    # create and fill triangular distances dict with plant
    tri_distance = {}
    for a in site_nodes:
        tri_distance[a] = {}

    for a in site_nodes:
        for b in all_end_nodes:
            tri_distance[a][b] = distance[a][b] + distance[a][p] + distance[p][b]

    ################################### create model ###################################
    m = LpProblem("Routing_simple", LpMinimize)

    ################################### create Variables ###################################
    x = LpVariable.dicts('assignment', (site_nodes, all_end_nodes), cat='Binary')
    ya = LpVariable('site-depot-detector', cat='Binary')
    yb = LpVariable('depot-site-detector', cat='Binary')

    ################################### basic constraints ###################################

    for a in dropoff_nodes:
        m += LpAffineExpression([(x[a][b], 1) for b in pickup_nodes]) + x[a][d] + x[a][
            p] == 1, 'every_dropoff_handled%s' % a

    for b in pickup_nodes:
        m += LpAffineExpression([(x[a][b], 1) for a in dropoff_nodes]) + x[b][d] + x[b][
            p] == 1, 'every_pickup_handled%s' % b

    # a node cannot have an edge with itself
    m += LpAffineExpression([(x[a][a], 1) for a in site_nodes]) == 0, 'no self edges'

    m += LpAffineExpression([(x[a][d], 1) for a in dropoff_nodes]) == 1 - ya, 'site-depot-detection'
    m += LpAffineExpression([(x[b][d], 1) for b in pickup_nodes]) == 1 - yb, 'depot-site-detection'

    ################################### Objective function ###################################
    m += LpAffineExpression(
        [(x[a][b], tri_distance[a][b]) for a in site_nodes for b in all_end_nodes]) + LpAffineExpression(
        [(x[n][p], 2 * distance[n][p]) for n in site_nodes]) + LpAffineExpression(
        [(x[n][d], distance[n][d] + distance[n][p]) for n in site_nodes]) + (ya + yb) * distance[d][p]
    ################################### Evaluate  results ###################################
    m.solve()
    # print(LpStatus[m.status])

    worst_edge_pair_distance = 0
    dict_worst_edge_pair = {}
    dict_worst_edge_pair['dropoff'] = ''
    dict_worst_edge_pair['pickup'] = ''
    worst_edge_pickup_distance = 0
    worst_edge_pickup = ''
    worst_edge_dropoff_distance = 0
    worst_edge_dropoff = ''


    edges = 0

    ## fill routing sequence, edges value and worst edge data - the actual order doesn't matter
    #depot as start_node

    routing_sequence = [d]

    for b in pickup_nodes:
        if x[b][d].varValue > 0:
            routing_sequence += [b,p]
    if ya.varValue == 0:
        routing_sequence.append(p)
    edges += 1



    #node pairs
    for a in dropoff_nodes:
        for b in pickup_nodes:
            #print("from {} to {} is {}".format(a.name, b.name, x[a][b].varValue))
            if x[a][b].varValue > 0:
                #print("from {} to {}".format(a.name, b.name))
                edges += 3
                routing_sequence += [a,b,p] #append triangular sequence

                if distance[a][b] > worst_edge_pair_distance:
                    #resolve job into dict
                    dict_worst_edge_pair['dropoff'] = dict_node_job[a]
                    dict_worst_edge_pair['pickup'] = dict_node_job[b]
                    worst_edge_pair_distance = distance[a][b]


    #single nodes
    for n in site_nodes:
        if x[n][p].varValue > 0:
            routing_sequence += [n, p]

            if distance[n][p] > worst_edge_pickup_distance and n in pickup_nodes:
                worst_edge_pickup = dict_node_job[n]
                worst_edge_pickup_distance = distance[n][p]
            elif distance[n][p] > worst_edge_dropoff_distance and n in dropoff_nodes:
                worst_edge_dropoff = dict_node_job[n]
                worst_edge_dropoff_distance = distance[n][p]


    #depot as end_note
    for a in dropoff_nodes:
        if x[a][d].varValue > 0:
            routing_sequence += [a,d]
    if yb.varValue == 0:
        routing_sequence.append(d)

    distance = m.objective.value()
    ################################### write back to tour ###################################

    #key values
    tour.edges = edges
    tour.routing_sequence = routing_sequence
    tour.distance = distance

    #worst edge data
    tour.worst_edge_pair_distance = worst_edge_pair_distance
    tour.dict_worst_edge_pair = dict_worst_edge_pair
    tour.worst_edge_pickup_distance = worst_edge_pickup_distance
    tour.worst_edge_pickup = worst_edge_pickup
    tour.worst_edge_dropoff_distance = worst_edge_dropoff_distance
    tour.worst_edge_dropoff = worst_edge_dropoff

    #update value
    tour.update_totals
    tour.distance_uptodate = True


########################################################################################################################
#original routing with time windows
########################################################################################################################
def routing_extended(tour: cl.Tour):
    # double plants need to be handled

    # create working nodes to level different object types
    depot_node = cl.Worknode('depot', tour.depot.name, tour.depot)
    plant_node = cl.Worknode('plant', tour.list_plants[0].name, tour.list_plants[0])  # only first plant is selected
    dropoff_nodes = []
    pickup_nodes = []

    for j in tour.list_dropoffs:
        wn_j = cl.Worknode('dropoff_task', j.name, j.site)
        dropoff_nodes.append(wn_j)
    for j in tour.list_pickups:
        wn_j = cl.Worknode('pickup_task', j.name, j.site)
        pickup_nodes.append(wn_j)

    # combined lists
    site_nodes = dropoff_nodes + pickup_nodes
    visitable_nodes = site_nodes.copy()
    visitable_nodes.append(plant_node)
    all_nodes = visitable_nodes.copy()
    all_nodes.append(depot_node)

    for n in pickup_nodes:
        if n in dropoff_nodes:
            print("error with %s" % n)

    # create enough timeslots at least one plantslot after each site
    i_timeslots = len(all_nodes) * 2
    timeslots = []

    for i in range(i_timeslots):
        timeslots.append(i)

    # create distance dict
    distances = {}
    for n in all_nodes:
        distances[n] = {}

    for i in all_nodes:
        for j in all_nodes:
            distances[i][j] = get_distance(i, j)

    ################################### create model ###################################
    m = LpProblem("Routing", LpMinimize)

    ################################### create Variables ###################################
    x = LpVariable.dicts('edge', (all_nodes, all_nodes, timeslots), cat='Binary')
    y = LpVariable.dicts('load_truck', (timeslots), cat='Binary')
    z = LpVariable.dicts('load_silo', (timeslots), cat='Binary')

    # define constraints

    ################################### basic constraints ###################################

    # a node cannot have an edge with itself
    for i in visitable_nodes:
        m += LpAffineExpression([(x[i][i][t], 1) for t in timeslots]) == 0, 'no_selfvisit_%s' % i

    # flow in and out constraints node
    for j in site_nodes:
        m += LpAffineExpression(
            [(x[i][j][t], 1) for t in timeslots for i in all_nodes if i != j]) == 1, 'flow_in_%s' % j

    for i in site_nodes:
        m += LpAffineExpression(
            [(x[j][i][t], 1) for t in timeslots for j in all_nodes if i != j]) == 1, 'flow_out_%s' % i

    # fat every given time slot max one edge can be used
    for t in timeslots:
        m += LpAffineExpression(
            [(x[i][j][t], 1) for i in all_nodes for j in all_nodes]) <= 1, '1_edge_per_timeslot_%s' % t

    # sites and plant need to be left in subsequent timeslot after entering # not for the last timeslot to avoid key errors, should be handled by depot constraints
    max_time_index = len(timeslots) - 1

    for v in visitable_nodes:
        for t in range(0, max_time_index):
            k = t + 1
            m += LpAffineExpression([(x[i][v][t], 1) for i in all_nodes]) == LpAffineExpression(
                [(x[v][j][k], 1) for j in all_nodes]), 'subsequent_ts_for_{}_at_{}_object: {}'.format(v.node_type, t,
                                                                                                      v.name)

    # depot is the first origin
    m += LpAffineExpression([(x[depot_node][j][0], 1) for j in visitable_nodes]) == 1, 'depot first'

    # flow in and out  constraint for depot
    m += LpAffineExpression(
        [(x[i][depot_node][t], 1) for i in visitable_nodes for t in timeslots]) == 1, 'flow_in_depot'
    m += LpAffineExpression(
        [(x[depot_node][j][t], 1) for j in visitable_nodes for t in timeslots]) == 1, 'flow_out_depot'

    ################################### capacity constraints ###################################

    # silo on truck can only be filled if there's a silo on the truck
    for t in timeslots:
        m += z[t] <= y[t], 'silo_truck_link_t_%s' % t

    # silo on truck needs to be filled when visiting dropoff location
    for a in dropoff_nodes:
        for t in timeslots:
            m += LpAffineExpression([(x[i][a][t], 1) for i in all_nodes]) <= z[t], 'full_silo_enter_drop_{}_{}'.format(
                a.name, t)

    # truck is empty after visiting dropoff location
    for a in dropoff_nodes:
        for t in timeslots:
            m += LpAffineExpression([(x[a][j][t], 1) for j in all_nodes]) <= (
                    1 - y[t]), 'empty_truck_leaving_drop_{}_{}'.format(a.name, t)

    # pickup location can only be visted with an empty truck
    for b in pickup_nodes:
        for t in timeslots:
            m += LpAffineExpression([(x[i][b][t], 1) for i in all_nodes]) <= (
                    1 - y[t]), 'empty_truck_enter_pickup_{}_{}'.format(b.name, t)

    # pickup location can only be left with a full truck
    for b in pickup_nodes:
        for t in timeslots:
            m += LpAffineExpression([(x[b][j][t], 1) for j in all_nodes]) <= y[
                t], 'full_truck_leave_pickup_{}_{}'.format(b.name, t)

    # a pickup location can only be left with a empty silo(but still full truck)
    for b in pickup_nodes:
        for t in timeslots:
            m += LpAffineExpression([(x[b][j][t], 1) for j in all_nodes]) <= (
                    1 - z[t]), 'full_truck_empty_silo_leave_pickup_{}_{}'.format(b.name, t)

    # depot ist left with an empty truck
    m += y[0] == 0, 'inital_empty_truck'

    # the depot can only be entered with an empty truck
    for t in timeslots:
        m += LpAffineExpression([(x[i][depot_node][t], 1) for i in all_nodes]) <= (
                1 - y[t]), 'only_return_empty_t_%s' % t

    ################################### Objective function ###################################
    m += LpAffineExpression([(x[i][j][t], distances[i][j]) for i in all_nodes for j in all_nodes for t in timeslots])

    ################################### Evaluate  results ###################################
    m.solve()
    # print(LpStatus[m.status])

    worst_edge_distance = 0
    worst_edge_pickup = ''
    worst_edge_dropoff = ''

    routing_sequence = []

    edges = 0
    for t in timeslots:
        for i in all_nodes:
            for j in all_nodes:
                if i != j:
                    if x[i][j][t].varValue > 0:
                        # print("from {} to {} at {} - truck: {} - silo: {}".format(i.node_type,j.name,t,y[t].varValue,z[t].varValue))
                        routing_sequence.append(i)
                        # get the worst edge between to site nodes to reassign later
                        if distances[i][j] > worst_edge_distance:
                            if i.node_type == 'pickup_task' and j.node_type == 'dropoff_job':
                                worst_edge_distance = distances[i][j]
                                worst_edge_pickup = i
                                worst_edge_dropoff = j
                            elif i.node_type == 'dropoff_task' and j.node_type == 'pickup_job':
                                worst_edge_distance = distances[i][j]
                                worst_edge_pickup = i
                                worst_edge_dropoff = j

                        edges += 1

    distance = m.objective.value()
    ################################### write back to tour ###################################


    tour.edges = edges
    tour.routing_sequence = routing_sequence
    tour.total_distance = distance
    tour.distance_uptodate = True
    tour.worst_edge_pair_distance = worst_edge_distance
    tour.worst_edge_pickup = worst_edge_pickup
    tour.worst_edge_dropoff = worst_edge_dropoff

