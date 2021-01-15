from datetime import datetime
import pickle as pickle
import logging
import sys
import pandas as pd
from classes import classes as cl
import numpy as np
from functions import functions as fc
from functions import routing as rt
import copy
import math
import random
import plotly.graph_objects as go


#################################################################################
class Geometric_Schedule:
    def __init__(self, temp_initial: int, q: float, l: int):
        self.temp_initial = temp_initial
        self.q = q
        self.l = l
        self.dict_development = {0: temp_initial}

    def get_temp(self, step: int):
        temp_new = self.temp_initial * self.q ** math.floor(step / self.l)
        self.dict_development[step] = temp_new
        return temp_new


#################################################################################
class NormalizedExponentialAcceptance:
    def __init__(self, distance_inital: float):
        self.distance_inital = (distance_inital/10)

    def get_acc(self, temperature: float, distance_delta):
        # negative delta -> better solution -> accept
        if distance_delta < 0:
            return True
        else:
            random = np.random.uniform()
            val = math.exp(-distance_delta / (self.distance_inital * temperature))

            bol_curr = random < val
            #fc.print_log("{} < {} : {}".format(round(random,4), round(val,4), bol_curr))
            return bol_curr


class ExponentialAcceptance:
    def __init__(self, distance_inital: float):
        self.distance_inital = distance_inital

    def get_acc(self, temperature: float, distance_delta):
        # negative delta -> better solution -> accept
        if distance_delta < 0:
            return True
        else:
            bol_curr = np.random.uniform() < math.exp(-distance_delta / temperature)
            return bol_curr



#################################################################################
def reassign_job(job_type: str, tour_org: cl.Tour, tour_new: cl.Tour, move_job: cl.Job):
    # retrieve distance values from old tours
    old_distance_tour_org = copy.copy(tour_org.distance)
    old_distance_tour_new = copy.copy(tour_new.distance)
    day_new = tour_new.day

    # move job to new tour
    if job_type == "dropoff":
        tour_org.list_dropoffs.remove(move_job)
        tour_new.list_dropoffs.append(move_job)
    elif job_type == "pickup":
        tour_org.list_pickups.remove(move_job)
        tour_new.list_pickups.append(move_job)
    else:
        raise ValueError('Job type: {} not recognized.'.format(job_type))

    # adjust values in Tour class
    tour_org.distance_uptodate = False
    tour_new.distance_uptodate = False

    # route new
    rt.routing(tour_org)
    rt.routing(tour_new)

    # adjust values in Tour class
    tour_org.update_totals()
    tour_new.update_totals()

    # adjust values in Job class
    if job_type == "dropoff":
        move_job.dropoff_day = day_new
    elif job_type == "pickup":
        move_job.pickup_day = day_new
    else:
        raise ValueError('Job type: {} not recognized.'.format(job_type))

    # retrieve distance values from new tours
    new_distance_tour_org = tour_org.distance
    new_distance_tour_new = tour_new.distance
    distance_delta = new_distance_tour_org + new_distance_tour_new \
                     - old_distance_tour_org - old_distance_tour_new

    return distance_delta


#################################################################################
def reassign_job_min(job_type: str, tour_org: cl.Tour, copy_org: cl.Tour, tour_new: cl.Tour, copy_new: cl.Tour,
                     move_job: cl.Job):
    day_new = tour_new.day

    # overwrite values in Tour class
    tour_org = copy.copy(copy_org)
    tour_new = copy_new

    # adjust values in Job class
    if job_type == "dropoff":
        move_job.dropoff_day = day_new
    elif job_type == "pickup":
        move_job.pickup_day = day_new
    else:
        raise ValueError('Job type: {} not recognized.'.format(job_type))

    # retrieve distance values from new tours

    return None


#################################################################################
def reassign_pickup(tour_org: cl.Tour, tour_new: cl.Tour, move_job: cl.Job):
    # just call right reassign_job function
    return reassign_job('pickup', tour_org, tour_new, move_job)


def reassign_dropoff(tour_org: cl.Tour, tour_new: cl.Tour, move_job: cl.Job):
    # just call right reassign_job function
    return reassign_job('dropoff', tour_org, tour_new, move_job)


#################################################################################
def evaluate_move(job_type: str, tour_org: cl.Tour, tour_new: cl.Tour, move_job: cl.Job):
    # copy tours
    tour_org_copy = tour_org.hardcopy()
    tour_new_copy = tour_new.hardcopy()

    # retrieve distance values from old tours
    old_distance_tour_org = tour_org_copy.distance
    old_distance_tour_new = tour_new_copy.distance
    # day_new = tour_new_copy.day

    # move job to new tour
    if job_type == "dropoff":
        tour_org_copy.list_dropoffs.remove(move_job)
        tour_new_copy.list_dropoffs.append(move_job)
    elif job_type == "pickup":
        tour_org_copy.list_pickups.remove(move_job)
        tour_new_copy.list_pickups.append(move_job)
    else:
        raise ValueError('Job type: {} not recognized.'.format(job_type))

    # adjust values in Tour class
    tour_org_copy.update_totals()
    tour_new_copy.update_totals()
    tour_org_copy.distance_uptodate = False
    tour_new_copy.distance_uptodate = False

    # route new
    rt.routing(tour_org_copy)
    rt.routing(tour_new_copy)

    # retrieve distance values from new tours
    new_distance_tour_org = tour_org_copy.distance
    new_distance_tour_new = tour_new_copy.distance
    distance_delta = new_distance_tour_org + new_distance_tour_new \
                     - old_distance_tour_org - old_distance_tour_new

    return distance_delta


#################################################################################
def evaluate_pickup(tour_org: cl.Tour, tour_new: cl.Tour, move_job: cl.Job):
    # just call right reassign_job function
    return evaluate_move('pickup', tour_org, tour_new, move_job)


def evaluate_dropoff(tour_org: cl.Tour, tour_new: cl.Tour, move_job: cl.Job):
    # just call right reassign_job function
    return evaluate_move('dropoff', tour_org, tour_new, move_job)


#################################################################################

def find_pair_move_worst_random(depot: str, dict_tours_temp: dict, list_days: list):
    pickup_found = False
    dropoff_found = False
    try_count_total = 0

    while pickup_found == False or dropoff_found == False:
        try_count = 0
        pickup_found = False
        dropoff_found = False

        # retrieve random days
        day_org = random.choice(list_days)
        day_new_pickup = random.choice(list_days)
        day_new_dropoff = random.choice(list_days)

        # read random tour and move_job into local variables
        tour_org = dict_tours_temp[depot][day_org]

        # check if worst edge pair is filled
        if tour_org.dict_worst_edge_pair:
            # find a fitting dropoff job
            if tour_org.dict_worst_edge_pair['pickup']:
                move_job_pickup = tour_org.dict_worst_edge_pair['pickup']
                pickup_found = True
            else:
                continue

            while day_new_pickup < move_job_pickup.end:
                day_new_pickup = random.choice(list_days)
                try_count += 1
                # if not possible break after 1000 tries
                if try_count > 500: break

            # check if there is a dropoff job
            if tour_org.dict_worst_edge_pair['dropoff']:
                move_job_dropoff = tour_org.dict_worst_edge_pair['dropoff']
                dropoff_found = True
            else:
                continue

            while day_new_dropoff > move_job_dropoff.start:
                day_new_dropoff = random.choice(list_days)
                try_count += 1
                # if not possible break after 1000 tries
                if try_count > 500: break

        # retrieve new tours
        pickup_tour_new = dict_tours_temp[depot][day_new_pickup]
        dropoff_tour_new = dict_tours_temp[depot][day_new_dropoff]
        # check if list_plants is filled otherwise repeat
        if not pickup_tour_new.list_plants:
            pickup_found = False
        if not dropoff_tour_new.list_plants:
            dropoff_found = False
            # check for try count exit
        if try_count > 500:
            pickup_found = False
            dropoff_found = False

        # restrict number of total loops
        try_count_total += 1
        if try_count_total > 1000:
            fc.print_log("trycount exit")
            return '', '', '','',''

    return tour_org, move_job_pickup, move_job_dropoff, pickup_tour_new, dropoff_tour_new


#################################################################################
def find_pair_move_opposite(depot: str, dict_tours_temp: dict, list_days: list):
    pickup_found = False
    dropoff_found = False
    try_count_total = 0

    while pickup_found == False or dropoff_found == False:
        try_count = 0
        pickup_found = False
        dropoff_found = False

        # restrict number of total loops
        try_count_total += 1
        if try_count_total > 10000:
            fc.print_log("trycount exit")
            return '', '', '','',''

        # retrieve random days
        day_org = random.choice(list_days)
        day_new_pickup = random.choice(list_days)
        day_new_dropoff = random.choice(list_days)

        # read random tour and move_job into local variables
        tour_org = dict_tours_temp[depot][day_org]

        # check if worst edge pair is filled
        if tour_org.dict_worst_edge_pair:
            # find a fitting dropoff job
            if tour_org.dict_worst_edge_pair['pickup']:
                move_job_pickup = tour_org.dict_worst_edge_pair['pickup']
                pickup_found = True
            else:
                pickup_found = False
                continue

            # find the opposite worst edge and move it there
            worst_distance_value = 0
            worst_distance_day = 0
            for day in list_days:
                if day < move_job_pickup.end:
                    if worst_distance_value < dict_tours_temp[depot][day].worst_edge_dropoff_distance and \
                            dict_tours_temp[depot][day].total_tasks < 50:
                        worst_distance_value = dict_tours_temp[depot][day].worst_edge_dropoff_distance
                        worst_distance_day = day
            # if still inital, try again
            if worst_distance_day == 0:
                pickup_found = False
                continue
            else:
                day_new_pickup = worst_distance_day

            # check if there is a dropoff job
            if tour_org.dict_worst_edge_pair['dropoff']:
                move_job_dropoff = tour_org.dict_worst_edge_pair['dropoff']
                dropoff_found = True
            else:
                dropoff_found = False
                continue

            # find the opposite worst edge and move it there
            worst_distance_value = 0
            worst_distance_day = 0
            for day in list_days:
                if day > move_job_dropoff.start:
                    if worst_distance_value < dict_tours_temp[depot][day].worst_edge_dropoff_distance and \
                            dict_tours_temp[depot][day].total_tasks < 50:
                        worst_distance_value = dict_tours_temp[depot][day].worst_edge_dropoff_distance
                        worst_distance_day = day
            # if still inital, try again
            if worst_distance_day == 0:
                dropoff_found = False
                continue
            else:
                day_new_dropoff = worst_distance_day

        # retrieve new tours
        pickup_tour_new = dict_tours_temp[depot][day_new_pickup]
        dropoff_tour_new = dict_tours_temp[depot][day_new_dropoff]
        # check if list_plants is filled otherwise repeat
        if not pickup_tour_new.list_plants:
            pickup_found = False
        if not dropoff_tour_new.list_plants:
            dropoff_found = False
            # check for try count exit
        if try_count > 10000:
            pickup_found = False
            dropoff_found = False

    return tour_org, move_job_pickup, move_job_dropoff, pickup_tour_new, dropoff_tour_new


#################################################################################
def find_single_move_worst_random(move_type: str, depot: str, dict_tours_temp: dict, list_days: list):
    job_found = False
    try_count_total = 0

    while not job_found:
        job_found = False
        try_count = 0

        # restrict number of total loops
        try_count_total += 1
        if try_count_total > 500:
            fc.print_log("trycount exit")
            return '', '', ''

        # retrieve random days
        day_org = random.choice(list_days)
        day_new = random.choice(list_days)

        # read random tour and move_job into local variables
        tour_org = dict_tours_temp[depot][day_org]

        if move_type == 'pickup':
            if tour_org.worst_edge_pickup:
                move_job = tour_org.worst_edge_pickup
                job_found = True
            else:
                job_found = False
                continue

            while day_new < move_job.end:
                day_new = random.choice(list_days)
                try_count += 1
                # if not possible break after 500 tries
                if try_count > 500: break


        elif move_type == 'dropoff':
            if tour_org.worst_edge_dropoff:
                move_job = tour_org.worst_edge_dropoff
                job_found = True
            else:
                job_found = False
                continue

            while day_new < move_job.end:
                day_new = random.choice(list_days)
                try_count += 1
                # if not possible break after 500 tries
                if try_count > 500: break

        # retrieve new tour
        tour_new = dict_tours_temp[depot][day_new]
        # check if list_plants is filled otherwise repeat
        if not tour_new.list_plants:
            job_found = False
        # check for try count exit
        if try_count > 500:
            job_found = False

    return tour_org, move_job, tour_new


#################################################################################
def find_single_move_opposite(move_type: str, depot: str, dict_tours_temp: dict, list_days: list):
    try_count_total = 0
    job_found = False
    while not job_found:
        job_found = False
        try_count = 0
        # retrieve random days
        day_org = random.choice(list_days)
        day_new = random.choice(list_days)

        # restrict number of total loops
        try_count_total += 1
        if try_count_total > 10000:
            fc.print_log("trycount exit for {}".format(move_type))
            return '', '', ''

        # read random tour and move_job into local variables
        tour_org = dict_tours_temp[depot][day_org]

        if move_type == 'pickup':
            if tour_org.worst_edge_pickup:
                move_job = tour_org.worst_edge_pickup
                job_found = True
            else:
                job_found = False
                continue

            # find the opposite worst edge and move it there
            worst_distance_value = 0
            worst_distance_day = 0
            for day in list_days:
                if day < move_job.end:
                    # check if tour fits constraints
                    if worst_distance_value < dict_tours_temp[depot][day].worst_edge_dropoff_distance and \
                            dict_tours_temp[depot][day].total_tasks < 50:
                        worst_distance_value = dict_tours_temp[depot][day].worst_edge_dropoff_distance
                        worst_distance_day = day

            if worst_distance_day != 0:
                day_new = worst_distance_day
            # if still inital, try again
            else:
                job_found = False
                continue

        elif move_type == 'dropoff':
            if tour_org.worst_edge_dropoff:
                move_job = tour_org.worst_edge_dropoff
                job_found = True
            else:
                job_found = False
                continue

            # find the opposite worst edge and move it there
            worst_distance_value = 0
            worst_distance_day = 0
            for day in list_days:
                if day > move_job.start:
                    if worst_distance_value < dict_tours_temp[depot][day].worst_edge_pickup_distance and \
                            dict_tours_temp[depot][day].total_tasks < 50:
                        worst_distance_value = dict_tours_temp[depot][day].worst_edge_pickup_distance
                        worst_distance_day = day

            # if still inital, try again
            if worst_distance_day == 0:
                job_found = False
                continue
            else:
                day_new = worst_distance_day


        # retrieve new tour
        tour_new: cl.Tour = dict_tours_temp[depot][day_new]
        if tour_new:
            # check if list_plants is filled otherwise repeat
            if not tour_new.list_plants:
                job_found = False
            # check for try count exit
            if try_count > 10000:
                job_found = False


    if tour_new:
        return tour_org, move_job, tour_new
    else:
        return '', '', ''


#################################################################################
def find_single_move_random(move_type: str, depot: str, dict_tours_temp: dict, list_days: list):
    job_found = False
    try_count_total = 0

    while job_found == False:
        try_count = 0
        # restrict number of total loops
        try_count_total += 1
        if try_count_total > 500:
            fc.print_log("trycount exit")
            return '', '', ''

        # retrieve random days
        day_org = random.choice(list_days)
        day_new = random.choice(list_days)

        # read random tour and move_job into local variables
        tour_org = dict_tours_temp[depot][day_org]

        if move_type == 'pickup':

            # check if there is a job
            if tour_org.list_pickups:
                move_job = random.choice(tour_org.list_pickups)
                job_found = True
            else:
                continue

            while day_new < move_job.end:
                day_new = random.choice(list_days)
                try_count += 1
                # if not possible break after 1000 tries
                if try_count > 500: break

        elif move_type == 'dropoff':
            # check if there is a job
            if tour_org.list_dropoffs:
                move_job = random.choice(tour_org.list_dropoffs)
                job_found = True
            else:
                continue

            while day_new > move_job.start:
                day_new = random.choice(list_days)
                try_count += 1
                # if not possible break after 1000 tries
                if try_count > 500: break

        # retrieve new tour
        tour_new = dict_tours_temp[depot][day_new]
        # check if list_plants is filled otherwise repeat
        if not tour_new.list_plants:
            job_found = False

    return tour_org, move_job, tour_new


#################################################################################
def find_pickup_move_random(depot: str, dict_tours_temp: dict, list_days: list):
    return find_single_move_random('pickup', depot, dict_tours_temp, list_days)


#################################################################################
def find_dropoff_move_random(depot: str, dict_tours_temp: dict, list_days: list):
    return find_single_move_random('dropoff', depot, dict_tours_temp, list_days)

#################################################################################

def find_single_move_cluster(move_type: str, depot: str, dict_tours_temp: dict, list_days: list):
    try_count_total = 0
    job_found = False
    while not job_found:
        job_found = False
        try_count = 0
        # retrieve random days
        day_org = random.choice(list_days)
        day_new = random.choice(list_days)

        # restrict number of total loops
        try_count_total += 1
        if try_count_total > 10000:
            fc.print_log("trycount exit for {}".format(move_type))
            return '', '', ''

        # read random tour and move_job into local variables
        tour_org = dict_tours_temp[depot][day_org]

        if move_type == 'pickup':
            if tour_org.worst_edge_pickup:
                move_job = tour_org.worst_edge_pickup
                job_found = True
            else:
                job_found = False
                continue

            # find the opposite closest feasible opposite job type
            best_location_distance = 99999999999
            best_location_distance_day = 0

            for day in list_days:
                if day > move_job.end:
                    # check if tour fits constraints and find the closest opposit spot
                    tour_temp = dict_tours_temp[depot][day]
                    for j in tour_temp.list_dropoffs:
                        distance = rt.get_distance(j.site,move_job.site)
                        if distance < best_location_distance and tour_temp.total_tasks < 50:
                            best_location_distance = distance
                            best_location_distance_day = day


            if best_location_distance_day != 0:
                day_new = best_location_distance_day
            # if still inital, try again
            else:
                job_found = False
                continue

        elif move_type == 'dropoff':
            if tour_org.worst_edge_dropoff:
                move_job = tour_org.worst_edge_dropoff
                job_found = True
            else:
                job_found = False
                continue

            # find the opposite closest feasible opposite job type
            best_location_distance = 99999999999
            best_location_distance_day = 0

            for day in list_days:
                if day < move_job.start:
                    # check if tour fits constraints and find the closest opposit spot
                    tour_temp = dict_tours_temp[depot][day]
                    for j in tour_temp.list_pickups:
                        distance = rt.get_distance(j.site, move_job.site)
                        if distance < best_location_distance and tour_temp.total_tasks < 50:
                            best_location_distance = distance
                            best_location_distance_day = day

            if best_location_distance_day != 0:
                day_new = best_location_distance_day
            # if still inital, try again
            else:
                job_found = False
                continue

        # retrieve new tour
        tour_new: cl.Tour = dict_tours_temp[depot][day_new]
        if tour_new:
            # check if list_plants is filled otherwise repeat
            if not tour_new.list_plants:
                job_found = False
            # check for try count exit
            if try_count > 10000:
                job_found = False


    if tour_new:
        return tour_org, move_job, tour_new
    else:
        return '', '', ''


#################################################################################

def find_pair_move_cluster(depot: str, dict_tours_temp: dict, list_days: list):
    pickup_found = False
    dropoff_found = False
    try_count_total = 0

    while pickup_found == False or dropoff_found == False:
        try_count = 0
        pickup_found = False
        dropoff_found = False

        # restrict number of total loops
        try_count_total += 1
        if try_count_total > 10000:
            fc.print_log("trycount exit")
            return '', '', '','',''

        # retrieve random days
        day_org = random.choice(list_days)
        day_new_pickup = random.choice(list_days)
        day_new_dropoff = random.choice(list_days)

        # read random tour and move_job into local variables
        tour_org = dict_tours_temp[depot][day_org]

        # check if worst edge pair is filled
        if tour_org.dict_worst_edge_pair:
            # find a fitting pickup job
            if tour_org.dict_worst_edge_pair['pickup']:
                move_job_pickup = tour_org.dict_worst_edge_pair['pickup']
                pickup_found = True
            else:
                pickup_found = False
                continue

            # find the opposite closest feasible job
            best_location_distance = 99999999999
            best_location_distance_day = 0

            for day in list_days:
                if day > move_job_pickup.end:
                    # check if tour fits constraints and find the closest opposit spot
                    tour_temp = dict_tours_temp[depot][day]
                    for j in tour_temp.list_dropoffs:
                        distance = rt.get_distance(j.site, move_job_pickup.site)
                        if distance < best_location_distance and tour_temp.total_tasks < 50:
                            best_location_distance = distance
                            best_location_distance_day = day

            if best_location_distance_day != 0:
                day_new_pickup = best_location_distance_day
            # if still inital, try again
            else:
                dropoff_found = False
                continue

            # check if there is a dropoff job
            if tour_org.dict_worst_edge_pair['dropoff']:
                move_job_dropoff = tour_org.dict_worst_edge_pair['dropoff']
                dropoff_found = True
            else:
                dropoff_found = False
                continue

            # find the opposite closest feasible opposite job type
            best_location_distance = 99999999999
            best_location_distance_day = 0

            for day in list_days:
                if day < move_job_dropoff.start:
                    # check if tour fits constraints and find the closest opposit spot
                    tour_temp = dict_tours_temp[depot][day]
                    for j in tour_temp.list_pickups:
                        distance = rt.get_distance(j.site, move_job_dropoff.site)
                        if distance < best_location_distance and tour_temp.total_tasks < 50:
                            best_location_distance = distance
                            best_location_distance_day = day

            if best_location_distance_day != 0:
                day_new_dropoff = best_location_distance_day
            # if still inital, try again
            else:
                dropoff_found = False
                continue

        # retrieve new tours
        pickup_tour_new = dict_tours_temp[depot][day_new_pickup]
        dropoff_tour_new = dict_tours_temp[depot][day_new_dropoff]
        # check if list_plants is filled otherwise repeat
        if not pickup_tour_new.list_plants:
            pickup_found = False
        if not dropoff_tour_new.list_plants:
            dropoff_found = False
            # check for try count exit
        if try_count > 10000:
            pickup_found = False
            dropoff_found = False

    return tour_org, move_job_pickup, move_job_dropoff, pickup_tour_new, dropoff_tour_new


#################################################################################

def find_single_move_worst_cluster(move_type: str, depot: str, dict_tours_temp: dict, list_days):
    try_count_total = 0
    job_found = False
    while not job_found:
        job_found = False
        try_count = 0
        # retrieve random days
        day_org = random.choice(list_days)
        day_new = random.choice(list_days)

        # restrict number of total loops
        try_count_total += 1
        if try_count_total > 10000:
            fc.print_log("trycount exit for {}".format(move_type))
            return '', '', ''

        # read random tour and move_job into local variables
        tour_org = dict_tours_temp[depot][day_org]

        if move_type == 'pickup':
            if tour_org.worst_edge_pickup:
                move_job = tour_org.worst_edge_pickup
                job_found = True
            else:
                job_found = False
                continue

            # find the opposite closest feasible opposite job type
            best_location_distance = 99999999999
            best_location_distance_day = 0

            for day in list_days:
                if day > move_job.end:
                    # check if tour fits constraints and find the closest opposit spot
                    tour_temp = dict_tours_temp[depot][day]
                    if tour_temp.worst_edge_dropoff:
                        distance = rt.get_distance(tour_temp.worst_edge_dropoff.site,move_job.site)
                        if distance < best_location_distance and tour_temp.total_tasks < 50:
                            best_location_distance = distance
                            best_location_distance_day = day


            if best_location_distance_day != 0:
                day_new = best_location_distance_day
            # if still inital, try again
            else:
                job_found = False
                continue

        elif move_type == 'dropoff':
            if tour_org.worst_edge_dropoff:
                move_job = tour_org.worst_edge_dropoff
                job_found = True
            else:
                job_found = False
                continue

            # find the opposite closest feasible opposite job type
            best_location_distance = 99999999999
            best_location_distance_day = 0

            for day in list_days:
                if day < move_job.start:
                    # check if tour fits constraints and find the closest opposit spot
                    tour_temp = dict_tours_temp[depot][day]
                    if tour_temp.worst_edge_pickup:
                        distance = rt.get_distance(tour_temp.worst_edge_pickup.site,move_job.site)
                        if distance < best_location_distance and tour_temp.total_tasks < 50:
                            best_location_distance = distance
                            best_location_distance_day = day

            if best_location_distance_day != 0:
                day_new = best_location_distance_day
            # if still inital, try again
            else:
                job_found = False
                continue

        # retrieve new tour
        tour_new: cl.Tour = dict_tours_temp[depot][day_new]
        if tour_new:
            # check if list_plants is filled otherwise repeat
            if not tour_new.list_plants:
                job_found = False
            # check for try count exit
            if try_count > 10000:
                job_found = False


    if tour_new:
        return tour_org, move_job, tour_new
    else:
        return '', '', ''


#################################################################################

def find_pair_move_worst_cluster(depot: str, dict_tours_temp: dict, list_days: list):
    pickup_found = False
    dropoff_found = False
    try_count_total = 0

    while pickup_found == False or dropoff_found == False:
        try_count = 0
        pickup_found = False
        dropoff_found = False

        # restrict number of total loops
        try_count_total += 1
        if try_count_total > 10000:
            fc.print_log("trycount exit")
            return '', '', '','',''

        # retrieve random days
        day_org = random.choice(list_days)
        day_new_pickup = random.choice(list_days)
        day_new_dropoff = random.choice(list_days)

        # read random tour and move_job into local variables
        tour_org = dict_tours_temp[depot][day_org]

        # check if worst edge pair is filled
        if tour_org.dict_worst_edge_pair:
            # find a fitting pickup job
            if tour_org.dict_worst_edge_pair['pickup']:
                move_job_pickup = tour_org.dict_worst_edge_pair['pickup']
                pickup_found = True
            else:
                pickup_found = False
                continue

            # find the opposite closest feasible job
            best_location_distance = 99999999999
            best_location_distance_day = 0

            for day in list_days:
                if day > move_job_pickup.end:
                    # check if tour fits constraints and find the closest opposit spot
                    tour_temp = dict_tours_temp[depot][day]
                    if tour_temp.worst_edge_dropoff:
                        distance = rt.get_distance(tour_temp.worst_edge_dropoff.site,move_job_pickup.site)
                        if distance < best_location_distance and tour_temp.total_tasks < 50:
                            best_location_distance = distance
                            best_location_distance_day = day

            if best_location_distance_day != 0:
                day_new_pickup = best_location_distance_day
            # if still inital, try again
            else:
                dropoff_found = False
                continue

            # check if there is a dropoff job
            if tour_org.dict_worst_edge_pair['dropoff']:
                move_job_dropoff = tour_org.dict_worst_edge_pair['dropoff']
                dropoff_found = True
            else:
                dropoff_found = False
                continue

            # find the opposite closest feasible opposite job type
            best_location_distance = 99999999999
            best_location_distance_day = 0

            for day in list_days:
                if day < move_job_dropoff.start:
                    # check if tour fits constraints and find the closest opposit spot
                   tour_temp = dict_tours_temp[depot][day]
                   if tour_temp.worst_edge_pickup:
                       distance = rt.get_distance(tour_temp.worst_edge_pickup.site,move_job_dropoff.site)
                       if distance < best_location_distance and tour_temp.total_tasks < 50:
                           best_location_distance = distance
                           best_location_distance_day = day

            if best_location_distance_day != 0:
                day_new_dropoff = best_location_distance_day
            # if still inital, try again
            else:
                dropoff_found = False
                continue

        # retrieve new tours
        pickup_tour_new = dict_tours_temp[depot][day_new_pickup]
        dropoff_tour_new = dict_tours_temp[depot][day_new_dropoff]
        # check if list_plants is filled otherwise repeat
        if not pickup_tour_new.list_plants:
            pickup_found = False
        if not dropoff_tour_new.list_plants:
            dropoff_found = False
            # check for try count exit
        if try_count > 10000:
            pickup_found = False
            dropoff_found = False

    return tour_org, move_job_pickup, move_job_dropoff, pickup_tour_new, dropoff_tour_new