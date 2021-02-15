#import game_theory_controller_classes_three_car as gtcc
import reactive_controller_classes as rcc
#from reactive_controller_classes import ReactiveDrivingController
from trajectory_controller_classes import TrajectoryDrivingController
from trajectory_type_definitions import TrajectoryClasses
from trajectory_type_definitions import evaluate,putCarOnTraj
#import cost_function_classes

import datetime
import game_theory_controller_classes as gtcc
#import game_theory_controller_classes_three_car as gtcc3
import math
import matplotlib.pyplot as plt
import nashpy as nash
import numbers
import numpy as np
#import rrt_planner
import sys
sys.path.insert(0,'../driving_simulator')
import linear_controller_classes as lcc
import road_classes
import simulator
import vehicle_classes

from vehicle_classes import sideOfLine

SAFE_DISTANCE = None
MIN_FRONT_BACK_DISTANCE,MIN_SIDE_DISTANCE = None,None

VEHICLE = None

def initialiseSimulator(cars,speed_limit,graphics,init_speeds,lane_width=None,data_generation=False):
    """Takes in a list of cars and a boolean indicating whether to produce graphics.
       Outputs the standard straight road simulator environment with the input cars initialised
       on the map with the first car (presumed ego) ahead of the second"""
    #Construct the simulation environment
    if init_speeds is None:
        car_speeds = [speed_limit for _ in range(len(cars))] #both cars start going at the speed limit
    else:
        car_speeds = init_speeds

    num_junctions = 9
    num_roads = 8
    road_angles = [90,90,90,90,0,0,0,0]
    #road_lengths = [4,156] #Shorter track for generating results
    road_lengths = [6,4,10,20,6,4,10,20] #Shorter track for generating results
    junc_pairs = [(0,1),(1,2),(2,3),(3,4),(5,6),(6,7),(7,3),(3,8)]

    starts = [[(1,2),1],[(6,7),1]] #Follower car is initialised on the first road, leading car on the 3rd (assuring space between them)
    dests = [[(1,2),0],[(1,2),0]] #Simulation ends when either car passes the end of the 

    run_graphics = graphics
    draw_traj = False #trajectories are uninteresting by deafault
    debug = False #don't want debug mode

    runtime = 5.5 #120.0 #max runtime; simulation will terminate if run exceeds this length of time

    #Initialise the simulator object, load vehicles into the simulation, then initialise the action simulation
    sim = simulator.Simulator(run_graphics,draw_traj,runtime,debug,dt=cars[0].timestep)
    sim.loadCars(cars)

    sim.initialiseSimulator(num_junctions,num_roads,road_angles,road_lengths,junc_pairs,\
                                                    car_speeds,starts,dests,lane_width=lane_width)

    return sim


def canGo(cars,do_print=True):
    can_go = False
    for car in cars:
        if car.v<0:
           # print("{} is trying to go backwards: {}".format(car.label,car.v))
            return False
        if car.crashed or not car.on_road:
            if do_print:
                if car.crashed:
                    print("{} has crashed at {}".format(car.label,car.state["position"]))
                else:
                    print("{} has gone off the road at {}".format(car.label,car.state["position"]))
            return False
        if not car.is_complete: can_go = True
    return can_go


def followTrajectory(sim,veh_list,traj_list,speed_limit,show=True,compute_dist=False):
    """Given a list of vehicles and corresponding trajectories generates a graphical simulation of each car following
       their respective trajectories"""
    sim_graphic = sim.graphic #Store original graphics state of simulator
    isim.setGraphic(show) #Set simulator.graphics to match function intention
    move_dict = {}
    record_val = False
    #realised_traj_list = [[] for _ in traj_list]
    if compute_dist:
        dist = None
    for i,traj in enumerate(traj_list):
        #Simulator has been adapted to optionally take in a dictionary specifying trajectories for each vehicle
        veh = veh_list[i]
        move_dict[veh] = [x[1] for x in traj]

    #veh_list[1].y_com = veh_list[0].y_com #only uncomment if you want to compare two similar trajectories
    lead_veh = veh_list[0]

    #Run the simulation. Use singleStep so that you can keep track of the index/position in the trajectory
    if traj_list != []:
        #i = 0
        #while canGo(veh_list):
        for i in range(len(traj_list[0])):
            #for j,traj in enumerate(traj_list):
            #    veh = veh_list[j]
            #    realised_traj_list[j].append((((veh.x_com,veh.y_com),veh.v),move_dict[veh][i][0]))
            sim.singleStep(move_dict=move_dict,index=i)
            #for veh in veh_list:
            #    print("{}: {}".format(veh.label,veh.state))
            #print("\n")
            if compute_dist:
                temp_dist = computeDistance((lead_veh.x_com,lead_veh.y_com),(veh_list[1].x_com,veh_list[1].y_com))
                if i<len(traj_list[0]): # and traj_list[0][i][1] != 0:
                    record_val = True
                if record_val and (dist is None or temp_dist<dist):
                    dist = temp_dist
            #i += 1

        veh_list = veh_list[len(traj_list):]
        while canGo(veh_list) and max([veh.v for veh in veh_list])>.5:
            sim.singleStep(veh_list)
            temp_dist = computeDistance((lead_veh.x_com,lead_veh.y_com),(veh_list[0].x_com,veh_list[0].y_com))
            if record_val and (dist is None or temp_dist<dist):
                dist = temp_dist

        if compute_dist:
            dist -= veh_list[0].length

        #traj_list = realised_traj_list

    else:
        count = 0
        dist = 0
        #min_dist = None
        while canGo(veh_list):
            sim.singleStep()
            temp_dist = computeDistance((veh1.x_com,veh1.y_com),(veh2.x_com,veh2.y_com))
            dist += temp_dist
            count += 1
            #if min_dist is None or temp_dist<min_dist:
            #    min_dist = temp_dist
        dist = (dist/count)-veh_list[0].length
       # min_dist -= veh_list[0].length

    sim.setGraphic(sim_graphic) #Reset simulator graphics to original value when done
    if compute_dist:
        #print("RNR_EXP1: Min Dist between Vehicles: {}".format(min_dist))
        return dist#,traj_list
    else:
        return None#,None


def computeDistance(pt1,pt2):
    if pt2 is None: #Fix for when computing weights for factors that aren't being optimised for
        return 1
    if isinstance(pt1,numbers.Number)!= isinstance(pt2,numbers.Number):
        print("Error: Incompatible Types for Distance Computation\nPt1: {}\tPt1: {}".format(pt1,pt2))
        exit(-1)

    if isinstance(pt1,numbers.Number):
        return abs(pt1-pt2)
    else:
        return math.sqrt(sum([(x-y)**2 for x,y in zip(pt1,pt2)]))

####################################################################
#Application specific functions, required for rrt but change based on implementation

def nodeDistance(state1,state2,nd_weights={},**kwargs):
    #if state1.keys()!=state2.keys():
    #    print("STATIC_OBSTACLE_AVOID_RRT ERROR:")"
    #    print("Something is wrong. Keys don't match\n State1: {}\tState2: {}".format(state1.keys(),state2.keys()))
    #    exit(-1)

    weights = nd_weights
    key_vals = [x for x in state1.keys() if x in state2.keys()]

    dist = {}

    for entry in key_vals:
        if state1[entry] != None and state2[entry] != None:
            if weights == {}: #if no weights specified treat everything equally
                coef = 1
            # if some weights are specified then don't include in cost state elements with no specified weight
            elif entry not in weights.keys() or weights[entry] is None:
                coef = 0
            else:
                coef = weights[entry]
            dist[entry] = coef*computeDistance(state1[entry],state2[entry])
            #dist[entry] = computeDistance(state1[entry],state2[entry])

    #if dist["cost"]!=0: dist["x_com"] = 0

    return dist


def compareDistsALT(new_dist,set_dist,**kwargs):
    """Finds the largest value in 'new_dist' and compares it with the corresponding value in the 'set_dist'"""
    comparison = {}
    max_key,max_val = None,None
    for entry in new_dist:
        comp_val = set_dist[entry]
        if comp_val == 0:
            if new_dist[entry] != 0:
                comparison[entry] = 0
            else:
                comparison[entry] = 1
        else:
            comparison[entry] = max(0,1-new_dist[entry]/comp_val) #As nodes get further from root towards goal this tends to 1

    return comparison


def compareDists(new_dist,set_dist,cd_weights={},**kwargs):
    """Finds the largest value in 'new_dist' and compares it with the corresponding value in the 'set_dist'"""
    weights = cd_weights
    max_key,max_val = None,None
    for entry in new_dist:
        entry_val = new_dist[entry]
        #Reweight values here based on their importance (as specified byt the weights)
        if entry in weights: entry_val *= weights[entry]
        elif weights != {}: entry_val = 0 #In this case there are specified weights, but this entry is not included, therefore we presume it is unimportant
        if max_key is None or max_val*weights[max_key]<entry_val:
            max_key = entry
            max_val = new_dist[entry] #Store the true value instead of the weighted value here as the weights will cancel in the division anyway

    comp_val = set_dist[max_key]
    if comp_val == 0:
        if max_val != 0:
            return 0
        else:
            return 1
    else:
        return max(0,1-max_val/comp_val) #As nodes get further from root towards goal this tends to 1


#State for RRT for non-interactive vehicle
#def defineState(veh):
#    return {"position":(veh.x_com,veh.y_com),"velocity":veh.v,"x_com":veh.x_com,\
#            "heading":veh.heading,"acceleration":veh.accel,"yaw_rate":veh.yaw_rate}

#Requires that the vehicle be interactive
def defineState(veh):
    ic = veh.computeInteractionCost()
    return {"position":(veh.x_com,veh.y_com),"velocity":veh.v,"x_com":veh.x_com,\
            "heading":veh.heading,"acceleration":veh.accel,"yaw_rate":veh.yaw_rate,"cost":veh.computeInteractionCost(),\
            "can_go":int(canGo([veh],do_print=False))}


def specifyState(position,velocity,heading,acceleration,yaw_rate,cost):
    return {"position":position,"velocity":velocity,"x_com":position[0],\
            "heading":heading,"acceleration":acceleration,"yaw_rate":yaw_rate,"cost":cost,"can_go":1}


########################################################################
#Used to define the cost function for the interaction-aware vehicle
def distancePointToLine(pt,line_pt1,line_pt2):
    """Compute the perpendicular distance between a point pt not on the line, and a line defined by two points,
       line_pt1,line_pt2, on the line"""
    #If y-mx-c=0 then perpendicular distance is |-mx1+y1-C|/((-m)^2 + 1^(2))^(.5)
    if line_pt2[0]-line_pt1[0] != 0:
        slope_coef = (line_pt2[1]-line_pt1[1])/(line_pt2[0]-line_pt1[0])
        c = line_pt1[1]-slope_coef*line_pt1[0]
        denom = math.sqrt(slope_coef**2+1)
        d = abs(-slope_coef*pt[0] + pt[1] - c)/denom

    else:
        d = abs(line_pt1[0]-pt[0])

    return d


def obstacleInteractionCost(ego_state,obstacle_state):
    global SAFE_DISTANCE, MIN_FRONT_BACK_DISTANCE, MIN_SIDE_DISTANCE

    min_corner,min_dist = None,None

    for corner in ego_state["four_corners"]:
        dist = computeDistance(ego_state["four_corners"][corner],obstacle_state["position"])
        if dist<SAFE_DISTANCE and (min_corner is None or dist<min_dist):
            min_corner = corner
            min_dist = dist

    if min_corner is None:
        return 0 #No cost as closest point of car is safe distance from obstacle
    else:
        front_back_cost = max(0,min(1,1-(1/MIN_FRONT_BACK_DISTANCE)*min(distancePointToLine(ego_state["four_corners"][min_corner],obstacle_state["four_corners"]["back_left"],obstacle_state["four_corners"]["back_right"]),\
                distancePointToLine(ego_state["four_corners"][min_corner],obstacle_state["four_corners"]["front_left"],obstacle_state["four_corners"]["front_right"]))))
        side_cost = max(0,min(1,1-(1/MIN_SIDE_DISTANCE)*min(distancePointToLine(ego_state["four_corners"][min_corner],obstacle_state["four_corners"]["back_left"],obstacle_state["four_corners"]["front_left"]),\
                distancePointToLine(ego_state["four_corners"][min_corner],obstacle_state["four_corners"]["back_right"],obstacle_state["four_corners"]["front_right"]))))

        if side_cost != 0:
            #If this is true then the ego agent is next to the obstacle
            front_back_overlap = sideOfLine(ego_state["four_corners"][min_corner],obstacle_state["four_corners"]["back_left"],obstacle_state["four_corners"]["back_right"])!=\
                    sideOfLine(ego_state["four_corners"][min_corner],obstacle_state["four_corners"]["front_left"],obstacle_state["four_corners"]["front_right"])
        else:
            front_back_overlap = 0

        if front_back_cost != 0:
            # If this is true then the ego agent is in front of or behind the obstacle
            side_overlap = sideOfLine(ego_state["four_corners"][min_corner],obstacle_state["four_corners"]["back_left"],obstacle_state["four_corners"]["front_left"])!=\
                    sideOfLine(ego_state["four_corners"][min_corner],obstacle_state["four_corners"]["back_right"],obstacle_state["four_corners"]["front_right"])
        else:
            side_overlap = 0

        #print("SOAR: Front Cost: {}\tFRont Overlap: {}\tSide Cost: {}\tSide Overlap: {}".format(front_back_cost,front_back_overlap,side_cost,side_overlap))
        #side_cost should only apply if you are next to the obstacle, and front_back_cost should only apply when behind (or in front of) the obstacle
        return front_back_overlap*side_cost + side_overlap*front_back_cost


def changeLaneLeftRewardFunction(init_veh,final_veh):
    is_complete = False
    for obj in init_veh.on:
        #If car already on left lane no further incentives
        if isinstance(obj,road_classes.Lane) and obj.is_top_up and abs(obj.direction-init_veh.heading)<30:
            is_complete = True
            break

    #if is_complete: print("Vehicle has completed the objective");exit(-1)


    for obj in final_veh.on:
        if isinstance(obj,road_classes.Lane):
            if is_complete:
                #print("Vehicle would end up on: {}".format(obj.label))
                if obj.is_top_up and abs(obj.direction-final_veh.heading)<30: return 1
            if obj.is_top_up and abs(obj.direction-final_veh.heading)<30: return 1

    return 0


def changeLaneRightRewardFunction(init_veh,final_veh):
    is_complete = False
    for obj in init_veh.on:
        if isinstance(obj,road_classes.Lane) and not obj.is_top_up and abs(obj.direction-init_veh.heading)<30:
            is_complete = True
            break

    for obj in final_veh.on:
        if isinstance(obj,road_classes.Lane):
            if is_complete:
                if (not obj.is_top_up) and abs(obj.direction-180-final_veh.heading)<30: return 1
            if (not obj.is_top_up) and abs(obj.direction-180-final_veh.heading)<30: return 1

    return 0


def accelerateRewardFunction(init_veh,final_veh):
    if final_veh.v>init_veh.v: return 1
    else: return 0


def decelerateRewardFunction(init_veh,final_veh):
    if final_veh.v<init_veh.v: return 1
    else: return 0


if __name__ == "__main__":
    print("START HERE {}".format(datetime.datetime.now()))
    start_time = datetime.datetime.now()
    debug = False
    graphics = True
    lane_width = 8
    speed_limit = 13.9
    dt = .1

    #jerk = 1.1 #True value
    jerk = 3
    accel_range = [-9,3.5]
    yaw_rate_range = [-10,10]

    time_len = 3
    traj_classes = TrajectoryClasses(time_len=time_len,lane_width=lane_width,accel_range=accel_range,jerk=jerk)
    traj_classes_2 = TrajectoryClasses(time_len=1.2*time_len,lane_width=lane_width,accel_range=accel_range,jerk=jerk)
    traj_types = traj_classes.traj_types.keys()

    #veh0 = vehicle_classes.Car(controller=None,is_ego=False,debug=debug,label="Lane-Lead-grey",timestep=dt)
    veh1 = vehicle_classes.Car(controller=None,is_ego=False,debug=debug,label="Lane-Follow",timestep=dt)
    #veh1 = vehicle_classes.Car(controller=None,is_ego=False,debug=debug,label="Non-Ego",timestep=dt)
    veh2 = vehicle_classes.Car(controller=None,is_ego=True,debug=debug,label="Ego",timestep=dt)

    sim = initialiseSimulator([veh1,veh2],speed_limit,graphics,[speed_limit/2,speed_limit/2],lane_width,False)
    #sim = initialiseSimulator([veh1],speed_limit,True,[speed_limit/2],lane_width,False)
    #veh2.heading = (veh2.heading+180)%360
    #veh2.y_com  = veh1.y_com
    #veh2.initialisation_params["prev_disp_y"] = veh1.initialisation_params["prev_disp_y"]
    #veh2.initialisation_params["heading"] = veh2.heading
    #veh2.sense()

    #veh1.setMotionParams(posit=(veh1.x_com,veh1.y_com+1.25*veh1.length))
    #veh1.initialisation_params["prev_disp_y"] += 1.25*veh1.length
    #veh1.sense()

    sim.drawSimulation()

    #const_vel_controller = lcc.DrivingController("constant",speed_limit=speed_limit)
    veh1_traj_list = list(traj_types)
    veh2_traj_list = list(traj_types)
    veh1_traj_list.remove("ES")
    veh2_traj_list.remove("ES")
    #veh1_traj_list = [traj_classes.makeTrajectory(x,veh1.state) for x in traj_types]
    #veh2_traj_list = [traj_classes.makeTrajectory(x,veh2.state) for x in traj_types]
    #veh1_controller = gtcc.GameTheoryDrivingController(veh1,veh1_traj_list,traj_classes,other=veh2,other_traj_list=veh2_traj_list)
    #veh1_controller = gtcc.GameTheoryDrivingController(veh1,veh1_traj_list,traj_classes,goal_function=accelerateRewardFunction,other=veh2,other_traj_list=veh2_traj_list,write=False)
    veh1_traj_controller = TrajectoryDrivingController(veh1,traj_classes,"A",veh1.timestep,write=False)
    #veh1_reactive_controller = rcc.ReactiveDrivingController(veh1,traj_classes,"NA",other=veh2,trajectory_list=list(traj_types),timestep=veh1.timestep,write=True)
    veh2_controller = gtcc.GameTheoryDrivingController(veh2,veh2_traj_list,traj_classes,goal_function=changeLaneLeftRewardFunction,other=veh1,other_traj_list=veh1_traj_list,write=True)
    #veh2_controller = gtcc3.GameTheoryDrivingController(veh2,veh2_traj_list,traj_classes,goal_function=changeLaneLeftRewardFunction,reactive_car=veh1,non_ego_traj_list=veh1_traj_list,\
    #        non_reactive_car=veh0,write=True)
    #veh2_traj_controller = TrajectoryDrivingController(veh2,traj_classes,"LCL",veh2.timestep,write=False)

    veh1.addControllers({"game_theory":veh1_traj_controller})
    veh2.addControllers({"game_theory":veh2_controller})

    veh1.setController(tag="game_theory")
    veh2.setController(tag="game_theory")

    sim.runComplete()

    if veh1_traj_controller.write:
        veh1_traj_controller.closeFiles()
    if veh2_controller.write:
        veh2_controller.closeFiles()
    print("Files are closed")
    print("Computation took: {} seconds".format((datetime.datetime.now()-start_time).seconds))
    print("Simulation time is: {}".format(sim.time))

    SAFE_DISTANCE = veh1.length
    MIN_FRONT_BACK_DISTANCE = veh1.length/2 + 1
    MIN_SIDE_DISTANCE = veh1.width/2 + 1
