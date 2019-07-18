# -*- coding: utf-8 -*-
import math
import random

class GameTheoryDrivingController():

    def __init__(self,ego,epsilon=0,goal_function=None,rules=None,other=None,option_list=None,**kwargs):

        self.initialisation_params = {'ego':ego,'epsilon':epsilon,'other':other,'option_list':option_list}
        self.initialisation_params.update(kwargs)

        if goal_function is not None:
            self.goal_function = goal_function
        else:
            self.goal_function = zeroFunction

        self.rules = rules

        if isinstance(ego,iavc.InteractionConsciousCar):
            self.setup(ego=ego,other=ego.interactive_obstacles[0])
        else:
            self.setup(ego=ego,other=other)

        self.epsilon = epsilon
        self.other_epsilon = other_epsilon

        self.option_list = option_list
        #self.mid_option_list = mid_option_list

        self.cur_option = {self.ego: None, self.other: None}
        self.option_index = {self.ego: None, self.other: None}
        self.other_option_weights = None


    def setup(self,ego,other):
        if ego is not None:
            self.ego = ego
            self.initialisation_params['ego'] = ego
        if other is not None:
            self.other = other
            self.initialisation_params['other'] = other


    def updateOtherParams(self):
        distances = {}
        if self.cur_option[self.other] is not None:
            other_action = (self.other.accel,self.other.yaw_rate)
            for entry in self.option_list:
                if entry == self.cur_option[self.other]:
                    distances[entry] = computeDistance(other_action,entry[self.option_index])
                else:
                    distances[entry] = computeDistance(other_action,entry[0])

        distance_sum = sum([math.exp(-distances[x]) for x in distances])
        self.option_weights = {}
        for entry in distances:
            self.option_weights[entry] = math.exp(-distances[entry])/distance_sum


    def selectAction(self,**kwargs):
        E_global_cost,NE_global_cost = computeGlobalCostMatrix(self.ego,self.other,self.traj_list)
        ego_expected_cost = {}
        other_expected_cost = {}
        for ego_opt in self.option_choices[self.ego]:
            if ego_opt not in ego_expected_cost:
                ego_expected_cost[ego_opt] = 0
            for other_opt in self.option_choices[self.other]:
                if other_opt not in other_expected_cost:
                    other_expected_cost[other_opt] = 0
                ego_expected_cost[ego_opt] += cost_matrix_dict[(ego_opt,other_opt)]
                other_expected_cost[other_opt] += cost_matrix_dict[(ego_opt,other_opt)]

        max_other_option,max_other_ec = None,None
        max_ego_option,max_ego_ec = None,None
        for option in other_expected_cost:
            if max_other_option is None or other_expected_cost[option]>=max_other_ec:
                if max_other_option is None or other_expected_cost[option]>max_other_ec or random.random()<.5:
                    max_other_option = option
                    max_other_ec = other_expected_cost[option]

        for option in ego_expected_cost:
            if max_ego_option is None or ego_expected_cost[option]>=max_ego_ec:
                if max_ego_option is None or ego_expected_cost[option]>max_ego_ec or random.random()<.5:
                    max_ego_option = option
                    max_ego_ec = ego_expected_cost[option]

        best_option_ego,best_option_ego_ec = None,None
        for ego_option in [cost_matrix_dict[(x,max_other_option)] for x in self.option_choices[self.ego]]:
            if best_option_ego is None or cost_matrix_dict[(ego_option,max_other_option)]>=best_option_ego_ec:
                if best_option_ego is None or cost_matrix[(ego_option,max_other_option)]>best_option_ego_ec or \
                        (cost_matrix[(ego_option,max_other_option)]==best_option_ego_ec and (ego_option==self.cur_option[self.ego] \
                        or random.random()<.5)):
                    best_option_ego = ego_option
                    best_option_ego_ec = cost_matrix_dict[(ego_option,max_other_option)]

        if random.random()<self.epsilon:
            chosen_option = best_option_ego
        else:
            chosen_option = max_ego_option

        if chosen_option == self.cur_option[self.ego]:
            self.option_index[self.ego] += 1
            if self.option_oindex[self.ego]>len(chosen_option): self.option_index[self.ego] = 0
        else:
            self.option_index[self.ego] = 0

        action = chosen_option[self.option_index[self.ego]]

        return action[0],action[1]


def computeReward(veh1,traj1,veh2,traj2):
    t = traj1.traj_len_t/2
    deriv = None
    try_num = 100
    i = 0
    while i<try_num and (t>=0 and t<=traj1.traj_len_t) and (deriv is None or abs(deriv)>veh1.timestep):
        E_x = evaluate(t,traj1.x)
        E_x_dot = evaluate(t,traj1.x_dot)
        E_x_dot_dot = evaluate(t,traj1.x_dot_dot)

        E_y = evaluate(t,traj1.y)
        E_y_dot = evaluate(t,traj1.y_dot)
        E_y_dot_dot = evaluate(t,traj1.y_dot_dot)

        NE_x = evaluate(t,traj2.x)
        NE_x_dot = evaluate(t,traj2.x_dot)
        NE_x_dot_dot = evaluate(t,traj2.x_dot_dot)

        NE_y = evaluate(t,traj2.y)
        NE_y_dot = evaluate(t,traj2.y_dot)
        NE_y_dot_dot = evaluate(t,traj2.y_dot_dot)

        deriv = 2*((evaluate(t,traj1.x)-evaluate(t,traj2.x))*(evaluate(t,traj1.x_dot)-\
                evaluate(t,traj2.x_dot))+(evaluate(t,traj1.y)-evaluate(t,traj2.y))*\
                (evaluate(t,traj1.y_dot)-evaluate(t,traj2.y_dot)))
        t -= .01*deriv
        i+=1

    if t<0: t=0
    elif t>traj1.traj_len_t: t=traj1.traj_len_t

    putCarOnTraj(veh1,traj1,t)
    putCarOnTraj(veh2,traj2,t)

    #print("Cars are closest together at time {}\n{}: Pos: {}\tHeading: {}\tCorners: {}\n{}: Pos: {}\tHeading: {}\tCorners: {}".format(\
    #        t,veh1.label,veh1.state["position"],veh1.state["heading"],veh1.four_corners,veh2.label,veh2.state["position"],veh2.state["heading"],veh2.four_corners))

    r1 = 0
    r2 = 0
    if veh1.evaluateCrash(veh2):
        #print("Cars have Crashed")
        r1 = -1
        r2 = -1
    else:
        #print("No Crash found")

        putCarOnTraj(veh1,traj1,traj1.traj_len_t)
        putCarOnTraj(veh2,traj2,traj2.traj_len_t)
        #This is a bit sloppy as it doesn't account for the fact that a car might go off a road and back on during a trajectory. But for now we will allow it        
        #This also might require some refinement as it only works becausd teh copies cannot directly interact
        #print("E is on: {}".format([x.label for x in veh1.on]))
        #for entry in veh1.on:
        #    print("E: (Heading: {}) {}-{}\t{}: {}-{}".format(veh1.heading,veh1.four_corners["front_left"],veh1.four_corners["back_right"],entry.label,entry.four_corners["front_left"],entry.four_corners["back_right"]))
        #print("NE is on: {}".format([x.label for x in veh2.on]))
        #for entry in veh2.on:
        #    print("NE: (Heading: {}) {}-{}\t{}: {}-{}".format(veh2.heading,veh2.four_corners["front_left"],veh2.four_corners["back_right"],entry.label,entry.four_corners["front_left"],entry.four_corners["back_right"]))
        #print("\n")
        if veh1.on == []: r1 = -1 # not on anything => bad outcome
        if veh2.on == []: r2 = -1

    return r1,r2


def computeGlobalCostMatrix(veh1,veh2,traj_list):
    E_cost_list = [[0 for _ in traj_list] for _ in traj_list]
    NE_cost_list = [[0 for _ in traj_list] for _ in traj_list]
    for i,E_traj in enumerate(traj_list):
        for j,NE_traj in enumerate(traj_list):
            #print("E doing: {}\tNE doing: {}".format(E_traj,NE_traj))
            E_cost_list[i][j],NE_cost_list[j][i] = computeReward(veh1.copy(),traj_classes.makeTrajectory(E_traj,veh1.state),veh2.copy(),traj_classes.makeTrajectory(NE_traj,veh2.state))

    return E_cost_list,NE_cost_list


def zeroFunction(*args):
    return 0

def computeDistance(pt1,pt2):
    return math.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)
