# -*- coding: utf-8 -*-
import math
import nashpy as nash
import numbers
import random

class GameTheoryDrivingController():

    def __init__(self,ego,epsilon=0,goal_function=None,other=None,traj_list=None,**kwargs):

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

        self.t = 0

        self.traj_list = traj_list
        self.ego_preference = [1/len(traj_list) for _ in traj_list]
        self.other_preference = [1/len(traj_list) for _ in traj_list]

        #In reality we only need to store the most recent state of each agent, but this setup is here so...
        self.ego_states = []
        self.other_states = []


    def setup(self,ego,other):
        if ego is not None:
            self.ego = ego
            self.initialisation_params['ego'] = ego
        if other is not None:
            self.other = other
            self.initialisation_params['other'] = other


    def selectAction(self,**kwargs):
        self.ego_preference = updatePreference(self.t,self.ego,self.traj_list,self.ego_preference)
        self.other_preference = updatePreference(self.t,self.other,self.traj_list,self.other_preference)
        E_global_cost,NE_global_cost = computeGlobalCostMatrix(self.ego,self.other,self.traj_list)

        E_cost_estimate = list(E_global_cost)
        NE_cost_estimate = list(NE_global_cost)

        for i,row in enumerate(E_cost_estimate):
            new_row = []
            for entry in row:
                if entry != -1: entry*=self.ego_preference[i]
                new_row.append(entry)
            E_cost_estimate[i] = list(new_row)

        for i,row in enumerate(NE_cost_estimate):
            new_row = []
            for entry in row:
                if entry != -1: entry*=self.other_preference[i]
                new_row.append(entry)
            NE_cost_estimate[i] = list(new_row)

        #These policies give us the optimal policies for E to follow under the assumption that
        # NE is rational and the assumption that the preferences have been correctly estimated
        E_policies,NE_policies = computeNashEquilibria(E_cost_estimate,NE_cost_estimate)

        #BASED ON THE POLICIES THAT MAXIMISE NE'S EXPECTED PAYOFF E MUST NOW DETERMINE THEIR
        # BEST ACTION BASED ON THEIR KNOWN, TRUE REWARD FUNCTION

        #SOMEHOW INCORPORATE COMPLIANCE INTO EXPECTED VALUE COMPUTATION BY RELIEVING THE ASSUMPTION
        # OF RATIONALITY ON NE. I.E. WE NO LONGER ASSUME THEY CAN ANTICIPATE THAT A CRASH WILL OCCUR
        # PAYOFF IS AMPLIFIED BY COMPLIANCE FOR THE MAXIMAL ESTIMATED PREFERENCE
        # E MUST THEN DETERMINE THEIR OPTIMAL BEHAVIOUR
        # SOMEHOW THIS IS THEN USED TO DETERMINE WHAT E'S ACTION SHOULD BE

        #IF THE EXPECTED PAYOFF TO NE OF BEHAVING IRRATIONALLY IS HIGHER THAN BEHAVING RATIONALLY,
        # BEHAVE IN COMPLIANCE WITH THE OTHER

        #COMPLIANCE SHOULD BE AN ESTIMATE FOR HOW LIKELY NE IS TO BEHAVE AS WE WANT THEM TO
        #THUS WE SHOULD BE ABLE TO SPECIFY A DESIRED TARGET FOR NE, AND THEN BY CHOOSING ACTIONS
        # CONSTRUCT A GAME IN WHICH THEY ARE INCENTIVISED TO GO TO WHERE WE WANT TO, BY MANIPULATING
        # WHAT THEY WOULD WANT TO DO BASED ON THE COMPLIANCE VALUE

        return action[0],action[1]


    def recordStates(self):
        self.ego_states.append(self.ego.state)
        self.other_states.append(self.other.state)


    def endStep(self):
        """Called (through the vehicle object) at the end of each iteration/timestep of the simulation.
           Allows information about how the iteration actually played out to be gathered"""
        #We use this function to determine the actions each agent actually chose
        self.recordStates()

        #HERE WE WILL UPDATE COMPLIANCE (BETA)


def getState(veh,traj,t):
    state = {}
    #the action available to us is the last action taken
    # whereas the physical state available is the current state (time t)
    # therefore actions must be compared with t-1 traj state
    (state["acceleration"],state["yaw_rate"]) = traj.action(t-1,veh.Lr+veh.Lf)
    state["position"] = traj.position(t)
    state["velocity"] = traj.velocity(t)
    state["heading"] = traj.heading(t)

    return state


def getProb(state,traj,t):
    target_state = getState(traj,t)
    exponent = 0
    for entry in target_state:
        exponent += computeDistance(target_state[entry],state[entry])**2
    exponent = math.sqrt(exponent)
    return math.exp(-exponent)


def updatePreference(t,veh,veh_traj_list,veh_pref,veh_actions):
    """We update the preferences of both agents based on the previous observations using a Bayes Filter"""
    new_pref = []
    num_sum = 0
    states = veh_states[-1]
    for i,traj in enumerate(traj_list):
        new_pref = veh_pref[i]*getProb(state,traj,t)
        num_sum += new_pref
        new_pref.append(new_pref)

    new_pref = [x/num_sum for x in new_pref]
    return new_pref


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


def computeNashEquilibria(E_cost_matrix,NE_cost_matrix):
    E_cost_matrix = np.array(E_cost_matrix)
    NE_cost_matrix = np.array(NE_cost_matrix)
    game = nash.Game(E_cost_matrix,NE_cost_matrix)
    equilibria = game.support_enumeration()

    E_policies,NE_policies = [],[]
    for eq in equilibria:
        E_policies.append(eq[0])
        NE_policies.append(eq[1])

    return E_policies,NE_policies


def zeroFunction(*args):
    return 0


def computeDistance(pt1,pt2):
    if isinstance(pt1,numbers.Number) != isinstance(pt2,numbers.Number):
        print("GTCC Error: Points are not of the same type: PT1: {}\tPT2: {}".format(pt1,pt2))
        exit(-1)
    else:
        if isinstance(pt1,numbers.Number):
            return abs(pt1-pt2)
        else:
            return math.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)
