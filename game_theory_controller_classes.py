# -*- coding: utf-8 -*-
from trajectory_type_definitions import evaluate,putCarOnTraj,ReversedTrajectory,TrajectoryClasses
import math
import nashpy as nash
import numbers
import numpy as np
import random

class GameTheoryDrivingController():

    def __init__(self,ego,ego_traj_list,traj_builder,goal_function=None,other=None,other_traj_list=None,**kwargs):

        self.initialisation_params = {'ego':ego,'ego_traj_list':ego_traj_list,'traj_builder':traj_builder,\
                'goal_function':goal_function,'other':other,'other_traj_list':other_traj_list}
        self.initialisation_params.update(kwargs)

        if goal_function is not None:
            self.goal_function = goal_function
        else:
            self.goal_function = zeroFunction

        self.E_global_cost = None
        self.NE_global_cost = None

        self.traj_builder = traj_builder

        self.setup(ego=ego,ego_traj_list=ego_traj_list,other=other,other_traj_list=other_traj_list)

        self.t = 0
        self.ego_traj_index = None
        self.ego_traj = None

        #self.ego_states = []
        #self.other_states = []


    def setup(self,ego=None,other=None,ego_traj_list=None,other_traj_list=None):
        if ego is not None:
            self.ego = ego
            self.ego_traj_list = ego_traj_list
            self.ego_preference = [1/len(ego_traj_list) for _ in ego_traj_list]
            self.initialisation_params.update({'ego': ego, 'ego_traj_list': ego_traj_list})
        if other is not None:
            self.other = other
            self.other_traj_list = other_traj_list
            self.other_preference = [1/len(other_traj_list) for _ in other_traj_list]
            self.initialisation_params.update({'other': other, 'other_traj_list': other_traj_list})


    def selectAction(self,*args):
        #We maintain an estimate of both agents preferences in order to approximate a perspective
        #common to both agents

        if self.ego_traj is None:
            self.built_ego_traj_list = [self.traj_builder.makeTrajectory(x,self.ego.state) for x in self.ego_traj_list]
            self.built_other_traj_list = [self.traj_builder.makeTrajectory(x,self.other.state) for x in self.other_traj_list]

            #Compute the costs each agent experiences based on the assumption that both agents obey the rules of
            # the road and don't want to crash
            # This reward matrix does not include individual preferences of either agents
            self.E_global_cost,self.NE_global_cost = computeGlobalCostMatrix(self.ego,self.built_ego_traj_list,self.other,self.built_other_traj_list)

        self.ego_preference = updatePreference(self.t,self.ego,self.built_ego_traj_list,self.ego_preference)
        self.other_preference = updatePreference(self.t,self.other,self.built_other_traj_list,self.other_preference)

        #First step we will estimate the cost matrix that NE estimates in order to determine what they
        #  are motivated to do.
        E_cost_estimate = list(self.E_global_cost)
        NE_cost_estimate = list(self.NE_global_cost)

        for i,row in enumerate(E_cost_estimate):
            new_row = []
            for entry in row:
                if entry != -1: entry = self.ego_preference[i]
                new_row.append(entry)
            E_cost_estimate[i] = list(new_row)

        for i,row in enumerate(NE_cost_estimate):
            new_row = []
            for entry in row:
                if entry != -1: entry = self.other_preference[i]
                new_row.append(entry)
            NE_cost_estimate[i] = list(new_row)

        #These policies give us the optimal policies for E to follow under the assumption that
        # NE is rational and the assumption that the preferences have been correctly estimated
        E_policies,NE_policies = computeNashEquilibria(E_cost_estimate,NE_cost_estimate)

        print(f"Selecting Action for {self.ego.label}")
        print(f"\n Preferences are: \n\t\t{self.ego_traj_list}\nEgo: {self.ego_preference}\nNon-Ego: {self.other_preference}")
        print("E Cost Estimate:")
        for row in E_cost_estimate:
            print(row)

        print("\nNE Cost Estimate")
        for row in NE_cost_estimate:
            print(row)

        print("NE believes optimal policies are")
        for i,(E_pol,NE_pol) in enumerate(zip(E_policies,NE_policies)):
            print(f"{i}: {E_pol}\t{NE_pol}")


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

        max_ep = None
        NE_best_policies = []
        for a,(E_p,NE_p) in enumerate(zip(E_policies,NE_policies)):
            ep = 0
            for j in range(len(E_p)):
                for i in range(len(NE_p)):
                    ep += E_p[j]*NE_p[i]*NE_cost_estimate[i][j]
            if max_ep is None or ep>=max_ep:
                if max_ep is None or ep>max_ep:
                    max_ep = ep
                    NE_best_policies = []
                NE_best_policies.append(a)

        #INSTEAD THE BEST POLICIES ARE WHAT THE NE_AGENT MIGHT DO, WE MUST COMPUTE THE BEST RESPONSE TO EACH ONE
        # GIVEN THE TRUE E REWARD FUNCTION, AND THEN THAT IS E'S BEHAVIOUR
        if self.ego_traj is  None:
            #if ego traj is None then Ego is choosing a trajectory to follow, and can choose from any trajectory
            ego_traj_choices = self.built_ego_traj_list
        else:
            #ottherwise Ego is already following a trajectory and can euther choose to continue with that
            # trajectory, or reverse it (i.e. cancel the manoeuvre and go back)
            ego_traj_choices = [self.ego_traj,ReversedTrajectory(self.ego_traj,self.t)]

        print("\nEgo trajectory choices are: {}".format([x.label for x in ego_traj_choices]))

        #Here we can calculate the Ego agent's true reward matrix based on the known trajectories available
        # to them and their known reward function
        E_cost_second_estimate = []
        for ego_traj in ego_traj_choices:
            new_row = []
            for other_traj in self.built_other_traj_list:
                val,_ = computeReward(self.ego.copy(),ego_traj,self.other.copy(),other_traj,veh1_reward_function=self.goal_function)
                new_row.append(val)

            E_cost_second_estimate.append(new_row)

        print("\nE's true estimate of the cost")
        for row in E_cost_second_estimate:
            print(row)

        #We compute the expected payoff for each possible trajectory of E against each possible optimal
        # policy for NE
        expected_payoffs = []
        for index in NE_best_policies:
            payoff = []
            NE_p = NE_policies[index]
            for j in range(len(ego_traj_choices)):
                ep = 0
                for i in range(len(NE_p)):
                    ep += NE_p[i]*E_cost_second_estimate[j][i]
                payoff += [ep]
            expected_payoffs.append(list(payoff))

        print("\nE's expected payoff for each strategy")
        for entry in expected_payoffs:
            print(entry)


        #We identify the trajectory for E that would have the highest expected value
        ep_sum = [0 for _ in ego_traj_choices]
        for row in expected_payoffs:
            for i,ep in enumerate(row):
                ep_sum[i] += ep

        max_index,max_ep = None,None
        for i,entry in enumerate(ep_sum):
            if max_index is None or entry>max_ep:
                max_index = i
                max_ep = entry

        chosen_traj = ego_traj_choices[max_index]

        print("\nChoosing {} with expected reward {}".format(chosen_traj.label,max_ep))
        if chosen_traj != self.ego_traj:
            if self.ego_traj is not None:
                print("Ego proposing to change from {} to {}".format(self.ego_traj.label,chosen_traj.label))
            else:
                print("Ego proposing to change to {}".format(chosen_traj.label))
            self.t = 0
            self.ego_traj = chosen_traj


        action = self.ego_traj.action(self.t,self.ego.Lf+self.ego.Lr)
        self.t += self.ego.timestep

        if self.t>self.ego_traj.traj_len_t:
            print("\n{} Reached the end of the trajectory. Resetting Trajectory".format(self.ego.label))
            self.ego_traj = None

        print("\n")
        return action[0],action[1]


    def paramCopy(self,target=None):
        dup_initialisation_params = dict(self.initialisation_params)
        dup_initialisation_params["ego"] = target
        return dup_initialisation_params


    def copy(self,**kwargs):
        dup_init_params = self.paramCopy(**kwargs)
        return GameTheoryDrivingController(**dup_init_params)


    def recordStates(self):
        """Store the current state of the agents"""
        #Not currently used
        self.ego_states.append(self.ego.state)
        self.other_states.append(self.other.state)


    def endStep(self):
        """Called (through the vehicle object) at the end of each iteration/timestep of the simulation.
           Allows information about how the iteration actually played out to be gathered"""
        #We use this function to determine the actions each agent actually chose
        pass
        #self.recordStates()

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


def getProb(veh,state,traj,t):
    target_state = getState(veh,traj,t)
    exponent = 0
    for entry in target_state:
        exponent += computeDistance(target_state[entry],state[entry])**2
    exponent = math.sqrt(exponent)
    return math.exp(-exponent)


def updatePreference(t,veh,veh_traj_list,veh_pref):
    """We update the preferences of both agents based on the previous observations using a Bayes Filter"""
    new_pref = []
    num_sum = 0
    state = veh.state
    for i,traj in enumerate(veh_traj_list):
        new_pref_val = veh_pref[i]*getProb(veh,state,traj,t)
        num_sum += new_pref_val
        new_pref.append(new_pref_val)

    new_pref = [x/num_sum for x in new_pref]
    return new_pref


def computeReward(veh1,traj1,veh2,traj2,veh1_reward_function=None):
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

        if veh1_reward_function is not None and r1!=-1:
            r1 = veh1_reward_function(veh1)

    return r1,r2


def computeGlobalCostMatrix(veh1,veh1_traj_list,veh2,veh2_traj_list):
    E_cost_list = [[0 for _ in veh2_traj_list] for _ in veh1_traj_list]
    NE_cost_list = [[0 for _ in veh1_traj_list] for _ in veh2_traj_list]
    for i,E_traj in enumerate(veh1_traj_list):
        for j,NE_traj in enumerate(veh2_traj_list):
            #print("E doing: {}\tNE doing: {}".format(E_traj,NE_traj))
            E_cost_list[i][j],NE_cost_list[j][i] = computeReward(veh1.copy(),E_traj,veh2.copy(),NE_traj)

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
