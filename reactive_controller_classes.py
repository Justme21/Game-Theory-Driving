import datetime
from game_theory_controller_classes_three_car import checkForCrash
from game_theory_controller_classes_three_car import GameTheoryDrivingController

ROLLOUT_FILENAME = "Reactive_Controller_Rollout"

class ReactiveDrivingController():
    def __init__(self,ego,traj_builder,traj_label,trajectory_list=[],other=None,timestep=.1,write=False,**kwargs):
        self.traj_builder = traj_builder
        self.trajectory_list = trajectory_list

        self.initialisation_params = {"ego":ego,"traj_builder":traj_builder,"trajectory_list":trajectory_list,"traj_label":traj_label,"other":other,"timestep":timestep}

        self.setup(ego=ego,traj_label=traj_label,other=other)
        self.time = 0
        self.dt = timestep

        self.has_adjusted = False

        self.write = write

        if write and "DUMMY" not in ego.label:
            self.rollout_file = initialiseFile(ROLLOUT_FILENAME+"_{}_{}".format(ego.label,traj_label),ego.label,timestep,None)


    def setup(self,ego=None,traj_label=None,other=None,**kwargs):
        if ego is not None:
            self.ego = ego
            self.initialisation_params.update({"ego":ego})
        if traj_label is not None:
            self.traj_label = traj_label
            self.trajectory = None
            self.initialisation_params.update({"traj_label":traj_label})

        self.other = other
        self.initialisation_params.update({"other":other})

        self.time = 0


    def selectAction(self,*args):
        if self.traj_label is None:
            print("TCC Error: No trajectory initialised")
            exit(-1)

        if self.trajectory is None:
            #Ego is ahead of the other car, not going to give way
            if self.ego.four_corners["front_right"][1]<self.other.four_corners["back_right"][1]:
                self.trajectory = self.traj_builder.makeTrajectory("NA",self.ego.state)
            else:
                self.trajectory = self.traj_builder.makeTrajectory("D",self.ego.state)

#        if self.trajectory is None:
#            self.trajectory = self.traj_builder.makeTrajectory(self.traj_label,self.ego.state)

#        if (not self.has_adjusted) and self.other is not None and isinstance(self.other.controller,GameTheoryDrivingController):
#            print("Reactive Controller reacting")
#            if self.other.controller.built_ego_traj_list is not None:
#                lcl_index = self.other.controller.ego_traj_list.index("LCL")
#                other_lcl_traj = self.other.controller.built_ego_traj_list[lcl_index]
#                print("LCL is found")
#                r1,_ = checkForCrash(self.other.copy(),other_lcl_traj,0,self.ego.copy(),self.trajectory,self.time)
#                if r1 == -1:
#                    print("LCL will result in crash")
#                    exit(-1)
#                    for traj_label in self.trajectory_list:
#                        possible_traj = self.traj_builder.makeTrajectory(traj_label,self.ego.state)
#                        res,_ = checkForCrash(self.other.copy(),self.other.controller.ego_traj,self.other.controller.t,self.ego.copy(),possible_traj,0)
#                        if res != -1:
#                            print("Collision Avoiding Trajectory Found")
#                            exit(-1)
#                            self.trajectory = possible_traj
#                            self.time = 0
#                            self.has_adjusted = True
#                            break
#

        if self.time>self.trajectory.traj_len_t:
            #print(f"TCC: {self.ego.label} has completed their trajectory")
            #exit(-1)
            if self.trajectory.label != self.traj_label:
                self.trajectory = self.traj_builder.makeTrajectory(self.traj_label,self.ego.state)
                self.time = 0
                action = self.trajectory.action(self.time,self.ego.Lf+self.ego.Lr)
            else:
                action = (0,0)
            #action = self.trajectory.action(self.trajectory.traj_len_t,self.ego.Lf+self.ego.Lr)
            self.time += self.dt
            return action[0],action[1] #End of trajectory, repeat final action (does this make sense? Final action should be (0,0) if designed correctly)
            #return 0,0 #End of trajectory, keep going with no action
        else:
            action = self.trajectory.action(self.time,self.ego.Lf+self.ego.Lr)
            self.time += self.dt
            return action[0], action[1]


    def paramCopy(self,target=None):
        dup_initialisation_params = dict(self.initialisation_params)
        dup_initialisation_params["ego"] = target
        return dup_initialisation_params


    def copy(self,**kwargs):
        dup_init_params = self.paramCopy(**kwargs)
        return ReactiveDrivingController(**dup_init_params)


    def closeFiles(self):
        self.rollout_file.close()


    def endStep(self):
        if self.write:
            writeToFile(self.rollout_file,self.ego.state)
            self.rollout_file.write("\n")


def initialiseFile(filename,ego_name,timestep,data_labels):
    date = datetime.datetime.now()
    file = open(filename+"{}_{}.txt".format(date.hour,date.minute),"a")
    file.write("Ego is: {}\n".format(ego_name))
    file.write("Timestep Size: {}\n".format(timestep))
    if data_labels is not None:
        file.write("{}".format('\t'.join(data_labels)))
    file.write("\n")
    return file


def writeToFile(file,data):
    if isinstance(data,list):
        file.write("{}\n".format('\t'.join([str(x) for x in data])))
    elif isinstance(data,dict):
        file.write("{}\n".format(data))
