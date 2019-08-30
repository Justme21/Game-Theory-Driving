import datetime

ROLLOUT_FILENAME = "Traj_Controller_Rollout"

class TrajectoryDrivingController():
    def __init__(self,ego,traj_builder,traj_label,timestep=.1,write=False,**kwargs):
        self.traj_builder = traj_builder

        self.initialisation_params = {"ego":ego,"traj_builder":traj_builder,"traj_label":traj_label,"timestep":timestep}

        self.setup(ego=ego,traj_label=traj_label)
        self.time = 0
        self.dt = timestep

        self.write = write

        if write and "DUMMY" not in ego.label:
            self.rollout_file = initialiseFile(ROLLOUT_FILENAME+"_{}_{}".format(ego.label,traj_label),ego.label,timestep,None)


    def setup(self,ego=None,traj_label=None,**kwargs):
        if ego is not None:
            self.ego = ego
            self.initialisation_params.update({"ego":ego})
        if traj_label is not None:
            self.traj_label = traj_label
            self.trajectory = None
            self.initialisation_params.update({"traj_label":traj_label})

        self.time = 0


    def selectAction(self,*args):
        if self.traj_label is None:
            print("TCC Error: No trajectory initialised")
            exit(-1)

        if self.trajectory is None:
            self.trajectory = self.traj_builder.makeTrajectory(self.traj_label,self.ego.state)

        if self.time>self.trajectory.traj_len_t:
            #print(f"TCC: {self.ego.label} has completed their trajectory")
            #exit(-1)
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
        return TrajectoryDrivingController(**dup_init_params)


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
