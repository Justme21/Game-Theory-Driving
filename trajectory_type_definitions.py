import math
import numbers

class Line():
    def __init__(self,*args):
        args = list(args)
        args.reverse()
        self.coefs = [(x,i) for i,x in enumerate(args)]


    def dot(self,coefs=None):
        if coefs is None: coefs = list(self.coefs)
        else: coefs = list(coefs)

        for i in range(len(coefs)):
            if coefs[i][1] == 0:
                coefs[i] = (0,0)
            else:
                coefs[i] = (coefs[i][0]*coefs[i][1],coefs[i][1]-1)
        return coefs


class TrajectoryClasses():
    def __init__(self,time_len=1,lane_width=6,accel_range=[-2,2],jerk=1):
        self.traj_len = time_len
        self.traj_types = {"LCR":{"position":(lane_width,0)},"LCL":{"position":(-lane_width,0)},\
                "A": {"acceleration":1},"D":{"acceleration":-1},"NA":{}}

        self.boundary_constraints = {"accel_range":accel_range,"jerk":jerk}


    def makeTrajectory(self,traj_key,cur_state):
        heading = math.radians(cur_state["heading"]) #this might be the wrong way to incorporate orientation. Revise later maybe
        orient = (math.sin(heading),math.cos(heading)) #Adjustments perpendicular to current heading

        dest_state = dict(cur_state)
        state_change = self.traj_types[traj_key]
        for entry in state_change:
            if isinstance(dest_state[entry],numbers.Number): dest_state[entry] += state_change[entry]
            else:
                if entry is "position": coef = orient
                else: coef = [1 for _ in dest_state[entry]]
                dest_state[entry] = tuple([x + coef[i]*state_change[entry][i] for i,x in enumerate(dest_state[entry])])

        return Trajectory(traj_key,cur_state,dest_state,self.traj_len,**self.boundary_constraints)


class Trajectory():
    def __init__(self,traj_type,init_state,dest_state,time_len,accel_range,jerk):
        if init_state is None or dest_state is None:
            print("Error, default values for states are invalid")
            exit(-1)
        self.traj_len_t = time_len
        self.traj_func = {"LCL":laneChange,"LCR":laneChange,"A":velocityChange,"D":velocityChange,"NA":noAction}
        self.line_x,self.line_y,self.traj_len_t = self.traj_func[traj_type](init_state,dest_state,time_len,accel_range=accel_range,jerk=jerk)
        self.computeDerivatives()


    def computeDerivatives(self):
        #x = evaluate(t,self.line_x.coefs)
        #y = evaluate(t,self.line_y.coefs)
        self.x = self.line_x.coefs
        self.y = self.line_y.coefs
        self.x_dot = self.line_x.dot()
        self.y_dot = self.line_y.dot()
        self.x_dot_dot = self.line_x.dot(self.x_dot)
        self.y_dot_dot = self.line_y.dot(self.y_dot)


    def action(self,t,axle_length):
        x = evaluate(t,self.x)
        y = evaluate(t,self.y)

        x_dot = evaluate(t,self.x_dot)
        y_dot = evaluate(t,self.y_dot)
        x_dot_dot = evaluate(t,self.x_dot_dot)
        y_dot_dot = evaluate(t,self.y_dot_dot)

        denom_a = math.sqrt(x_dot**2 + y_dot**2)
        denom_yaw = denom_a**3

        #print("COEFS: X: {}\tY: {}\tX_DOT: {}\tY_DOT: {}\tX_DOT_DOT: {}\tY_DOT_DOT: {}".format(self.line_x.coefs,self.line_y.coefs,self.x_dot,self.y_dot,self.x_dot_dot,self.y_dot_dot))
        #print("X: {}\tY: {}\tX_DOT: {}\tY_DOT: {}\tX_DOT_DOT: {}\tY_DOT_DOT: {}".format(x,y,x_dot,y_dot,x_dot_dot,y_dot_dot))

        acceleration = ((x_dot*x_dot_dot)+(y_dot*y_dot_dot))/denom_a
        #I think due to the flipping of the y-axis yaw rate needs to be computed with negatives of y-associated values
        # This works but am not sure. Original is commented out below
        #yaw_rate = math.degrees(math.atan(((x_dot*y_dot_dot)-(y_dot*x_dot_dot))*axle_length/denom_yaw))
        yaw_rate = math.degrees(math.atan(((x_dot*-y_dot_dot)-(-y_dot*x_dot_dot))*axle_length/denom_yaw))

        return acceleration,yaw_rate


    def position(self,t):
        return (evaluate(t,self.x),evaluate(t,self.y))


    def velocity(self,t):
        return math.sqrt(evaluate(t,self.x_dot)**2 + evaluate(t,self.y_dot)**2)


    def completePositionList(self,dt=.1):
        t = 0
        position_list = []
        while t<=self.traj_len_t+dt:
            position_list.append(self.position(t))
            t += dt

        return position_list


    def completeVelocityList(self,dt=.1):
        t = 0
        velocity_list = []
        while t<=self.traj_len_t+dt:
            velocity_list.append(self.velocity(t))
            t += dt

        return velocity_list


    def completeActionList(self,axle_length,dt=.1):
        t = 0
        action_list = []
        while t<=self.traj_len_t+dt:
            action_list.append(self.action(t,axle_length))
            t += dt

        return action_list


def laneChange(init_state,dest_state,time_len,**kwargs):
    init_pos = init_state["position"]
    init_vel = init_state["velocity"]
    init_accel = init_state["acceleration"]
    init_heading = init_state["heading"]

    dest_pos = dest_state["position"]
    dest_vel = dest_state["velocity"]
    dest_heading = dest_state["heading"]

    #Translate to global coordinates
    init_vel = (init_vel*math.cos(math.radians(init_heading)),-init_vel*math.sin(math.radians(init_heading)))
    init_accel = (init_accel*math.cos(math.radians(init_heading)),init_accel*math.sin(math.radians(init_heading)))

    C = init_accel[0]
    D = init_vel[0]
    E = init_pos[0]
    A = (36/(time_len**4))*(C*(time_len**2)/6 + 2*D*time_len/3 + E + time_len*init_vel[0]/3 - dest_pos[0])
    B = (2/(time_len**2))*(init_vel[0] -A*(time_len**3)/3 - C*time_len - D)

    A_x = A/12
    B_x = B/6
    C_x = C/2
    D_x = D
    E_x = E

    line_x = Line(A_x,B_x,C_x,D_x,E_x)

    A_y = init_vel[1]
    B_y = init_pos[1]
    line_y = Line(A_y,B_y)

    return line_x,line_y,time_len


def velocityChange(init_state,dest_state,time_len,accel_range,jerk,**kwargs):
    init_pos = init_state["position"]
    init_vel = init_state["velocity"]
    init_heading = init_state["heading"]
    init_accel = init_state["acceleration"]

    if init_accel<dest_state["acceleration"]:
        is_accel = True
    else:
        is_accel = False

    init_vel = (init_vel*math.cos(math.radians(init_heading)),-init_vel*math.sin(math.radians(init_heading)))

    init_accel = (init_accel*math.cos(math.radians(init_heading)),-init_accel*math.sin(math.radians(init_heading)))

    #Boundary Conditions
    E = init_pos[1]
    D = init_vel[1]
    C = init_accel[1]
    #limit = (-6*C)/time_len
    if is_accel:
        B = -jerk
        #if jerk<limit: B = jerk
        #else: B = limit
    else:
        B = jerk
        #if limit>-jerk: B = limit
        #else: B = jerk
    A = -B/time_len

    A_y = A/12
    B_y = B/6
    C_y = C/2
    D_y = D
    E_y = E
    line_y = Line(A_y,B_y,C_y,D_y,E_y)

    A_x = init_vel[0]
    B_x = init_pos[0]
    line_x = Line(A_x,B_x)

    return line_x,line_y,time_len


def noAction(init_state,dest_state,time_len,**kwargs):
    init_pos = init_state["position"]
    init_vel = init_state["velocity"]
    init_heading = init_state["heading"]

    init_vel = (init_vel*math.cos(math.radians(init_heading)),-init_vel*math.sin(math.radians(init_heading)))

    A_x = init_vel[0]
    B_x = init_pos[0]

    line_x = Line(A_x,B_x)

    A_y = init_vel[1]
    B_y = init_pos[1]

    line_y = Line(A_y,B_y)

    return line_x,line_y,time_len


def putCarOnTraj(car,traj,time):
    posit = (evaluate(time,traj.x),evaluate(time,traj.y))
    v_x = evaluate(time,traj.x_dot)
    v_y = evaluate(time,traj.y_dot)
    velocity = math.sqrt(v_x**2 + v_y**2)
    if v_x != 0:
        # - here to account for the fact that y-axis is inverted but angles are not
        heading = math.degrees(math.atan(-v_y/v_x))
        if v_x<0: heading += 180 #tan has domain [-90.90], which constrains output of atan to left half-domain
        heading%=360
    else:
        if v_y>0: heading = 270
        else: heading = 90

    car.setMotionParams(posit,heading,velocity)


def evaluate(t,coefs):
    return sum([entry[0]*(t**entry[1]) for entry in coefs])
