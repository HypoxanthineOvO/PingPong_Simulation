import numpy as np
import taichi as ti
import matplotlib.pyplot as plt
import imageio
import configargparse
from typing import List

# Initialize
ti.init(arch=ti.cuda)
vec2f = ti.types.vector(2, float)

# Constants
resolution = (1800, 600) # 1800 - 3m, 600 - 1m -> 1cm = 6 pixels
#image = np.zeros((resolution[0], resolution[1], 3), dtype = np.float32)
image = ti.Vector.field(3, float, shape = resolution)

# Racket Color
racket_color = [
    # Deep Gray
    ti.Vector([0.3, 0.3, 0.3]),
    # Sponge Yellow
    ti.Vector([1, 0.8, 0.2]),
    # Wood Brown
    ti.Vector([0.4, 0.2, 0.1]),
    # Wood Brown
    ti.Vector([0.4, 0.2, 0.1]),
    # Sponge Orange
    ti.Vector([1, 0.5, 0.2]),
    # Red
    ti.Vector([1, 0, 0])
]

## Other Constants
dt_ = 2e-3

## Table Tennis Constants
e_t_table = 0.93
mu_table = 0.25

e_t_rubber = 0.9
k_pv, k_pw = 0.42, 1.5 * 10 ** 3

@ti.dataclass
class Racket:
    position: vec2f
    normal: vec2f
    velocity: vec2f

@ti.data_oriented
class Ball:
    def __init__(self, init_pos, init_vel, init_angvel, m = 2.7 * (10 ** -3), r = 0.02, scale = 600):
        self.mass = m
        self.radius = r
        
        # Position: p = (x, y)
        self.pos = ti.Vector.field(2, float, shape = (1))
        self.pos[0] = init_pos
        # Velocity: v = (vx, vy)
        self.vel = ti.Vector.field(2, float, shape = (1))
        self.vel[0] = init_vel
        # Angular Position: theta
        self.ang_pos = ti.field(float, shape = (1))
        # Angular Velocity: w
        self.ang_vel = ti.field(float, shape = (1))
        # Translate c/s to rad/s
        self.ang_vel[0] = init_angvel * 2 * ti.math.pi
        # Acceleration: a = (ax, ay)
        self.acc = ti.Vector.field(2, float, shape = (1))

        # Force: f = (fx, fy)
        self.force = ti.Vector.field(2, float, shape = (1))

        # Torque: tau =
        self.torque = ti.field(float, shape = (1))
        
        # Load the locs.npy
        locs = [
            [6,4], [6,5], [6,6], [6,18], [7,3], [7,4], [7,5], [7,6], [7,7], [7,15], [7,16], [7,17], [7,18], [7,19], [7,20], [8,3],
            [8,4], [8,5], [8,6], [8,7], [8,8], [8,9], [8,14], [8,15], [8,16], [8,17], [8,18], [8,19], [8,20], [9,3], [9,4], [9,5],
            [9,6], [9,7], [9,8], [9,9], [9,13], [9,14], [9,15], [9,16], [9,17], [9,18], [9,19], [9,20], [10,4], [10,5], [10,6], 
            [10,7], [10,8], [10,9], [10,10], [10,13], [10,14], [10,15], [10,16], [10,17], [10,18], [10,19], [11,4], [11,5], [11,6], 
            [11,7], [11,8], [11,9], [11,10], [11,13], [11,14], [11,15], [11,16], [11,17], [11,18], [11,19], [12,5], [12,6], [12,7], 
            [12,8], [12,9], [12,10], [12,12], [12,13], [12,14], [12,15], [12,16], [12,17], [12,18], [13,6], [13,7], [13,8], [13,9], 
            [13,10], [13,12], [13,13], [13,14], [13,15], [13,16], [13,17], [14,8], [14,9], [14,13], [14,14], [14,15]
            ]
        self.art_index =  ti.Vector.field(2, int, shape = (101))
        for i in range(101):
            self.art_index[i] = ti.Vector([locs[i][1] - 12, -locs[i][0] + 9])
        
        # # Generate Circle Index
        circ_idx_offset = []
        for i in range(-int(r * scale), int(r * scale) + 1):
            for j in range(-int(r * scale), int(r * scale) + 1):
                if i ** 2 + j ** 2 <= (r * scale) ** 2:
                    circ_idx_offset.append(ti.Vector([i, j]))

        self.circ_idx_off = ti.Vector.field(2, int, shape = (len(circ_idx_offset)))
        for i in range(len(circ_idx_offset)):
            self.circ_idx_off[i] = circ_idx_offset[i]
    
        self.is_collected = False
    
    @ti.kernel
    def add_force(self, f: vec2f):
        self.force[0] += f
    
    @ti.kernel
    def add_torque(self, t: ti.f32):
        self.torque[0] += t
    
    @ti.kernel
    def clear_force_and_torque(self):
        self.force[0] = ti.Vector([0, 0])
        self.torque[0] = 0
    
    @ti.func
    def Check_Bounce_with_table(self):
        # Table is at y = 0 ~ y = 0.01 m
        return (self.pos[0][1] - self.radius < 0.01) and (self.vel[0][1] < 0)
    
    @ti.func
    def Bounce_on_table_Naive(self):
        # Naive Bounce
        self.vel[0][1] = -self.vel[0][1]
        self.vel[0][1] *= 0.9
    @ti.func
    def Bounce_on_table(self):
        # Bounce on the table considering the rotation
        ## v: [vx, 0, vz]; w: [0, w, 0]
        v_in = ti.Vector([-self.vel[0][0], 0, self.vel[0][1]])
        w_in = ti.Vector([0, self.ang_vel[0], 0])
        v_in_T = ti.Vector([v_in[0] - self.radius * w_in[1], v_in[1] + self.radius * w_in[0], 0])
        v_in_T_norm = v_in_T.norm()
        
        nu_s = 1 - (5 / 2) * (mu_table) * (1 + e_t_table) * (abs(v_in[2]) / v_in_T_norm)
        alpha = mu_table * (1 + e_t_table) * (abs(v_in[2]) / v_in_T_norm)
        
        A_v_1 = ti.Matrix([[1 - alpha, 0, 0], [0, 1 - alpha, 0], [0, 0, -e_t_table]])
        B_v_1 = ti.Matrix([[0, alpha * self.radius, 0], [-alpha * self.radius, 0, 0], [0, 0, 0]])
        A_w_1 = ti.Matrix([[0, -1.5 * alpha / self.radius, 0], [1.5 * alpha / self.radius, 0, 0], [0, 0, 0]])
        B_w_1 = ti.Matrix([[1 - 1.5 * alpha, 0, 0], [0, 1 - 1.5 * alpha, 0], [0, 0, 1]])
        
        A_v_2 = ti.Matrix([[0.6, 0, 0], [0, 0.6, 0], [0, 0, -e_t_table]])
        B_v_2 = ti.Matrix([[0, 0.4 * self.radius, 0], [-0.4 * self.radius, 0, 0], [0, 0, 0]])
        A_w_2 = ti.Matrix([[0, -0.6 / self.radius, 0], [0.6 / self.radius, 0, 0], [0, 0, 0]])
        B_w_2 = ti.Matrix([[0.4, 0, 0], [0, 0.4, 0], [0, 0, 1]])
        
        A_v = A_v_1 if nu_s > 0 else A_v_2
        B_v = B_v_1 if nu_s > 0 else B_v_2
        A_w = A_w_1 if nu_s > 0 else A_w_2
        B_w = B_w_1 if nu_s > 0 else B_w_2
        
        # Noticed that v_in and w_in need be a column vector
        v_out = A_v @ v_in + B_v @ w_in
        w_out = A_w @ v_in + B_w @ w_in
        
        self.vel[0] = ti.Vector([-v_out[0], v_out[2]])
        self.ang_vel[0] = w_out[1]
    
    @ti.func
    def PreCheck(self, racket: Racket):
        dist_ver = (self.pos[0] - racket.position).dot(racket.normal)
        dist_hor = (self.pos[0] - racket.position).cross(racket.normal)
        return abs(dist_ver) < (self.radius + 0.03) and abs(dist_hor) < 0.075
    @ti.func
    def PreCheck_Get_Time(self, racket: Racket):
        dist_ver = (self.pos[0] - racket.position).dot(racket.normal) - self.radius - 0.01
        delta_v_ver = -self.vel[0].dot(racket.normal)
        return dist_ver / delta_v_ver
    
    @ti.func
    def Check_Bounce_with_racket(self, racket: Racket):
        # racket have two parameters: position and normal. Assume the width 7.5cm, as a line in the 2D plane
        ## Compute the distance between the ball and the line
        dist_ver = (self.pos[0] - racket.position).dot(racket.normal)
        dist_hor = (self.pos[0] - racket.position).cross(racket.normal)
        return (
            abs(dist_ver) < (self.radius + 0.01) 
            and 
            abs(dist_hor) < 0.075 
            and 
            #self.vel[0].dot(racket.normal) < 0
            # Also Consider the relative velocity
            (self.vel[0] - racket.velocity).dot(racket.normal) < 0
        )
    
    @ti.func
    def Bounce_with_racket(self, racket: Racket):
        # According to the paper
        ## Compute the relative velocity(assume racket is static)
        v_rel = self.vel[0] - racket.velocity
        normal_y = racket.normal
        normal_x = ti.Vector([-racket.normal[1], racket.normal[0]])
        ## Compute the normal component of the relative velocity
        v_rel_y = v_rel.dot(normal_y)
        v_rel_x = v_rel.dot(-normal_x)
        print(f"v_rel: {v_rel}, v_rel_x: {v_rel_x}, v_rel_y: {v_rel_y}")
        
        v_in = ti.Vector([v_rel_x, 0, v_rel_y])
        w_in = ti.Vector([0, self.ang_vel[0], 0])
        
        A_v = ti.Matrix([[1 - k_pv, 0, 0], [0, 1 - k_pv, 0], [0, 0, -e_t_rubber]])
        B_v = ti.Matrix([[0, self.radius, 0], [-self.radius, 0, 0], [0, 0, 0]]) * k_pv
        
        A_w = ti.Matrix([[0, -self.radius, 0], [self.radius, 0, 0], [0, 0, 0]]) * k_pw
        B_w = ti.Matrix([[1 - k_pw * self.radius ** 2, 0, 0], [0, 1 - k_pw * self.radius ** 2, 0], [0, 0, 1]])
        
        v_out_racket_coord = A_v @ v_in + B_v @ w_in
        w_out_racket_coord = A_w @ v_in + B_w @ w_in
        
        
        # Translate the result back to the original coordinate according to the normal
        v_out = v_out_racket_coord[0] * normal_x + v_out_racket_coord[2] * normal_y
        w_out= w_out_racket_coord[1]
        
        self.vel[0] = ti.Vector([v_out[0], v_out[1]])
        self.ang_vel[0] = -w_out
        
    @ti.kernel
    def update_ball(self, 
                    dt_: ti.f32, 
                    racket_l: Racket,
                    racket_r: Racket
    ) -> ti.i32:
        # Calculate the acceleration
        self.acc[0] = self.force[0] / self.mass
        # Calculate the new velocity
        self.vel[0] += self.acc[0] * dt_
        # Calculate the new position
        self.pos[0] += self.vel[0] * dt_
        
        # Calculate the torque
        I = 2 / 5 * self.mass * self.radius ** 2
        M = self.torque[0] / I
        
        # Calculate the new angular velocity
        self.ang_vel[0] += M * dt_
        # Calculate the new angular position
        self.ang_pos[0] += self.ang_vel[0] * dt_

        bounce_res = 0
        # Check List
        if self.Check_Bounce_with_racket(racket_l):
            self.Bounce_with_racket(racket_l)
            self.show_racket(racket_l, 1.0)
            print("========== Bounce with Racket Left ==========")
            bounce_res = 1
        elif (self.PreCheck(racket_l)):
            delta_time = self.PreCheck_Get_Time(racket_l)
            pre_racket = Racket(racket_l.position - racket_l.velocity * delta_time, racket_l.normal, racket_l.velocity)
            self.show_racket(pre_racket, 1.0)
        else:
            
            self.show_racket(racket_l, 0.35)
        
        if self.Check_Bounce_with_racket(racket_r):
            self.Bounce_with_racket(racket_r)
            self.show_racket(racket_r, 1.0)
            print("========== Bounce with Racket Right ==========")
            bounce_res = 2
        elif (self.PreCheck(racket_r)):
            delta_time = self.PreCheck_Get_Time(racket_r)
            pre_racket = Racket(racket_r.position - racket_r.velocity * delta_time, racket_r.normal, racket_r.velocity)
            self.show_racket(pre_racket, 1.0)
        else:
            self.show_racket(racket_r, 0.35)
        
        if self.Check_Bounce_with_table():
            self.Bounce_on_table()
            bounce_res = 3
            print("!!!======= Bounce with Table =======!!!")
        
        return bounce_res

    # GUI Function
    @ti.kernel
    def visualize(self, image: ti.template(), scale: ti.f32): # type: ignore
        # Generate Image
        for i in range(int(self.radius * scale * 2 + 1)):
            for j in range(int(self.radius * scale * 2 + 1)):
                idx = self.pos[0] * scale + self.circ_idx_off[i * int(self.radius * scale * 2 + 1) + j]
                if 0 <= idx[0] < resolution[0] and 0 <= idx[1]< resolution[1]:
                    image[int(idx[0]), int(idx[1])] = ti.Vector([1, 0.85, 0])
        # Draw Art Pixels
        for i in range(101):
            # Consider the rotation
            idx = self.pos[0] * scale + (
                self.art_index[i][0] * ti.Vector([ti.cos(self.ang_pos[0]), ti.sin(self.ang_pos[0])]) + 
                self.art_index[i][1] * ti.Vector([-ti.sin(self.ang_pos[0]), ti.cos(self.ang_pos[0])])
            )
            if 0 <= idx[0] < resolution[0] and 0 <= idx[1] < resolution[1]:
                image[int(idx[0]), int(idx[1])] = ti.Vector([1, 0.07, 0.6])
    @ti.func
    def show_racket(self, racket: Racket, color_scale: ti.f32):
        # Draw a line with center at racket.position, and normal as racket.normal, width 7.5cm
        ## The line is 7.5cm long, 1cm wide, map to 45 pixels long, 6 pixels wide
        ## The center is at (racket.position[0], racket.position[1]), map to (racket.position[0] * 600, racket.position[1] * 600)
        for i in range(-22, 23):
            # Calculate the position. i is the horizontal offset
            hor_vec = ti.Vector([-racket.normal[1], racket.normal[0]]).normalized()
            idx = racket.position * 600 + hor_vec * i
            if 0 <= idx[0] < resolution[0] and 0 <= idx[1] < resolution[1]:
                image[int(idx[0]), int(idx[1])] = racket_color[5] * color_scale
            idx = racket.position * 600 + hor_vec * i + racket.normal
            if 0 <= idx[0] < resolution[0] and 0 <= idx[1] < resolution[1]:
                image[int(idx[0]), int(idx[1])] = racket_color[4] * color_scale
            idx = racket.position * 600 + hor_vec * i + 2 * racket.normal
            if 0 <= idx[0] < resolution[0] and 0 <= idx[1] < resolution[1]:
                image[int(idx[0]), int(idx[1])] = racket_color[3] * color_scale
            idx = racket.position * 600 + hor_vec * i + 3 * racket.normal
            if 0 <= idx[0] < resolution[0] and 0 <= idx[1] < resolution[1]:
                image[int(idx[0]), int(idx[1])] = racket_color[2] * color_scale
            idx = racket.position * 600 + hor_vec * i + 4 * racket.normal
            if 0 <= idx[0] < resolution[0] and 0 <= idx[1] < resolution[1]:
                image[int(idx[0]), int(idx[1])] = racket_color[1] * color_scale
            idx = racket.position * 600 + hor_vec * i + 5 * racket.normal
            if 0 <= idx[0] < resolution[0] and 0 <= idx[1] < resolution[1]:
                image[int(idx[0]), int(idx[1])] = racket_color[0] * color_scale

    
@ti.kernel
def compute_g(m: ti.f32) -> ti.types.vector(2, float): # type: ignore
    g = ti.Vector([0, -9.6468]) # Consider the gravity and the buoyancy together as F_b = 1/64 * F_g
    return g * m

@ti.kernel
def compute_drag(v: ti.types.vector(2, float)) -> ti.types.vector(2, float): # type: ignore
    # Drag Coefficient
    ## C_d = 0.5, rho = 1.225, A = pi * r^2
    C_d, rho = 0.5, 1.225
    A = 3.14 * (0.02 ** 2)
    v_dir = v.normalized()
    
    # TODO: Check if |v| is too small
        
    vv = v.norm() ** 2
    drag = 0.5 * C_d * rho * A * vv * (-v_dir)
    
    return drag

@ti.kernel
def compute_magnus(v: ti.types.vector(2, float), w: ti.f32) -> ti.types.vector(2, float): # type: ignore
    # Magnus Coefficient
    # It's an 2D simulation
    C_s, rho = 1.23, 1.225
    A = 3.14 * (0.02 ** 2)
    v_dir = v.normalized()
    
    # TODO: Check if |v| is too small
    
    # fm_dir: Magnus Force Direction, rotate 90 degree if w > 0, -90 degree if w < 0
    fm_dir = ti.Vector([-v_dir[1], v_dir[0]])
    magnus = 0.5 * C_s * rho * A * 0.02 * v.norm() * w * (fm_dir)
    return magnus

@ti.kernel
def compute_drag_torque(w: ti.f32) -> ti.f32:
    # Drag Coefficient
    ## Formula: 0.5 * C_d * rho * A * v^2 * w * r
    ## C_d = 0.5, rho = 1.225, A = pi * r^2
    C_d, rho = 0.5, 1.225
    A = 3.14 * (0.02 ** 2)
    drag_torque = 0.5 * C_d * rho * A * (0.02 ** 2) * w * 0.02
    if w > 0:
        drag_torque = -drag_torque
    return drag_torque

# GUI Function

class GIF_Manager:
    def __init__(self) -> None:
        self.images = []
    
    def write_frame(self, img, correct_orientation=True):
        # Assert image is a numpy array
        assert isinstance(img, np.ndarray)
        # Check the dtype of the image
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        # Rotate the image
        if correct_orientation:
            img = np.rot90(img)
        
        self.images.append(img)
    
    def save_gif(self, filename, acc_ratio = 1.0, fps = 30):
        img_cnt = len(self.images)
        new_images = []
        cnt = 0
        while cnt < img_cnt:
            new_images.append(self.images[int(cnt)])
            cnt += acc_ratio
            
        # Save the images as a gif
        imageio.mimsave(filename, new_images, fps = fps)
    
    def save_video(self, filename, acc_ratio = 1.0, fps = 30):
        img_cnt = len(self.images)
        new_images = []
        cnt = 0
        while cnt < img_cnt:
            new_images.append(self.images[int(cnt)])
            cnt += acc_ratio
            
        # Save the images as a gif
        imageio.mimsave(filename, new_images, fps = fps)
    
    def save_images(self, filename, acc_ratio = 1.0):
        img_cnt = len(self.images)
        cnt = 0
        while cnt < img_cnt:
            imageio.imwrite(filename + str((self.images[int(cnt)])) + ".png", self.images[(self.images[int(cnt)])])
            cnt += acc_ratio
    
    def clear(self):
        self.images = []

@ti.kernel
def reset_image():
    image.fill(0.0)
    # Draw the Net. The net is 15cm high, 1 cm wide, map to 90 pixels high, 6 pixels wide
    # The center is at (1.5, 0.01), map to (900, 6)
    for i in range(3):
        for j in range(90):
            image[900 + i, j + 6] = ti.Vector([0.5, 0.5, 0.5])
            image[900 - i, j + 6] = ti.Vector([0.5, 0.5, 0.5])
    # Draw the Table. Table is 2.74m Long, and we visualize it surface at y = 0 ~ y = 0.01 m
    # The center is (1.5, 0) map to (900, 0); 2.74m -> 1644 pixels->Half: 822 pixels
    for i in range(822):
        for j in range(6):
            image[900 + i, j] = ti.Vector([0, 0, 0.55])
            
            image[900 - i, j] = ti.Vector([0, 0, 0.55])


if __name__ == "__main__":
    gui = ti.GUI("Table Tennis Simulator", resolution, fast_gui=True)
    
    # GIF Manager
    gif_manager = GIF_Manager()
    
    # Initialize the racket and the ball
    racket_left = Racket(
        position = vec2f(0.1, 0.15),
        normal = vec2f(0.15, 0.2).normalized(),
        #normal = vec2f(0, 1).normalized(),
        velocity = vec2f(6.5, -5)
    )
    
    racket_right = Racket(
        position = vec2f(2.5, 0.25),
        normal = vec2f(-0.5, -0.6).normalized(),
        velocity = vec2f(-8.5, 1)
    )
    
    racket_ls = [racket_left]
    racket_rs = [racket_right]
    
    # Add more rackets
    racket_l1 = Racket(
        position = vec2f(0.1, 0.25),
        normal = vec2f(0.5, -0.4).normalized(),
        velocity = vec2f(0, 2)
    )
    
    racket_r1 = Racket(
        position = vec2f(2.9, 0.15),
        normal = vec2f(-0.5, -0.3).normalized(),
        velocity = vec2f(-6.5, 4.5)
    )
    
    # If we need more rackets, we can add more. Now we repeat the same racket for 10 times.
    for i in range(10):
        racket_ls.append(racket_l1)
    
    for i in range(10):
        racket_rs.append(racket_r1)
    
    # Initialize the ball
    init_pos = vec2f(0.1, 0.25)
    init_vel = vec2f(0, 2)
    init_angvel = 0
    ball = Ball(init_pos, init_vel, init_angvel)
    
    ball_locs = []
    ball_vels = []
    ball_angvels = []
    
    reset = False
    
    bounce_record = [0, 0, 0] # [Left, Right, Table]
    
    while gui.running:
        # Reset
        if reset:
            reset = False
            ball = Ball(init_pos, init_vel, init_angvel)
            ball_locs = []
            
            reset_image()
            gif_manager.clear()
        
        # Clear
        reset_image()
        if gui.get_event(ti.GUI.PRESS):
            e = gui.event
            if e.key == ti.GUI.ESCAPE:
                break
            elif e.key == 'r':
                reset = True
        
        ball.clear_force_and_torque()
        g, drag, magnus = compute_g(ball.mass), compute_drag(ball.vel[0]), compute_magnus(ball.vel[0], ball.ang_vel[0])

        ball.add_force(g + drag + magnus)
        drag_torque = compute_drag_torque(ball.ang_vel[0])
        ball.add_torque(drag_torque)
        
        
        res = ball.update_ball(dt_, racket_left, racket_right)
        if res == 1:
            bounce_record[0] += 1
        elif res == 2:
            bounce_record[1] += 1
        elif res == 3:
            bounce_record[2] += 1
        ball.visualize(image, 600)
        
        
        # Print Every 4 frames
        if gui.frame % 10 == 0:
           print(f"Position: {ball.pos[0]}, \tVelocity: {ball.vel[0]}, \tAngular Velocity: {ball.ang_vel[0] / (2 * ti.math.pi)}r/s")
            
        if gui.frame % 100 == 0:
           print(f"Frame: {gui.frame}, \tBounce Record: {bounce_record}")
           print(f"Left racket: {racket_left.position}, \tRight racket: {racket_right.position}")
        # Record the ball location
        ball_locs.append(ball.pos[0].to_numpy())
        ball_vels.append(ball.vel[0].to_numpy())
        ball_angvels.append(ball.ang_vel[0])
        
        gui.set_image(image)
        gif_manager.write_frame(image.to_numpy())
        gui.show()
        
        # If the ball is out of the screen, break
        if (
            (ball.pos[0][0] - ball.radius) * 600 > resolution[0] or (ball.pos[0][1] - ball.radius) * 600 > resolution[1] or
            (ball.pos[0][0] + ball.radius) * 600 < 0 or (ball.pos[0][1] + ball.radius) * 600 < 0
            ):
            break
            
        # Set the racket location
        racket_left = racket_ls[bounce_record[0]]
        racket_right = racket_rs[bounce_record[1]]
    
    gif_manager.save_gif("Table_Tennis_Simulator.gif", acc_ratio = 2.5)
    gif_manager.save_video("Table_Tennis_Simulator.mp4", acc_ratio = 2.5)
    
    # Visualize the ball locations
    ball_locs = np.array(ball_locs)
    ball_vels = np.array(ball_vels)
    ball_angvels = np.array(ball_angvels)

    np.save("Ball_Locations.npy", ball_locs)
    np.save("Ball_Velocities.npy", ball_vels)
    np.save("Ball_Angular_Velocities.npy", ball_angvels)