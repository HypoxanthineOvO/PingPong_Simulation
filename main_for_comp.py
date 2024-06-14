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


## Other Constants
dt_ = 2e-3

## Table Tennis Constants
e_t_table = 0.93
mu_table = 0.25

e_t_rubber = 0.9
k_pv, k_pw = 0.42, 1.5 * 10 ** 3


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
    
        
    @ti.kernel
    def update_ball(self, 
                    dt_: ti.f32
    ):
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
        
        if self.Check_Bounce_with_table():
            self.Bounce_on_table()
            print("!!!======= Bounce with Table =======!!!")
        

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
    
    init_pos = vec2f(0.1, 0.1)
    init_vel = vec2f(6, 2)
    init_angvel = 0
    ball = Ball(init_pos, init_vel, init_angvel)
    
    ball_locs = []
    
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
        
        
        res = ball.update_ball(dt_)
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
            
        # Record the ball location
        ball_locs.append(ball.pos[0].to_numpy())
        
        gui.set_image(image)
        #gif_manager.write_frame(image.to_numpy())
        gui.show()
        
        # If the ball is out of the screen, break
        if (ball.pos[0][0] - ball.radius) * 600 > resolution[0] or (ball.pos[0][1] - ball.radius) * 600 > resolution[1]:
            break
            
    
    #gif_manager.save_gif("Table_Tennis_Simulator.gif", acc_ratio = 2.5)
    #gif_manager.save_video("Table_Tennis_Simulator.mp4", acc_ratio = 2.5)
    
    # Visualize the ball locations
    ball_locs = np.array(ball_locs)
    # Save the ball locations
    np.save("ball_locs.npy", ball_locs)