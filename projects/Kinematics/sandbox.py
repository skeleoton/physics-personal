#imports
import matplotlib.pyplot as plt
import tkinter as tk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from eq import position, velocity, acceleration
from aux import on_closing

class KinematicsApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1450x800")
        self.root.title("Kinematic Graphs")
        self.root.resizable(width=False, height=False)
        self.root.attributes('-fullscreen', False)
        self.root.attributes('-zoomed', False)
        self.root.protocol("WM_DELETE_WINDOW", lambda: on_closing(self.root))
        
        #Frame Constructor
        self.graphs_Frame = tk.Frame(self.root)
        self.pva_Frame = tk.Frame(self.root)
        self.inputs_Frame = tk.Frame(self.root, bg="white", borderwidth=2, relief="solid")
        
        #Input Constructor 
        self.x0_slider = tk.Scale(self.inputs_Frame, from_=0, to=50, orient=tk.HORIZONTAL, length=250, command=lambda x: self.update_graphs(), resolution= 0.1, label="Initial position (x0)")
        self.v0_slider = tk.Scale(self.inputs_Frame, from_=-50, to=25, orient=tk.HORIZONTAL, length=250, command=lambda x: self.update_graphs(), resolution= 0.1, label="Initial velocity (v0)")
        self.a_slider = tk.Scale(self.inputs_Frame, from_=-25, to=10, orient=tk.HORIZONTAL, length=250, command=lambda x: self.update_graphs(), resolution= 0.1, label="Constant Acceleration (a)")
        self.t_slider = tk.Scale(self.inputs_Frame, from_=0, to=25, orient=tk.HORIZONTAL, length=250, command=lambda x: self.update_graphs(), resolution= 0.1, label="Time (s)")
        
        self.x0_slider.pack()
        self.v0_slider.pack()
        self.a_slider.pack()
        self.t_slider.pack()
        
        #Plot & Canvas Constructor
        self.position_figure, self.x_axis = plt.subplots(figsize=(3.5,3))
        self.velocity_figure, self.v_axis = plt.subplots(figsize=(3.5,3))
        self.acceleration_figure, self.a_axis = plt.subplots(figsize=(3.5,3))
        self.pva_figure, self.pva_axis = plt.subplots(figsize=(12,4))
        
        #Position Figure Config
        self.position_figure.suptitle("Position [L] vs. Time [T] Graph")
        self.x_axis.set_xlabel("Time (s)")
        self.x_axis.set_ylabel("Position (m)")
        self.x_axis.grid(True)

        #Velocity Config
        self.velocity_figure.suptitle("Velocity [L/T] vs. Time [T] Graph")
        self.v_axis.set_xlabel("Time (s)")
        self.v_axis.set_ylabel("Velocity (m/s)")
        self.v_axis.grid(True)

        #Acceleration Config
        self.acceleration_figure.suptitle("Acceleration [L/T^2] vs. Time [T] Graph")
        self.a_axis.set_xlabel("Time (s)")
        self.a_axis.set_ylabel("Velocity (m/s)")
        self.a_axis.grid(True)
        
        self.pva_figure.suptitle("Combinatory")
        
        self.x_canvas = FigureCanvasTkAgg(self.position_figure, master=self.graphs_Frame)
        self.v_canvas = FigureCanvasTkAgg(self.velocity_figure, master=self.graphs_Frame)
        self.a_canvas = FigureCanvasTkAgg(self.acceleration_figure, master=self.graphs_Frame)
        self.pva_canvas = FigureCanvasTkAgg(self.pva_figure, master=self.pva_Frame)
        
        #Packing
        self.x_canvas.get_tk_widget().pack(side=tk.LEFT, padx = 5, pady = 5)
        self.v_canvas.get_tk_widget().pack(side=tk.LEFT, padx = 5, pady = 5)
        self.a_canvas.get_tk_widget().pack(side=tk.LEFT, padx = 5, pady = 5)
        self.pva_canvas.get_tk_widget().pack()
        
        self.inputs_Frame.pack(side=tk.LEFT, anchor= tk.SW)
        self.graphs_Frame.pack(side=tk.TOP, anchor= tk.NW)
        self.pva_Frame.pack(side=tk.RIGHT)

        #Tightening layout & Updating graph
        self.position_figure.tight_layout()
        self.velocity_figure.tight_layout()
        self.acceleration_figure.tight_layout()
        self.pva_figure.tight_layout()
        self.update_graphs()

    
    def update_graphs(self):
        # Get slider values
        x0 = self.x0_slider.get()
        v0 = self.v0_slider.get()
        a = self.a_slider.get()
        t_max = self.t_slider.get()
        
        # Create time array
        t = np.linspace(0, t_max, 100)
        
        # Calculate data
        x_data = position(t, x0, v0, a)
        v_data = velocity(t, v0, a)
        a_data = acceleration(t, a)
        
        #Resetting axis and clearing plot
        self.x_axis.clear()
        self.x_axis.plot(t, x_data)
        self.x_axis.set_xlabel("Time (s)")
        self.x_axis.set_ylabel("Position (m)")
        self.x_axis.grid(True)
        
        self.v_axis.clear()
        self.v_axis.plot(t, v_data)
        self.v_axis.set_xlabel("Time (s)")
        self.v_axis.set_ylabel("Velocity (m/s)")
        self.v_axis.grid(True)
        
        self.a_axis.clear()
        self.a_axis.plot(t, a_data)
        self.a_axis.set_xlabel("Time (s)")
        self.a_axis.set_ylabel("Acceleration (m/s²)")
        self.a_axis.grid(True)
        
        self.pva_axis.clear()
        self.pva_axis.plot(t, x_data, label="Position")
        self.pva_axis.plot(t, v_data, label="Velocity")
        self.pva_axis.plot(t, a_data, label="Acceleration")
        self.pva_axis.set_xlabel("Time (s)")
        self.pva_axis.legend()
        self.pva_axis.grid(True)
        
        # Redraw
        self.x_canvas.draw()
        self.v_canvas.draw()
        self.a_canvas.draw()
        self.pva_canvas.draw()

        #Tightening after redraw
        self.position_figure.tight_layout()
        self.velocity_figure.tight_layout()
        self.acceleration_figure.tight_layout()
        self.pva_figure.tight_layout()


root = tk.Tk()
app = KinematicsApp(root)
root.mainloop()