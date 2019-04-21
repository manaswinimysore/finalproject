#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 23:04:12 2019

@author: manaswini
"""

import tkinter as tk
import Linearregression
import graph
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
window=tk.Tk()

window.title("House hold power consumption Prediction")
window.geometry("800x700")

#--display function to button



def display():
	#.get() gets the input value from the user and return a string type
    Sub_metering_1=(float)(E_Sub_metering_1.get())
    Sub_metering_2=(float)(E_Sub_metering_2.get())
    Sub_metering_3=(float)(E_Sub_metering_3.get())
    Global_reactive_power = (float)(E_reactivepower.get())
    Voltage = (float)(E_voltage.get())
    Global_intensity = (float)(E_intensity.get())
    predictor =Linearregression.predict(Global_reactive_power,Voltage,Global_intensity,Sub_metering_1,Sub_metering_2,Sub_metering_3)
    msg="The global active for the next day is estimated as"+str(predictor)
    info_display = tk.Text(master = window,height = 10,width = 50,font=("Gadugi",13,"bold"))
    info_display.grid(column=0,row=14)

    info_display.insert(tk.END,msg)
    
    
def plot():
    array=graph.show()
    array=np.array(array)
    days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']

    fig = Figure(figsize=(6,6))
    a = fig.add_subplot(111)
        
    a.scatter(days,array,color='red')
    a.plot(days,array,color='blue')
    a.set_title ("Estimation Grid", fontsize=16)
    a.set_ylabel("Y", fontsize=14)
    a.set_xlabel("X", fontsize=14)

    canvas = FigureCanvasTkAgg(fig, window)
    canvas.get_tk_widget().grid(column=0,row=18)
    canvas.draw()
    '''figure2 = plt.Figure(figsize=(5,4), dpi=100)
    ax2 = figure2.add_subplot(111)
    line2 = FigureCanvasTkAgg(figure2, root)
    line2.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    df2.plot(kind='line', legend=True, ax=ax2, color='r',marker='o', fontsize=10)
    ax2.set_title('Year Vs. Unemployment Rate')'''



#-----Labels------
#--title---
myTitle = tk.Label(text="Household power consumption Prediction",font=("Algerian",15,"bold"))

myTitle.grid(column=0,row=0)

L_reactivepower = tk.Label(text="ReactivePOwer",font=("Sitka Subheading",12))
L_reactivepower.grid(column=0,row=2)

E_reactivepower = tk.Entry()
E_reactivepower.grid(column=1,row=2)

L_voltage=tk.Label(text="Voltage",font=("Sitka Subheading",12))
L_voltage.grid(column=0,row=3)

E_voltage = tk.Entry()
E_voltage.grid(column=1,row=3)

L_intensity = tk.Label(text="Intensity",font=("Sitka Subheading",12))
L_intensity.grid(column=0,row=4)

E_intensity = tk.Entry()
E_intensity.grid(column=1,row=4)

L_Sub_metering_1 = tk.Label(text="Sub_metering_1",font=("Sitka Subheading",12))
L_Sub_metering_1.grid(column=0,row=6)

E_Sub_metering_1 = tk.Entry()
E_Sub_metering_1.grid(column=1,row=6)

L_Sub_metering_2 = tk.Label(text="Sub_metering_2",font=("Sitka Subheading",12))
L_Sub_metering_2.grid(column=0,row=8)

E_Sub_metering_2 = tk.Entry()
E_Sub_metering_2.grid(column=1,row=8)

L_Sub_metering_3 = tk.Label(text="Sub_metering_3",font=("Sitka Subheading",12))
L_Sub_metering_3.grid(column=0,row=10)

E_Sub_metering_3= tk.Entry()
E_Sub_metering_3.grid(column=1,row=10)




#--Button--
#--command is to give an action to button
B_predict = tk.Button(text="PREDICT",font=("Sitka Subheading",12),command=display)
B_predict.grid(column=1,row=12)
#

B_plot = tk.Button(text="show power consumption for next week",font=("Sitka Subheading",12),command=plot)
B_plot.grid(column=1,row=18)
window.mainloop()
