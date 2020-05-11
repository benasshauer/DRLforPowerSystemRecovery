# -*- coding: utf-8 -*-
"""
contains wrapper modules, plot functions and other methods that help the agent 
to interact with the environment. 
"""
import pandapower as pp
import pandapower.plotting as plot
import pandapower.topology as top
import matplotlib.pyplot as plt
import pandas
import networkx as nx
from gym import spaces
import random
import numpy as np
import seaborn
from PIL import Image

#def createNet(name):
#    verySimpleNet(name=name)

    
def create_line_net(): 
    net = pp.create_empty_network()
    
    b0 = pp.create_bus(net, vn_kv=110, geodata=(0,0))
    b1 = pp.create_bus(net, vn_kv=110, geodata=(1,1))
    b2 = pp.create_bus(net, vn_kv=110, geodata=(2,2))
    b3 = pp.create_bus(net, vn_kv=110, geodata=(3,3))
    b4 = pp.create_bus(net, vn_kv=110, geodata=(4,2))
    b5 = pp.create_bus(net, vn_kv=110, geodata=(5,1))
    b6 = pp.create_bus(net, vn_kv=110, geodata=(6,0))
    
    pp.create_ext_grid(net, b0) 
    
    pp.create_line(net, b0, b1, length_km=50., std_type="149-AL1/24-ST1A 110.0")
    pp.create_line(net, b1, b2, length_km=50., std_type="149-AL1/24-ST1A 110.0")
    pp.create_line(net, b2, b3, length_km=50., std_type="149-AL1/24-ST1A 110.0")
    pp.create_line(net, b3, b4, length_km=50., std_type="149-AL1/24-ST1A 110.0")
    pp.create_line(net, b4, b5, length_km=50., std_type="149-AL1/24-ST1A 110.0")
    pp.create_line(net, b5, b6, length_km=50., std_type="149-AL1/24-ST1A 110.0")
    
#    pp.create_load(net, b0, p_mw=50)
    pp.create_load(net, b1, p_mw=50)
    pp.create_load(net, b2, p_mw=50)
    pp.create_load(net, b3, p_mw=50)
    pp.create_load(net, b4, p_mw=50)
    pp.create_load(net, b5, p_mw=50)
    pp.create_load(net, b6, p_mw=50)
    
    pp.create_gen(net, b2, p_mw=150, max_p_mw=150, min_p_mw=0, slack=True)
    pp.create_gen(net, b5, p_mw=150, max_p_mw=150, min_p_mw=0, slack=True)
    
    return net
    
def switch_line_at_switch(net, switch):
    """
    takes a net and the index of a switch in net as input. 
    finds the other switch to the line at which the input switch is connected to,
    and switches both switches to the opposite position. 
    """
    l = net.switch.element.at[switch]
    connected_switches = net.switch[net.switch["element"]==l].index
    switch_position = net.switch.closed.at[switch]
    for s in connected_switches: 
        net.switch.closed.at[s] = not switch_position
        
def switch_line(net, line):
    """
    switches all switches connected to the given line to the opposite position
    the new switch position is the opposite position of the given connected with 
    the lowest index number
    """
    connected_switches = net.switch[net.switch["element"]==line].index
    switch_position = net.switch.closed.at[connected_switches[0]]
    for s in connected_switches: 
        # TODO change at to loc (pandas)
        net.switch.closed.at[s] = not switch_position
        
def get_line_states(net):
    """
    returns the switch states of all power lines of the given network
    NOTE: power lines are only between two buses
    NOTE 2: every power line has two switches (one at each bus)
    if the switch states of both switches are not the same, both switches are 
    switched to the switch position of the first switch (with the lower index)
    """
    line_states = np.array([])
    for i in range(len(net.line.index)):
        connected_switches = net.switch[net.switch['element']==i].index
        if net.switch.closed.at[connected_switches[0]] == net.switch.closed.at[connected_switches[1]]:
            line_states = np.append(line_states, net.switch.closed.at[connected_switches[0]])
        else:
            switch_line(net, i)
            line_states = np.append(line_states, net.switch.closed.at[connected_switches[0]])
    return line_states

def add_switches_to_lines(net):
    """
    adds two switches to every line of the network (bus-line-switches)
    """
    buses = net.bus.index 
    ls = net.line[(net.line.from_bus.isin(buses)) & (net.line.to_bus.isin(buses))]
    for _, line in ls.iterrows():
        pp.create_switch(net, line.from_bus, line.name, et='l', closed=True, 
                         type='LBS', name='Switch %s - %s' % (net.bus.name.at[line.from_bus], line['name']))
        pp.create_switch(net, line.to_bus, line.name, et='l', closed=True, type='LBS', 
                         name='Switch %s - %s' % (net.bus.name.at[line.to_bus], line['name']))
        
def run_powerflow(net, scale_loads=False, scale_gens=False):
    """
    runs a simple pf. If an error occurs (powerflow doesn't converge), 
    all loads and gens are scaled down to 0. Then the pf is run again. 
    
    Return
    ------
    single boolean: True if pf converged, false otherwise
    """
    pf_converges = True
    try:
        pp.runpp(net)
    except: 
#        print("there was an error: scaling isolated elements to 0")
        pf_converges = False
        # include connected elements for each bus in ub 
        if scale_loads: net.load.scaling = 0
        if scale_gens: 
            net.gen.scaling = 0
            net.sgen.scaling = 0
    finally:
        pp.runpp(net)
    return pf_converges    

def run_dcpowerflow(net, scale_loads=False, scale_gens=False):
    """
    runs a simple DC pf. If an error occurs (powerflow doesn't converge), 
    all loads and gens are scaled down to 0. Then the DC pf is run again 
    
    Return
    ------
    single boolean: True if pf converged, false otherwise
    """
    pf_converges = True
    try:
        pp.rundcpp(net)
    except: 
        pf_converges = False
        if scale_loads: net.load.scaling = 0
        if scale_gens: 
            net.gen.scaling = 0
            net.sgen.scaling = 0
    finally:
        pp.rundcpp(net)
    
    if scale_loads:
        net.load.scaling[net.res_load.p_mw == 0] = 0
        
    if scale_gens:
        ub = top.unsupplied_buses(net)
        islanded_gens = net.gen[net.gen["bus"].isin(ub)]
        net.gen.in_service[islanded_gens.index] = 0
        
    return pf_converges         
        
def plot_line_numbers(net):
    """
    shows an advanced plot of the grid that includes line numbers and loads
    """
    colors = seaborn.color_palette()
    
    lc = plot.create_line_collection(net, net.line.index, use_bus_geodata=True, color="grey", zorder=1) #create lines
    bc = plot.create_bus_collection(net, net.bus.index, size=.1, color=colors[0], zorder=2) #create buses
    tc = plot.create_trafo_collection(net, size=.1, color='grey')    
    lo = plot.create_load_collection(net, size=.1, color='green')
    ec = plot.create_ext_grid_collection(net, size=.3, color=colors[1])
    
    lines = net.line.index.tolist()
    x_coords = np.mean([net.bus_geodata.x.loc[net.line.from_bus].values, net.bus_geodata.x.loc[net.line.to_bus].values], axis=0)
    y_coords = np.mean([net.bus_geodata.y.loc[net.line.from_bus].values, net.bus_geodata.y.loc[net.line.to_bus].values], axis=0)    
    coords = zip(x_coords, y_coords)
    lic = plot.create_annotation_collection(size=.2, texts=np.char.mod('Line %d', lines), coords=coords, zorder=3, color='k')

    plot.draw_collections([lc, bc, tc, lic, lo, ec], figsize=(8,6)) # plot lines, buses and bus indices

def plot_visualize(net):
    """
    visualization of the network that includes more grid elements
    """
    colors = seaborn.color_palette()
    
    lc = plot.create_line_collection(net, net.line.index, use_bus_geodata=True, color="grey", zorder=1) #create lines
    bc = plot.create_bus_collection(net, net.bus.index, size=.1, color=colors[0], zorder=2) #create buses
    tc = plot.create_trafo_collection(net, size=.1, color='grey')    
    lo = plot.create_load_collection(net, size=.1, color='green')
    ec = plot.create_ext_grid_collection(net, size=.3, color=colors[1])
    sc = plot.create_line_switch_collection(net, size=.1, distance_to_bus=0.2, color='grey')
    gc = plot.create_gen_collection(net, size=.12)
    su = plot.create_sgen_collection(net, size=.15)
    
    lines = net.line.index.tolist()
    x_coords = np.mean([net.bus_geodata.x.loc[net.line.from_bus].values, net.bus_geodata.x.loc[net.line.to_bus].values], axis=0)
    y_coords = np.mean([net.bus_geodata.y.loc[net.line.from_bus].values, net.bus_geodata.y.loc[net.line.to_bus].values], axis=0)    
    coords = zip(x_coords, y_coords)
    lic = plot.create_annotation_collection(size=.2, texts=np.char.mod('Line %d', lines), coords=coords, zorder=3, color='k')

    buses = net.bus.index.tolist()
    bic = plot.create_annotation_collection(size=.15, texts=np.char.mod('%d', buses), coords=zip(net.bus_geodata.x, net.bus_geodata.y), zorder=3, color='k')

    plot.draw_collections([lc, sc, su, bc, tc, lic, bic, gc, lo, ec], figsize=(10,8)) # plot lines, buses and bus indices        
#    plot.draw_collections([lc, su, bc, ec], figsize=(10,8)) # plot lines, buses and bus indices 

def plot_render(net, save=False, draw=True, num='00', action='action_name'): 
    """
    renders a plot of the network with the last action executed. Can be used to 
    safe a frame of the restoration process. 
    """
    colors = seaborn.color_palette('RdBu_r', 7)
    
    lc = plot.create_line_collection(net, net.line.index, use_bus_geodata=True, color="grey", zorder=1) #create lines
    bc = plot.create_bus_collection(net, net.bus.index, size=.1, color=colors[0], zorder=2) #create buses
    ubc = plot.create_bus_collection(net, top.unsupplied_buses(net), size=.1, color='grey', zorder=3)
    ibc = plot.create_bus_collection(net, get_islanded_buses(net), size=.1, color='green', zorder=4)
    sbc = plot.create_bus_collection(net, net.storage.bus, size=.07, color='yellow', zorder=5)

    tc = plot.create_trafo_collection(net, size=.1, color='grey')    
    lo_on = plot.create_load_collection(net, loads=net.load.index[net.load.scaling==1], size=.1, color=colors[0])
    lo_off = plot.create_load_collection(net, loads=net.load.index[net.load.scaling==0], size=.1, color=colors[6])
    ec = plot.create_ext_grid_collection(net, size=.3, color=colors[4])
    sc = plot.create_line_switch_collection(net, size=.1, distance_to_bus=0.2, color='grey')
    gc = plot.create_gen_collection(net, size=.1)
    sg = plot.create_sgen_collection(net, sgens=net.sgen.index[net.sgen.in_service==1], size=.15, color='k', orientation=.5)
    sgo = plot.create_sgen_collection(net, sgens=net.sgen.index[net.sgen.in_service==0], size=.05, color='k', orientation=.5)

    lines = net.line.index.tolist()
    x_coords = np.mean([net.bus_geodata.x.loc[net.line.from_bus].values, net.bus_geodata.x.loc[net.line.to_bus].values], axis=0)
    y_coords = np.mean([net.bus_geodata.y.loc[net.line.from_bus].values, net.bus_geodata.y.loc[net.line.to_bus].values], axis=0)    
    coords = zip(x_coords, y_coords)
    lic = plot.create_annotation_collection(size=.15, texts=np.char.mod('L%d', lines), coords=coords, zorder=5, color='k')

    buses = net.bus.index.tolist()
    bic = plot.create_annotation_collection(size=.15, texts=np.char.mod('B%d', buses), coords=zip(net.bus_geodata.x, net.bus_geodata.y), zorder=5, color='k')

    plot.draw_collections([lc, sc, gc, bc, ubc, ibc, tc, sg, sgo, sbc, lic, bic, lo_on, lo_off, ec], figsize=(10,8), draw=draw) # plot lines, buses and bus indices        
    
    plt.text(np.mean(net.bus_geodata.x), max(net.bus_geodata.y)+.3, action, ha='center', va='center',
             size=20, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="k", lw=2))
    
    if save:
        plt.savefig("plots/frame{}.png".format(num))
        
def plot_sequence(env, model, runs=1, steps=19): 
    """
    Generates a gif of the restoration process
    """
    obs = env.reset()
    num = 0
    env.env_method('_render', True, False, format(num, '02d'))
    rewards = 0
    done = False
    action_list = env.env_method("get_action_df")[0]
    while not done:
        num += 1
        action, _states = model.predict(obs)
#        action_name = action_list.type.values[action][0]
        action_name = action_list.name[action[0]]
        obs, reward, done, info = env.step(action)
        env.env_method('_render', True, False, format(num, '02d'), action_name)
        rewards += reward
        
    names = ['plots/frame{:02d}.png'.format(i) for i in range(num)]
    images = []
    for n in names:
        frame = Image.open(n)
        images.append(frame)
    
    firstframe = Image.new("RGBA", (720, 576), (255,255,255))    
    firstframe.save('plots/sequence.gif', save_all=True, append_images=images[0:], duration=1000, loop=0)
    print("rewards: {}".format(rewards))
    
#def destroy_grid(net, loc, lines=[], n = 0):
#    """
#    opens n lines around the location (a bus that is connected to a line)
#    """
#    lines1 = net.line[net.line["to_bus"]==loc].index
#    lines2 = net.line[net.line["from_bus"]==loc].index
#    lines1_from = net.line.from_bus.at[lines1[0]]
#    lines2_to = net.line.to_bus.at[lines2[0]]
#    lines3 = net.line[net.line["to_bus"]==lines1_from].index
#    lines4 = net.line[net.line["from_bus"]==lines2_to].index
##    next_loc1 = net.line[net.line["to_bus"]==loc].from_bus.values[0]
##    next_loc2 = net.line[net.line["from_bus"]==loc].to_bus.values[0]
##    locs = np.append(next_loc1, next_loc2)
##    destroy_grid(net, next_loc1)
##    lines = lines1.append(lines2)
#    lines_12 = np.append(lines1, lines2)
#    lines_34 = np.append(lines3, lines4)
#    lines = np.append(lines_12, lines_34)
#    for i in lines: switch_line(net, i)
#    pp.runpp(net)   
#
#def destroy_grid2(net, line, n_max = 6, dc = True):
#    """
#    receives a pp network and a line, switches the corresponing line, and all lines
#    connected to the from_bus and the to_bus of the line
#    """
#    lines = [line]
#    to_bus = net.line.to_bus.at[line]
#    from_bus = net.line.from_bus.at[line]
#    lines = np.append(lines, list(net.line[net.line["from_bus"]==from_bus].index))
#    lines = np.append(lines, list(net.line[net.line["to_bus"]==to_bus].index))
#    if to_bus in net.line.from_bus:
#        next_line1 = net.line[net.line["from_bus"]==to_bus].index
#        lines = np.append(lines, list(next_line1))
#    if from_bus in net.line.to_bus:
#        next_line2 = net.line[net.line["to_bus"]==from_bus].index
#        lines = np.append(lines, list(next_line2))
#    for i in lines: switch_line(net, i)
#    
#    if dc: pp.rundcpp(net) 
#    else: pp.runpp(net)
    
def destroy_grid3(net, pos, n=2, dc=True):
    """
    receives a pp network and a bus index pos. The parameter n determines the 
    distance to pos, in which lines are switched. All lines with n nodes 
    between their corrensponding buses and pos are switched. 
    """
    distances = top.calc_distance_to_bus(net, pos, respect_switches=False)
    affected_buses = distances[distances <= n].index
    lines = []
    lines = np.append(lines, list(net.line[net.line["from_bus"].isin(affected_buses)].index))
    lines = np.append(lines, list(net.line[net.line["to_bus"].isin(affected_buses)].index))
    lines = np.unique(lines)
    for i in lines: switch_line(net, i)
    
    if dc: pp.rundcpp(net) 
    else: pp.runpp(net)
    
def add_sgen(net, bus, p_mw, q_mvar=0):
    """
    Adds sgen at given bus with p_mw
    """
    pp.create_sgen(net, bus, p_mw, q_mvar)
    
def switch_load(net, load):
    """
    switches laod from OFF to ON or vice versa. 
    """
    if net.load.scaling[load] == 1:
        net.load.scaling[load] = 0
    elif net.load.scaling[load] == 0:
        net.load.scaling[load] = 1
        
def switch_sgen(net, sgen):
    """
    switches sgen scaling from 0 to 1 or vice versa.
    """
    if net.sgen.scaling[sgen] == 1:
        net.sgen.scaling[sgen] = 0
    elif net.sgen.scaling[sgen] == 0:
        net.sgen.scaling[sgen] = 1
        
def get_islanded_buses(net):
    """
    returns buses with loads that are without connection to an external grid, 
    but supplied by a generator
    
    Return
    -------
    df of islanded buses of net
    """
    ub = top.unsupplied_buses(net)
    isolated_loads = net.load[net.load["bus"].isin(ub)]
    run_dcpowerflow(net)
    supplied_loads = net.load[net.res_load.p_mw > 0]
    islanded_loads = pandas.merge(supplied_loads, isolated_loads)
    return islanded_loads.bus

def scale_islanded_areas_old(net):
    """
    downscales loads in islanded areas where the nominal loads exceeds the 
    maximal power output of the supplying generator
    """
    run_dcpowerflow(net, scale_loads=True, scale_gens=False)
    islanded_gens = net.gen[net.gen["bus"].isin(get_islanded_buses(net))]
    mg = top.create_nxgraph(net)
    for gen_bus in islanded_gens.bus:
        area = list(top.connected_component(mg, gen_bus))
        area_loads = net.load[net.load["bus"].isin(area)].index
        area_gens = net.gen[net.gen["bus"].isin(area)].index
        area_slacks = net.gen[net.gen["bus"].isin(area) & net.gen["slack"] == 1].index
        area_bats = net.storage[net.storage["bus"].isin(area)].index
        area_sgens = net.sgen[net.sgen["bus"].isin(area)].index
        total_area_load = net.load.p_mw[area_loads].sum()
        total_area_genpower = net.gen.max_p_mw[area_gens].sum()
        total_area_batpower = - net.res_storage.p_mw[area_bats].sum() #negative for discharging 
        total_area_sgenpower = sum(net.sgen.p_mw[area_sgens] * net.sgen.scaling[area_sgens])
        total_area_p_mw = total_area_genpower + total_area_batpower + total_area_sgenpower
        area_slackpower = net.res_gen.p_mw[area_slacks].sum()
        # scale batteries and their corresponding virtual slack nodes
        for index, bat in net.storage[net.storage["bus"].isin(area)].iterrows():
            slack_index = net.gen.index[net.gen["bus"]==bat["bus"] & net.gen.index.isin(area_slacks)]
            slack_power = round(net.res_gen.p_mw[slack_index])
        for load_bus in net.load[net.load["bus"].isin(area)].bus:
            if total_area_load > total_area_p_mw:
                net.load.scaling[net.load["bus"]==load_bus] = 0
#                print("scaled islanded load at bus {} to {}".format(load_bus, 0))
#                run_dcpowerflow(net, scale_gens=False, scale_loads=False)
#                total_area_load -= net.load.p_mw[net.load["bus"]==load_bus][load_bus]
                total_area_load -= net.load.p_mw[load_bus]
                
def scale_islanded_areas(net): 
    """
    downscales loads in islanded areas, so that the supplying slack gen provides 
    no or negative power. If the slack gen provides negative power, it is assumed 
    that power generating units are scaled down. The actual process of downscaling 
    is not implemented in this version.  
    """
    mg = top.create_nxgraph(net)
    ub = top.unsupplied_buses(net)
    run_dcpowerflow(net, scale_gens=False, scale_loads=True)
    slacks = net.gen[net.gen['slack']==True]
    slack_power = round(net.res_gen.p_mw[slacks.index])
    for index, slack in slacks.iterrows():
        bus = slack["bus"]
        if bus in ub:
            area = list(top.connected_component(mg, bus))
            for load_bus in net.load[net.load["bus"].isin(area)].bus:
                if slack_power[index] > 0:
                    net.load.scaling[net.load["bus"]==load_bus] = 0
#                    slack_power[index] -= net.load[net.load["bus"]==load_bus].p_mw
                    load_index = net.load[net.load["bus"]==load_bus].index
                    slack_power[index] -= net.res_load.p_mw[load_index]
                
def set_unsupplied_areas_out_of_service(net):
    """
    scales loads in unsupplied areas to zero and sets gens and sgens out of 
    service
    """
    ub = top.unsupplied_buses(net)
    net.load.scaling[net.load["bus"].isin(ub)] = 0
    net.gen.in_service[(net.gen["bus"].isin(ub))&(net.gen["slack"]==False)] = 0
    net.sgen.in_service[net.sgen["bus"].isin(ub)] = 0
    net.storage.scaling[net.storage["bus"].isin(ub)] = 0
    
def plot_restoration_process(env, model, action_list, save=False):
    """
    plots the restoration process of the system over time
    """
    obs = env.reset()
    time = []
    load_supply = []
    rewards = []
    actions = []
    action_names = []
    done = False
    while not done:
        action, _states = model.predict(obs)
        actions.append(action)
        load_supply.append(env.get_attr("load_supply")[0])
        time.append(action_list[0].duration[action[0]])
        action_names.append(action_list[0].name[action[0]])
        obs, reward, done, info = env.step(action)
        rewards.append(reward[0])
    
    load_supply[-1] = 1
    time = np.cumsum(time)
    time = time - time[0]
    
    fig, ax1 = plt.subplots()
    color = 'black'
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Load Supply Level', color = color)
    ax1.plot(time, load_supply, color=color)
    
    ax2 = ax1.twinx()
    color = '#C0C0C0'
    ax2.set_ylabel('Rewards', color="gray")
    bar_plot = ax2.bar(time, rewards, color=color, width=.5)
    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.patch.set_visible(False)
    
    def autolabel(rects):
        for idx,rect in enumerate(bar_plot):
            ax2.text(rect.get_x() + rect.get_width()/2., 1.1*max(rewards),
                action_names[idx],
                ha='center', va='bottom', rotation=60)

#    autolabel(bar_plot)
    print("ACTION SEQUENCE: ", action_names)
    print("REWARDS: ", rewards)
    print("Loads supplied: ", load_supply)
    
    fig.tight_layout()
    if save: plt.savefig('plots/restoration_process.png')
    
    plt.show()
    
#    return load_supply, time-time[0], actions, rewards
    return

def add_slack_gens(net, slacks):
#    TODO implemet automated additions of gens to an existing pp network to 
#    modelling 
    return

def check_for_cranking_power(net, sgen_bus):
    """
    checks whether the provided bus is connected to an available power source e.g.
    external grid or a storage system that can supply the sgen with the required 
    cranking power to start.
    
    returns a single bool
    """
    ub = top.unsupplied_buses(net)
    if sgen_bus not in ub: return True
    
    mg = top.create_nxgraph(net)
    storages_av = net.storage.index[net.storage.soc_percent*net.storage.scaling > .2]
    storages_av_bus = net.storage.bus[storages_av] 
    
    cranking_av = [nx.has_path(mg, sgen_bus, stor) for stor in storages_av_bus]    
    
    return any(cranking_av)

def update_storage_SOC(net, duration_m):
    """
    updates SOCs of storage units, independent from cranking power provision
    """
    ub = top.unsupplied_buses(net)
    # connected stprages shouldnt be discharged 
    duration_h = duration_m/60
    storage_p = - net.res_storage.p_mw
    storage_p[~net.storage.bus.isin(ub)] = net.res_storage.p_mw #charging
    storage_e_mwh = net.storage.max_e_mwh
    discharge_soc = storage_p * duration_h / storage_e_mwh
    current_soc = net.storage.soc_percent
    
    # update SOCs and scale empty storage units to 0
    net.storage.soc_percent[current_soc - discharge_soc <= 0] = 0
    net.storage.scaling[current_soc - discharge_soc <= 0] = 0
    net.storage.soc_percent[current_soc - discharge_soc >= 0] -= discharge_soc 
    
def crank_sgen(net, sgen_index):
    """
    checks whether the provided sgen index is connected to an available power 
    source e.g. external grid or a storage system that can supply the sgen with the required 
    cranking power to start.
    
    Return
    -------
    single boolean that contains information whether the given sgen 
    was provided with cranking power by an external grid or via a storage unit. 
    """
    cranked_by_net = False
    cranked_by_storage = False
    
    sgen_bus = net.sgen.bus.at[sgen_index]
    
    ub = top.unsupplied_buses(net)
    
    if sgen_bus not in ub: 
        cranked_by_net = True
    
    else: 
        # identify sgen type to use technology dependent cranking power (optional)
        sgen_type = net.sgen.type[net.sgen["bus"] == sgen_bus]
    
        mg = top.create_nxgraph(net)
        storages_av = net.storage.index[net.storage.soc_percent * net.storage.in_service > .25]
        if len(storages_av) > 0:
            storages_av_bus = net.storage.bus[storages_av] 
    
            cranking_av = [nx.has_path(mg, sgen_bus, stor) for stor in storages_av_bus] 
            
            if len(cranking_av) > 0:
                # use the first storage unit (by index) to start the sgen
                net.storage.soc_percent[storages_av[0]] -= .25
                cranked_by_storage = any(cranking_av)
    
    return cranked_by_net, cranked_by_storage


    
             

