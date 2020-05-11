# -*- coding: utf-8 -*-
"""
module that creates pandapower networks modified to be used with the restoration 
environment. Most notably, storage units are equipped with slack gen to enable 
the creation of power islands. 
"""
import pandapower as pp
import pandapower.networks as nw
import pp_helpers


def create_line_net(): 
    """
    returns a pandapower network that is shaped like a string with 7 nodes lined
    up. 
    """
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

def create_4GS_PV_Wind_Storage():
    """
    creates a simple 4 node network fitted for providing cranking power to 
    islanded areas i.e. storage units are treated as slack nodes
    
    Return
    -------
    4 node pp network
    """
    net = nw.case4gs()
    net["gen"].drop(0, inplace=True)

    pp.create_storage(net, 1, p_mw=-50, max_e_mwh=10)
    pp.create_gen(net, 1, p_mw=0, slack=True, type="bat")
    pp.create_sgen(net, 2, p_mw= 500, type="solar")
    pp.create_sgen(net, 3, p_mw = 500, type="wind")
    
    net.load.p_mw = 0.5*net.load.p_mw
    
    return net

def create_case14_PV_Wind_Storage(): 
    """
    creates IEEE 14 node network fitted for providing cranking power to 
    islanded areas i.e. storage units are treated as slack nodes
    
    Return
    ------
    pp IEEE 14 node network
    """
    
    net = nw.case14()
    for gen in net.gen.index: 
        net["gen"].drop(gen, inplace=True)
   
    net["shunt"].drop(0, inplace=True)
    
    pp.create_storage(net, 2, p_mw=-10, max_e_mwh=10, soc_percent=1)
    pp.create_gen(net, 2, p_mw=0, slack=True, type="bat")
    
    pp.create_storage(net, 12, p_mw=-10, max_e_mwh=10, soc_percent=1)
    pp.create_gen(net, 12, p_mw=0, slack=True, type="bat")
    
    pp.create_sgen(net, 7, p_mw=200, type="solar")
    pp.create_sgen(net, 10, p_mw=200, type="solar")
    
    pp.create_sgen(net, 11, p_mw = 200, type="wind")
    pp.create_sgen(net, 13, p_mw = 200, type="wind")
    
    return net

def create_case39_PV_Wind_Storage():
    """
    creates IEEE 39 bus test case a pp network. 
    
    Return
    ------
    pp IEEE 39 nide test case
    """
    net = nw.case39()
    
    for gen in net.gen.index: 
        net["gen"].drop(gen, inplace=True)
        
    pp.create_storage(net, 1, p_mw=-20, max_e_mwh=20, soc_percent=1) 
    pp.create_storage(net, 21, p_mw=-20, max_e_mwh=40, soc_percent=1)
    pp.create_storage(net, 12, p_mw=-10, max_e_mwh=20, soc_percent=1)
    pp.create_storage(net, 32, p_mw=-20, max_e_mwh=40, soc_percent=1)
    
    pp.create_gen(net, 1, p_mw=0, slack=True, type="bat")
    pp.create_gen(net, 21, p_mw=0, slack=True, type="bat")
    pp.create_gen(net, 12, p_mw=0, slack=True, type="bat")
    pp.create_gen(net, 32, p_mw=0, slack=True, type="bat")
    
    pp.create_sgen(net, 29, p_mw=250, type = "solar")
    pp.create_sgen(net, 31, p_mw=100, type = "solar")
    pp.create_sgen(net, 33, p_mw=50, type = "wind")
    pp.create_sgen(net, 34, p_mw=500, type = "solar")
    pp.create_sgen(net, 35, p_mw=100, type = "solar")
    pp.create_sgen(net, 36, p_mw=250, type = "solar")
    pp.create_sgen(net, 37, p_mw=300, type = "wind")
    pp.create_sgen(net, 38, p_mw=100, type = "wind")
    pp.create_sgen(net, 16, p_mw=200, type = "wind")
    pp.create_sgen(net, 4, p_mw=250, type = "solar")
    
#    decrease load size to match size of sgen capacity 
    net.load.p_mw = .2*net.load.p_mw
    
    return net