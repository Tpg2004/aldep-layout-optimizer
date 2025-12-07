import streamlit as st
import random
import math
import copy
import os
import json
import heapq
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque
from contextlib import redirect_stdout
from matplotlib.colors import LinearSegmentedColormap

# --- App Configuration ---
st.set_page_config(layout="wide", page_title="ALDEP Layout Constructor")

# --- CUSTOM CSS (ALDEP Focused Theme) ---
CSS_STYLE = """
<style>

/* --- Main App Background (Purple Tones) --- */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #D8CCFF 0%, #FFFFFF 100%);
    background-attachment: fixed;
    color: #2E1A47;
}
/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #4F359B; /* Deep Purple Sidebar */
}
[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

/* Headings */
h1, h2, h3 {
    color: #4F359B; 
}
h1 {
    border-bottom: 3px solid #FF8C00; /* Orange Accent */
    padding-bottom: 5px;
}

/* Run Button (Orange) */
[data-testid="stButton"] button {
    background: linear-gradient(90deg, #FF8C00, #FFA500); 
    color: white;
    border-radius: 12px;
    font-weight: bold;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(255, 140, 0, 0.3);
}
[data-testid="stButton"] button:hover {
    transform: scale(1.05);
}

/* Metric Cards */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #C8B8FF;
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 2px 10px rgba(79, 53, 155, 0.1);
}
/* Metric Value Color */
[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #4F359B; 
}

/* Info Box */
[data-testid="stInfo"] {
    background-color: #E6E0FF;
    border: 1px solid #C8B8FF;
    color: #2E1A47;
}

</style>
"""
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# --- CONFIGURATION ---
SECONDS_PER_HOUR = 3600

# Default Machine Definitions (from previous context)
DEFAULT_MACHINES_JSON = """
[
    {"id": 0, "name": "Raw Material Input", "footprint": [2, 2], "cycle_time": 20, "clearance": 1, "zone_group": null},
    {"id": 1, "name": "1st Cutting", "footprint": [3, 3], "cycle_time": 35, "clearance": 1, "zone_group": 1},
    {"id": 2, "name": "Milling Process", "footprint": [4, 2], "cycle_time": 45, "clearance": 1, "zone_group": 1},
    {"id": 3, "name": "Drilling", "footprint": [2, 2], "cycle_time": 25, "clearance": 1, "zone_group": 1},
    {"id": 4, "name": "Heat Treatment A", "footprint": [3, 4], "cycle_time": 70, "clearance": 2, "zone_group": null},
    {"id": 5, "name": "Precision Machining A", "footprint": [3, 2], "cycle_time": 40, "clearance": 1, "zone_group": 2},
    {"id": 6, "name": "Assembly A", "footprint": [2, 3], "cycle_time": 55, "clearance": 2, "zone_group": 3},
    {"id": 7, "name": "Final Inspection A", "footprint": [1, 2], "cycle_time": 15, "clearance": 1, "zone_group": 3},
    {"id": 8, "name": "2nd Cutting", "footprint": [3, 2], "cycle_time": 30, "clearance": 1, "zone_group": 1},
    {"id": 9, "name": "Surface Treatment", "footprint": [2, 4], "cycle_time": 50, "clearance": 2, "zone_group": null},
    {"id": 10, "name": "Washing Process 1", "footprint": [2, 2], "cycle_time": 20, "clearance": 1, "zone_group": 2},
    {"id": 11, "name": "Heat Treatment B", "footprint": [4, 4], "cycle_time": 75, "clearance": 2, "zone_group": null},
    {"id": 12, "name": "Precision Machining B", "footprint": [2, 3], "cycle_time": 42, "clearance": 1, "zone_group": 2},
    {"id": 13, "name": "Component Assembly", "footprint": [3, 3], "cycle_time": 60, "clearance": 1, "zone_group": 3},
    {"id": 14, "name": "Quality Inspection B", "footprint": [2, 1], "cycle_time": 18, "clearance": 1, "zone_group": 3},
    {"id": 15, "name": "Packaging Line A", "footprint": [4, 3], "cycle_time": 30, "clearance": 2, "zone_group": null}
]
"""
# Default Process Sequence (as a JSON string)
DEFAULT_PROCESS_SEQUENCE_JSON = "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]"

# --- UTILITY FUNCTIONS (Needed for ALDEP placement checks and metrics) ---

def initialize_layout_grid(width, height):
    return [[-1 for _ in range(height)] for _ in range(width)]

def can_place_machine(grid, machine_footprint, machine_clearance, x, y, factory_w, factory_h):
    m_width, m_height = machine_footprint
    if not (0 <= x and x + m_width <= factory_w and 0 <= y and y + m_height <= factory_h): return False
    check_x_start = x - machine_clearance
    check_x_end = x + m_width + machine_clearance
    check_y_start = y - machine_clearance
    check_y_end = y + m_height + machine_clearance
    for i in range(max(0, check_x_start), min(factory_w, check_x_end)):
        for j in range(max(0, check_y_start), min(factory_h, check_y_end)):
            is_machine_body = (x <= i < x + m_width) and (y <= j < y + m_height)
            if grid[i][j] != -1:
                if is_machine_body or machine_clearance > 0: return False
    return True

def place_machine_on_grid(grid, machine_id, machine_footprint, x, y):
    m_width, m_height = machine_footprint
    for i in range(x, x + m_width):
        for j in range(y, y + m_height):
            grid[i][j] = machine_id

def get_machine_cycle_time(machine_id, all_machines_data):
    for m_def in all_machines_data:
        if m_def["id"] == machine_id: return m_def["cycle_time"]
    return float('inf')

def calculate_total_distance(machine_positions, process_sequence, distance_type='euclidean'):
    total_distance = 0
    if not machine_positions or len(process_sequence) < 2: return float('inf')
    
    for i in range(len(process_sequence) - 1):
        m1_id, m2_id = process_sequence[i], process_sequence[i+1]
        pos1 = machine_positions.get(m1_id); pos2 = machine_positions.get(m2_id)
        if not pos1 or not pos2: return float('inf')
        dx = pos1['center_x'] - pos2['center_x']; dy = pos1['center_y'] - pos2['center_y']
        
        if distance_type == 'manhattan': distance = abs(dx) + abs(dy)
        else: distance = math.sqrt(dx**2 + dy**2)
        total_distance += distance
    return total_distance

def calculate_area_metrics(machine_positions, machines_defs_ordered_by_proc_seq, factory_w, factory_h):
    total_footprint_area = sum(m["footprint"][0] * m["footprint"][1] for m in machines_defs_ordered_by_proc_seq if m["id"] in machine_positions)
    factory_area = factory_w * factory_h
    return total_footprint_area, total_footprint_area / factory_area

def calculate_machine_utilization_and_bottleneck(machine_positions, process_sequence, all_machines_data, travel_speed):
    if not machine_positions or not process_sequence: return (0.0, {})
    stage_times = {}
    for i in range(len(process_sequence)):
        current_machine_id = process_sequence[i]
        machine_cycle_time = get_machine_cycle_time(current_machine_id, all_machines_data)
        travel_time_to_next = 0.0
        if i < len(process_sequence) - 1:
            next_machine_id = process_sequence[i+1]
            pos_curr = machine_positions.get(current_machine_id); pos_next = machine_positions.get(next_machine_id)
            if pos_curr and pos_next and travel_speed > 0:
                distance = math.sqrt((pos_curr['center_x'] - pos_next['center_x'])**2 + (pos_curr['center_y'] - pos_next['center_y'])**2)
                travel_time_to_next = distance / travel_speed
        current_stage_total_time = machine_cycle_time + travel_time_to_next
        stage_times[current_machine_id] = current_stage_total_time

    max_stage_time = max(stage_times.values()) if stage_times else 0.0
    if max_stage_time <= 0 or max_stage_time == float('inf'): return (0.0, {})
    throughput = SECONDS_PER_HOUR / max_stage_time
    
    utilization_data = {
        mid: min(1.0, get_machine_cycle_time(mid, all_machines_data) / max_stage_time) 
        for mid in process_sequence
    }
    return (max_stage_time, utilization_data)

def calculate_layout_metrics(machine_positions, machines_defs, process_seq_ids, factory_w, factory_h, target_prod_throughput, material_travel_speed):
    
    fitness_weights = {
        "throughput": 1.0, "distance": 0.005, "zone_penalty": 0.5, "mhs_turn_penalty": 0.001, 
        "utilization_bonus": 0.1, "bonus_for_target_achievement": 0.2
    }
    constraint_params = {
        "zone_1_target_machines": [1, 2, 3, 8], "zone_1_max_spread_distance": 12.0, "area_util_min_threshold": 0.40
    }
    
    max_stage_time, utilization_data = calculate_machine_utilization_and_bottleneck(
        machine_positions, process_seq_ids, machines_defs, material_travel_speed)
    throughput = SECONDS_PER_HOUR / max_stage_time if max_stage_time > 0 and max_stage_time != float('inf') else 0.0
    
    total_euclidean_dist = calculate_total_distance(machine_positions, process_seq_ids, distance_type='euclidean')
    total_manhattan_dist = calculate_total_distance(machine_positions, process_seq_ids, distance_type='manhattan')

    zone_penalty = 0.0 # ALDEP is complex to integrate constraints, simplify to 0 for this demo
    _, utilization_ratio = calculate_area_metrics(machine_positions, machines_defs, factory_w, factory_h)
    
    utilization_bonus = 0.0
    if utilization_ratio >= constraint_params['area_util_min_threshold']:
        utilization_bonus = (utilization_ratio - constraint_params['area_util_min_threshold']) * factory_w * factory_h * fitness_weights["utilization_bonus"]

    fitness_val = (fitness_weights["throughput"] * throughput) - (fitness_weights["distance"] * total_euclidean_dist) + utilization_bonus

    return {
        "fitness": fitness_val,
        "throughput": throughput,
        "distance": total_euclidean_dist,
        "utilization_ratio": utilization_ratio,
        "machine_utilization": utilization_data
    }

def visualize_layout_plt(machine_positions_map, factory_w, factory_h, process_sequence_list, machine_definitions_list):
    """Visualizes the final ALDEP layout."""
    
    fig, ax = plt.subplots(1, figsize=(max(10, factory_w/2), max(10, factory_h/2 + 1))) 
    
    # Styling
    fig.patch.set_facecolor('#E6E0FF')
    ax.set_facecolor('#FFFFFF') 
    ax.tick_params(colors='#4F359B')
    ax.xaxis.label.set_color('#4F359B')
    ax.yaxis.label.set_color('#4F359B')
    ax.title.set_color('#4F359B')
    ax.spines['bottom'].set_color('#4F359B')
    ax.spines['top'].set_color('#4F359B')
    ax.spines['left'].set_color('#4F359B')
    ax.spines['right'].set_color('#4F359B')

    ax.set_xlim(-0.5, factory_w - 0.5)
    ax.set_ylim(-0.5, factory_h - 0.5)
    ax.set_xticks(range(factory_w))
    ax.set_yticks(range(factory_h))
    ax.set_xticklabels(range(factory_w))
    ax.set_yticklabels(range(factory_h))
    ax.grid(True, linestyle='--', alpha=0.5, color='#4F359B')
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis() 

    cmap = plt.colormaps.get_cmap('viridis')
    num_machines = len(machine_definitions_list)
    machines_dict_by_id = {m['id']: m for m in machine_definitions_list}
    
    # Draw Machines
    for i, machine_id_in_seq in enumerate(process_sequence_list):
        if machine_id_in_seq in machine_positions_map:
            pos_data = machine_positions_map[machine_id_in_seq]
            machine_info = machines_dict_by_id.get(machine_id_in_seq)

            if machine_info:
                x, y = pos_data['x'], pos_data['y']
                width, height = machine_info['footprint']
                clearance = machine_info.get('clearance', 0)
                
                color_value = i / max(num_machines - 1, 1) 
                
                # Draw Clearance
                if clearance > 0:
                    rect_clearance = patches.Rectangle(
                        (x - clearance - 0.5, y - clearance - 0.5),
                        width + 2 * clearance, height + 2 * clearance,
                        linewidth=1, edgecolor=cmap(color_value),
                        facecolor='none', linestyle=':', alpha=0.4
                    )
                    ax.add_patch(rect_clearance)

                # Draw Machine Body
                rect_body = patches.Rectangle((x - 0.5, y - 0.5), width, height,
                                              linewidth=1.5, edgecolor='black',
                                              facecolor=cmap(color_value), alpha=0.9)
                ax.add_patch(rect_body)

                # Label
                ax.text(x + width / 2 - 0.5, y + height / 2 - 0.5, 
                        f"M{machine_id_in_seq}\n({machine_info['name'][:5]}...)",
                        ha='center', va='center', fontsize=6, color='white', weight='bold')

    plt.title("ALDEP Constructed Layout (Flow Adjacency Optimized)", fontsize=12)
    plt.xlabel("Factory Width (X)"); plt.ylabel("Factory Height (Y)")
    
    return fig


# ----------------------------------------------------------------------------------------------------
# ----------------------------------- ALDEP Core Function --------------------------------------------
# ----------------------------------------------------------------------------------------------------

def run_aldep_construction(factory_w, factory_h, machines_definitions, process_sequence):
    """
    Simulates the ALDEP construction process: sequentially placing machines adjacent 
    to the previously placed machine to maximize relationship scores (adjacency).
    """
    
    machines_dict = {m['id']: m for m in machines_definitions}
    machines_for_placement = [machines_dict[pid] for pid in process_sequence]
    
    aldep_layout_coords = []
    aldep_positions_map = {}
    grid = initialize_layout_grid(factory_w, factory_h)
    
    # 1. Place the first machine (ALDEP principle: usually central or strategic)
    first_machine = machines_for_placement[0]
    m_footprint, m_clearance, m_id = first_machine["footprint"], first_machine.get("clearance", 0), first_machine["id"]
    
    # Start at a guaranteed corner (0, 0)
    start_x, start_y = 0, 0
    
    # Fallback to nearest valid space if (0,0) is invalid (rare for M0)
    if not can_place_machine(grid, m_footprint, m_clearance, start_x, start_y, factory_w, factory_h):
        start_x, start_y = factory_w // 2, factory_h // 2 

    place_machine_on_grid(grid, m_id, m_footprint, start_x, start_y)
    aldep_layout_coords.append((start_x, start_y))
    aldep_positions_map[m_id] = {"x": start_x, "y": start_y, 
                                 "center_x": start_x + m_footprint[0]/2.0, 
                                 "center_y": start_y + m_footprint[1]/2.0}

    # 2. Sequentially place remaining machines adjacent to the previous one
    for i in range(1, len(machines_for_placement)):
        
        current_machine = machines_for_placement[i]
        previous_machine_id = process_sequence[i - 1]
        
        # Get the position and def of the machine placed in the previous step
        previous_pos_coords = aldep_layout_coords[-1]
        previous_def = machines_dict[previous_machine_id]
        
        m_footprint, m_clearance, m_id = current_machine["footprint"], current_machine.get("clearance", 0), current_machine["id"]
        
        best_new_pos = None

        # Try to place adjacent to the previous machine (4 main directions)
        px, py = previous_pos_coords
        pw, ph = previous_def["footprint"]
        
        adj_candidates = []
        # Right, Left, Bottom, Top of previous machine
        adj_candidates.append((px + pw + m_clearance, py)) 
        adj_candidates.append((px - m_footprint[0] - m_clearance, py)) 
        adj_candidates.append((px, py + ph + m_clearance)) 
        adj_candidates.append((px, py - m_footprint[1] - m_clearance)) 
        
        random.shuffle(adj_candidates) # Randomize the order of direction checking

        for new_x, new_y in adj_candidates:
            if can_place_machine(grid, m_footprint, m_clearance, new_x, new_y, factory_w, factory_h):
                best_new_pos = (new_x, new_y)
                break
        
        if best_new_pos:
            # Successful adjacent placement
            place_machine_on_grid(grid, m_id, m_footprint, best_new_pos[0], best_new_pos[1])
            aldep_layout_coords.append(best_new_pos)
            aldep_positions_map[m_id] = {"x": best_new_pos[0], "y": best_new_pos[1], 
                                         "center_x": best_new_pos[0] + m_footprint[0]/2.0, 
                                         "center_y": best_new_pos[1] + m_footprint[1]/2.0}
        else:
            # Fallback (Sweep): Find the first available random spot if adjacency fails
            fallback_pos = None
            for x_rand in range(factory_w - m_footprint[0] + 1):
                for y_rand in range(factory_h - m_footprint[1] + 1):
                    if can_place_machine(grid, m_footprint, m_clearance, x_rand, y_rand, factory_w, factory_h):
                        fallback_pos = (x_rand, y_rand)
                        break
                if fallback_pos: break
            
            if fallback_pos:
                place_machine_on_grid(grid, m_id, m_footprint, fallback_pos[0], fallback_pos[1])
                aldep_layout_coords.append(fallback_pos)
                aldep_positions_map[m_id] = {"x": fallback_pos[0], "y": fallback_pos[1], 
                                             "center_x": fallback_pos[0] + m_footprint[0]/2.0, 
                                             "center_y": fallback_pos[1] + m_footprint[1]/2.0}
            else:
                 # Should not happen if factory is large enough
                 st.warning(f"Warning: Could not place Machine {m_id} using ALDEP method.")
                 aldep_layout_coords.append((-1, -1))
                 
    # 3. Calculate metrics for the resulting layout
    aldep_metrics = calculate_layout_metrics(aldep_positions_map, machines_definitions, process_sequence, 
                                             factory_w, factory_h, 35, 0.5) # Use default target/speed for metrics
    
    return aldep_positions_map, aldep_metrics


# ----------------------------------------------------------------------------------------------------
# ------------------------------------ MAIN STREAMLIT EXECUTION --------------------------------------
# ----------------------------------------------------------------------------------------------------

st.header("ALDEP Factory Layout Construction")
st.info("This application constructs a single layout using the Automated Layout Design Program (ALDEP) principle, prioritizing adjacency based on the sequential process flow. This result can be compared against a GA search.")

col_input, col_metrics = st.columns([1, 2])

with col_input:
    st.subheader("Factory & Flow Inputs")
    factory_w = st.number_input("Factory Width (units)", min_value=10, max_value=100, value=20) 
    factory_h = st.number_input("Factory Height (units)", min_value=10, max_value=100, value=20) 
    
    st.markdown("---")
    material_travel_speed = st.slider("Material Travel Speed (units/sec)", 0.1, 5.0, 0.5, 0.1)
    target_tph = st.number_input("Target Production (units/hr)", min_value=1, max_value=200, value=35)

    with st.expander("Machine & Process Data", expanded=False):
        machines_json = st.text_area("Machines Definitions (JSON)", value=DEFAULT_MACHINES_JSON, height=200)
        process_seq_json = st.text_area("Process Sequence (JSON)", value=DEFAULT_PROCESS_SEQUENCE_JSON, height=50)

    run_button = st.button("🏗️ Construct ALDEP Layout", type="primary", use_container_width=True)

with col_metrics:
    
    if run_button:
        try:
            machines_definitions = json.loads(machines_json)
            process_sequence = json.loads(process_seq_json)
        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON input: {e}")
            st.stop()

        with st.spinner("Constructing layout and calculating metrics..."):
            
            aldep_positions, aldep_metrics = run_aldep_construction(
                factory_w, factory_h, machines_definitions, process_sequence
            )
            
            # --- RESULTS DISPLAY ---
            st.subheader("ALDEP Construction Results")
            col_m1, col_m2, col_m3 = st.columns(3)
            
            col_m1.metric("ALDEP Fitness Score", f"{aldep_metrics['fitness']:.2f}", help="Fitness is primarily driven by minimizing distance in this constructive model.")
            col_m2.metric("Total Euclidean Distance", f"{aldep_metrics['distance']:.2f} units")
            col_m3.metric("Hourly Throughput (TPH)", f"{aldep_metrics['throughput']:.2f}")
            
            st.markdown("---")
            
            st.pyplot(visualize_layout_plt(aldep_positions, factory_w, factory_h, process_sequence, machines_definitions))
            
            st.subheader("Layout Details")
            st.metric("Area Utilization", f"{aldep_metrics['utilization_ratio']:.2%}")
            
            st.info("""
            **Verification Note:** Since ALDEP optimizes for adjacency rather than fitness, its score is often 
            lower than a true GA optimization. The primary comparison here is the resulting **distance** and **throughput**.
            """)

    else:
        st.info("Click 'Construct ALDEP Layout' to generate the rule-based solution.")
