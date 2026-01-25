"""
QUICK CUSTOMIZATION TEMPLATE
=============================

Copy this template to create your own block diagrams quickly.
Just fill in the blocks and connections for your specific system.
"""

from generatefigs import BlockDiagram

def create_my_diagram(filename: str = "my_custom_diagram") -> BlockDiagram:
    """
    Create your custom control system block diagram.
    
    STEP 1: Initialize diagram
    """
    diagram = BlockDiagram(filename, "My Control System Title")
    
    """
    STEP 2: Add blocks
    
    Common shapes:
    - 'box' (default): Rectangular blocks for systems
    - 'circle': For summing junctions
    - 'ellipse': For oval shapes
    - 'diamond': For decision points
    
    Sizes: width and height in inches (typical: 1.5 x 0.8)
    """
    
    # Example blocks - CUSTOMIZE THESE
    diagram.add_block(
        "input",                    # Unique ID (no spaces)
        "Input\nSignal",            # Display label (use \n for new lines)
        width=1.2,                  # Width in inches
        height=0.8                  # Height in inches
    )
    
    diagram.add_block(
        "controller",
        "PID\nController",
        width=1.5,
        height=0.8
    )
    
    diagram.add_block(
        "actuator",
        "Motor\nDynamics",
        width=1.5,
        height=0.8
    )
    
    diagram.add_block(
        "plant",
        "Physical\nSystem",
        width=1.5,
        height=1.0
    )
    
    diagram.add_block(
        "sensor",
        "Sensor",
        width=1.2,
        height=0.6
    )
    
    """
    STEP 3: Add summing junctions
    
    inputs: List of '+' or '-' for each input arm
    size: Diameter of circle (typical: 0.4)
    """
    diagram.add_summing_junction(
        "sum_error",                # Unique ID
        inputs=['+', '-'],          # Signs for inputs
        size=0.4
    )
    
    diagram.add_summing_junction(
        "sum_disturbance",
        inputs=['+', '+'],
        size=0.4
    )
    
    """
    STEP 4: Add forward path connections
    
    Basic format:
    diagram.add_edge(from_node, to_node, label="signal_name")
    
    With ports (for precise connection points):
    diagram.add_edge(from_node, to_node, from_port='e', to_port='w', label="signal")
    Ports: 'n'=north/top, 's'=south/bottom, 'e'=east/right, 'w'=west/left
    """
    
    # Forward signal flow - CUSTOMIZE THESE
    diagram.add_edge("input", "sum_error", label="r(t)")
    diagram.add_edge("sum_error", "controller", label="e", from_port='e')
    diagram.add_edge("controller", "actuator", label="u_c")
    diagram.add_edge("actuator", "sum_disturbance", label="u")
    diagram.add_edge("sum_disturbance", "plant", label="u_total")
    diagram.add_edge("plant", "sensor", label="x")
    
    # Output signal
    diagram.add_invisible_node("output")  # Invisible endpoint
    diagram.add_edge("sensor", "output", label="y(t)", from_port='e')
    
    """
    STEP 5: Add feedback paths
    
    Use add_feedback_path for signals that close the loop.
    Set constraint=False to prevent layout distortion.
    """
    diagram.add_feedback_path(
        "sensor",                   # From node
        "sum_error",                # To node
        label="y",                  # Signal label
        constraint=False            # Don't constrain layout
    )
    
    """
    STEP 6: Add disturbances (optional)
    
    Creates an arrow entering from above/below/side
    """
    diagram.add_disturbance_input(
        "disturbance1",             # Unique ID for label
        "External\nDisturbance",    # Display text
        "sum_disturbance",          # Target block
        'n'                         # Port: 'n'=from top
    )
    
    diagram.add_disturbance_input(
        "noise",
        "Sensor\nNoise",
        "sensor",
        'n'
    )
    
    """
    STEP 7: Advanced - Align blocks (optional)
    
    Forces blocks to appear at same vertical/horizontal level
    """
    # diagram.set_rank_same(['controller', 'actuator', 'plant'])
    
    return diagram


# ============================================================================
# To use this template:
# ============================================================================
# 1. Copy this file and rename it
# 2. Edit the block names, labels, and connections above
# 3. Import and use in main generatefigs.py:
#
#    from my_custom_template import create_my_diagram
#    
#    # Then in main():
#    diagram = create_my_diagram("my_output_name")
#    diagram.render(format='pdf', output_dir='figures')
#
# 4. Or run standalone:
if __name__ == '__main__':
    diagram = create_my_diagram()
    diagram.render(format='pdf', output_dir='figures_diagrams')
    print("✓ Generated my_custom_diagram.pdf")


# ============================================================================
# COMMON PATTERNS
# ============================================================================

"""
PATTERN 1: Simple Feedback Loop
--------------------------------
Reference -> [Sum] -> Controller -> Plant -> Output
               ↑                              ↓
               └──────────[Sensor]────────────┘
"""
def pattern_simple_feedback():
    d = BlockDiagram("simple_feedback")
    d.add_block("ref", "r(t)", width=0.8)
    d.add_summing_junction("sum1")
    d.add_block("ctrl", "C(s)", width=1.2)
    d.add_block("plant", "P(s)", width=1.2)
    
    d.add_edge("ref", "sum1")
    d.add_edge("sum1", "ctrl", label="e")
    d.add_edge("ctrl", "plant", label="u")
    d.add_feedback_path("plant", "sum1", label="y")
    
    d.add_invisible_node("out")
    d.add_edge("plant", "out", label="y(t)")
    return d


"""
PATTERN 2: Cascaded (Inner + Outer) Control
--------------------------------------------
Ref -> [Sum1] -> Outer -> [Sum2] -> Inner -> Plant
        ↑    Ctrl         ↑    Ctrl            ↓
        │                 └────────────────────┤
        └──────────────────────────────────────┘
"""
def pattern_cascaded():
    d = BlockDiagram("cascaded_control")
    d.add_block("ref", "x_ref", width=1.0)
    d.add_summing_junction("sum_outer")
    d.add_block("outer", "Position\nControl", width=1.5)
    d.add_summing_junction("sum_inner")
    d.add_block("inner", "Velocity\nControl", width=1.5)
    d.add_block("plant", "Motor", width=1.2)
    
    d.add_edge("ref", "sum_outer")
    d.add_edge("sum_outer", "outer", label="e_x")
    d.add_edge("outer", "sum_inner", label="v_ref")
    d.add_edge("sum_inner", "inner", label="e_v")
    d.add_edge("inner", "plant", label="i")
    
    d.add_feedback_path("plant", "sum_outer", label="x")
    d.add_feedback_path("plant", "sum_inner", label="v")
    
    d.add_invisible_node("out")
    d.add_edge("plant", "out", label="x(t)")
    return d


"""
PATTERN 3: Observer-Based Control
----------------------------------
Ref -> Controller -> Plant -> Output
       ↑              ↓
       └───Observer───┘
"""
def pattern_observer():
    d = BlockDiagram("observer_based")
    d.add_block("ref", "r(t)", width=1.0)
    d.add_summing_junction("sum1")
    d.add_block("ctrl", "Controller", width=1.5)
    d.add_block("plant", "Plant", width=1.5)
    d.add_block("observer", "State\nObserver", width=1.5)
    
    d.add_edge("ref", "sum1")
    d.add_edge("sum1", "ctrl", label="e")
    d.add_edge("ctrl", "plant", label="u")
    d.add_edge("plant", "observer", label="y")
    d.add_feedback_path("observer", "sum1", label="x̂")
    
    d.add_invisible_node("out")
    d.add_edge("plant", "out", label="y(t)")
    return d


"""
PATTERN 4: Feedforward + Feedback
----------------------------------
           ┌─────Feedforward────┐
           ↓                     ↓
Ref -> [Sum] -> Feedback -> [Sum] -> Plant
        ↑                           ↓
        └───────────────────────────┘
"""
def pattern_feedforward():
    d = BlockDiagram("feedforward_feedback")
    d.add_block("ref", "r(t)", width=1.0)
    d.add_summing_junction("sum_err")
    d.add_block("fb_ctrl", "Feedback\nControl", width=1.5)
    d.add_block("ff_ctrl", "Feedforward", width=1.5)
    d.add_summing_junction("sum_ctrl")
    d.add_block("plant", "Plant", width=1.5)
    
    d.add_edge("ref", "sum_err")
    d.add_edge("ref", "ff_ctrl", label="r", from_port='e')
    d.add_edge("sum_err", "fb_ctrl", label="e")
    d.add_edge("fb_ctrl", "sum_ctrl", label="u_fb")
    d.add_edge("ff_ctrl", "sum_ctrl", label="u_ff", to_port='n')
    d.add_edge("sum_ctrl", "plant", label="u")
    d.add_feedback_path("plant", "sum_err", label="y")
    
    d.add_invisible_node("out")
    d.add_edge("plant", "out", label="y(t)")
    return d
