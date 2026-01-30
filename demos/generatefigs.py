#!/usr/bin/env python3
"""
Block Diagram Generator for Research Papers using Graphviz

This script creates publication-quality block diagrams (control systems,
signal flow, etc.) with customizable styling. Outputs to PDF, PNG, or SVG.

Features:
- Easy-to-edit block definitions
- Professional styling for research papers
- Support for signal labels, summing junctions, and disturbances
- Configurable fonts, colors, and layout

Usage:
    python generatefigs.py                    # Generate default example
    python generatefigs.py --output my_fig    # Custom output name
    python generatefigs.py --format png       # Change output format

Requirements:
    pip install graphviz

Note: Graphviz must be installed on your system:
    - Windows: https://graphviz.org/download/
    - Linux: sudo apt-get install graphviz
    - macOS: brew install graphviz

Author: MicroPrecisionGimbal Project
Date: January 2026
"""

from graphviz import Digraph
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import argparse


# =============================================================================
# STYLING CONFIGURATION
# =============================================================================

class DiagramStyle:
    """Centralized styling for publication-quality diagrams."""
    
    # Font settings (LaTeX-compatible)
    FONT_NAME = 'Times-Roman'
    FONT_SIZE = '14'
    LABEL_FONT_SIZE = '12'
    
    # Colors (professional palette)
    BLOCK_COLOR = '#FFE5CC'         # Warm beige
    BLOCK_BORDER = '#CC8800'        # Dark orange
    SUMMING_COLOR = '#FFFFFF'       # White
    SUMMING_BORDER = '#000000'      # Black
    SIGNAL_COLOR = '#000000'        # Black
    DISTURBANCE_COLOR = '#666666'   # Gray
    
    # Block styling
    BLOCK_STYLE = 'rounded,filled'
    BLOCK_PENWIDTH = '2.0'
    
    # Arrow styling
    ARROW_STYLE = 'normal'
    ARROW_SIZE = '0.8'
    
    # Layout
    RANKDIR = 'LR'  # Left to Right (change to 'TB' for Top to Bottom)
    RANKSEP = '0.8'
    NODESEP = '0.6'
    
    # PDF settings
    DPI = '300'


# =============================================================================
# BLOCK DIAGRAM CLASS
# =============================================================================

class BlockDiagram:
    """
    Professional block diagram generator for control systems.
    
    Example Usage:
    --------------
    diagram = BlockDiagram("my_diagram", "Control System Block Diagram")
    
    # Add blocks
    diagram.add_block("controller", "PID\nController")
    diagram.add_block("plant", "Plant\nG(s)")
    
    # Add connections
    diagram.add_edge("controller", "plant", label="u(t)")
    
    # Generate output
    diagram.render(format='pdf')
    """
    
    def __init__(self, filename: str = "block_diagram", 
                 title: Optional[str] = None,
                 style: DiagramStyle = None):
        """
        Initialize block diagram.
        
        Parameters
        ----------
        filename : str
            Output filename (without extension)
        title : str, optional
            Diagram title (not rendered in graph, for reference)
        style : DiagramStyle, optional
            Custom styling (uses default if None)
        """
        self.filename = filename
        self.title = title or filename
        self.style = style or DiagramStyle()
        
        # Create Graphviz Digraph
        self.graph = Digraph(
            name=filename,
            comment=self.title,
            format='pdf',
            engine='dot'
        )
        
        # Set global graph attributes
        self.graph.attr(
            rankdir=self.style.RANKDIR,
            ranksep=self.style.RANKSEP,
            nodesep=self.style.NODESEP,
            fontname=self.style.FONT_NAME,
            fontsize=self.style.FONT_SIZE,
            dpi=self.style.DPI,
            splines='polyline',  # Orthogonal routing
            bgcolor='transparent'
        )
        
        # Set default node attributes
        self.graph.attr(
            'node',
            fontname=self.style.FONT_NAME,
            fontsize=self.style.FONT_SIZE,
            penwidth=self.style.BLOCK_PENWIDTH
        )
        
        # Set default edge attributes
        self.graph.attr(
            'edge',
            fontname=self.style.FONT_NAME,
            fontsize=self.style.LABEL_FONT_SIZE,
            arrowhead=self.style.ARROW_STYLE,
            arrowsize=self.style.ARROW_SIZE,
            color=self.style.SIGNAL_COLOR
        )
        
        # Track nodes for validation
        self.nodes = set()
        self.invisible_nodes = set()
    
    def add_block(self, node_id: str, label: str, 
                  shape: str = 'box',
                  width: Optional[float] = None,
                  height: Optional[float] = None,
                  color: Optional[str] = None,
                  **kwargs) -> None:
        """
        Add a rectangular block to the diagram.
        
        Parameters
        ----------
        node_id : str
            Unique identifier for the node
        label : str
            Display text (use \\n for multi-line)
        shape : str
            Node shape ('box', 'ellipse', 'circle', 'diamond', etc.)
        width : float, optional
            Node width in inches
        height : float, optional
            Node height in inches
        color : str, optional
            Fill color (hex or name)
        **kwargs : dict
            Additional Graphviz node attributes
        """
        attrs = {
            'label': label,
            'shape': shape,
            'style': self.style.BLOCK_STYLE,
            'fillcolor': color or self.style.BLOCK_COLOR,
            'color': self.style.BLOCK_BORDER
        }
        
        if width:
            attrs['width'] = str(width)
        if height:
            attrs['height'] = str(height)
        
        attrs.update(kwargs)
        self.graph.node(node_id, **attrs)
        self.nodes.add(node_id)
    
    def add_summing_junction(self, node_id: str, 
                             inputs: List[str] = ['+', '+'],
                             size: float = 0.4) -> None:
        """
        Add a summing junction (circle with + and - signs).
        
        Parameters
        ----------
        node_id : str
            Unique identifier
        inputs : List[str]
            Signs for each input (e.g., ['+', '-', '+'])
        size : float
            Circle diameter in inches
        """
        # Create label with signs
        label = '∑'  # Summation symbol
        
        self.graph.node(
            node_id,
            label=label,
            shape='circle',
            style='filled',
            fillcolor=self.style.SUMMING_COLOR,
            color=self.style.SUMMING_BORDER,
            width=str(size),
            height=str(size),
            fixedsize='true',
            fontsize='16'
        )
        self.nodes.add(node_id)
    
    def add_disturbance_input(self, node_id: str, label: str,
                              target: str, target_port: str = 'n') -> None:
        """
        Add an external disturbance/input arrow entering from above.
        
        Parameters
        ----------
        node_id : str
            Unique identifier for the disturbance label
        label : str
            Disturbance name (e.g., "Uncertainty", "Noise")
        target : str
            Node ID where disturbance enters
        target_port : str
            Port on target ('n'=north, 's'=south, 'e'=east, 'w'=west)
        """
        # Create invisible node for label positioning
        self.graph.node(
            node_id,
            label=label,
            shape='plaintext',
            fontsize=self.style.LABEL_FONT_SIZE
        )
        self.invisible_nodes.add(node_id)
        
        # Add arrow to target
        self.graph.edge(
            node_id,
            f"{target}:{target_port}",
            arrowhead='normal',
            color=self.style.DISTURBANCE_COLOR,
            style='solid'
        )
    
    def add_edge(self, from_node: str, to_node: str,
                 label: Optional[str] = None,
                 from_port: Optional[str] = None,
                 to_port: Optional[str] = None,
                 style: str = 'solid',
                 **kwargs) -> None:
        """
        Add a directed edge (signal connection) between nodes.
        
        Parameters
        ----------
        from_node : str
            Source node ID
        to_node : str
            Destination node ID
        label : str, optional
            Signal label (e.g., "u(t)", "y(t)")
        from_port : str, optional
            Port on source ('e', 'w', 'n', 's')
        to_port : str, optional
            Port on destination
        style : str
            Line style ('solid', 'dashed', 'dotted')
        **kwargs : dict
            Additional edge attributes
        """
        # Build port specifications
        src = f"{from_node}:{from_port}" if from_port else from_node
        dst = f"{to_node}:{to_port}" if to_port else to_node
        
        attrs = {'style': style}
        if label:
            attrs['label'] = f"  {label}  "  # Add spacing for readability
        
        attrs.update(kwargs)
        self.graph.edge(src, dst, **attrs)
    
    def add_feedback_path(self, from_node: str, to_node: str,
                          label: Optional[str] = None,
                          constraint: bool = False) -> None:
        """
        Add a feedback connection (typically drawn differently).
        
        Parameters
        ----------
        from_node : str
            Source node ID
        to_node : str
            Destination node ID
        label : str, optional
            Signal label
        constraint : bool
            If False, edge doesn't affect layout
        """
        attrs = {
            'style': 'solid',
            'constraint': 'false' if not constraint else 'true',
            'color': self.style.SIGNAL_COLOR
        }
        if label:
            attrs['label'] = f"  {label}  "
        
        self.graph.edge(from_node, to_node, **attrs)
    
    def add_invisible_node(self, node_id: str) -> None:
        """
        Add an invisible node for layout control (routing connections).
        
        Parameters
        ----------
        node_id : str
            Unique identifier
        """
        self.graph.node(
            node_id,
            label='',
            shape='point',
            width='0.01',
            height='0.01',
            style='invis'
        )
        self.invisible_nodes.add(node_id)
    
    def set_rank_same(self, node_ids: List[str]) -> None:
        """
        Force nodes to appear at the same vertical level.
        
        Parameters
        ----------
        node_ids : List[str]
            List of node IDs to align
        """
        with self.graph.subgraph() as s:
            s.attr(rank='same')
            for node_id in node_ids:
                s.node(node_id)
    
    def render(self, output_dir: str = '.', format: str = 'pdf',
               view: bool = False, cleanup: bool = True) -> Path:
        """
        Render the diagram to file.
        
        Parameters
        ----------
        output_dir : str
            Output directory
        format : str
            Output format ('pdf', 'png', 'svg', 'eps')
        view : bool
            If True, open the file after rendering
        cleanup : bool
            If True, delete intermediate .gv file
        
        Returns
        -------
        Path
            Path to generated file
        """
        from graphviz.backend.execute import ExecutableNotFound
        
        self.graph.format = format
        output_path = Path(output_dir) / self.filename
        
        try:
            # Render
            result = self.graph.render(
                filename=str(output_path),
                view=view,
                cleanup=cleanup
            )
            return Path(result)
        except ExecutableNotFound as e:
            # Graphviz not installed - save source only
            print(f"  ⚠ Graphviz not installed. Saving source file only.")
            print(f"    Install from: https://graphviz.org/download/")
            return self.save_source(output_dir)
    
    def save_source(self, output_dir: str = '.') -> Path:
        """
        Save the Graphviz source code (.gv file) for manual editing.
        
        Returns
        -------
        Path
            Path to .gv file
        """
        output_path = Path(output_dir) / f"{self.filename}.gv"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.graph.source)
        return output_path


# =============================================================================
# EXAMPLE DIAGRAMS
# =============================================================================

def create_robot_manipulator_diagram(filename: str = "robot_control_diagram") -> BlockDiagram:
    """
    Create the robot manipulator control diagram from the user's image.
    
    This diagram shows:
    - Desired trajectories input
    - Tracking error computation
    - Computed torque control
    - Robot manipulator with uncertainties
    - Extended state observer
    - Feedback loops
    """
    diagram = BlockDiagram(filename, "Robot Manipulator Control System")
    
    # Add main blocks
    diagram.add_block(
        "trajectories", 
        "Desired\nTrajectories",
        width=1.5, 
        height=0.8
    )
    
    diagram.add_block(
        "errors",
        "Tracking\nErrors",
        width=1.5,
        height=0.8
    )
    
    diagram.add_block(
        "controller",
        "Computed Torque Control",
        width=2.5,
        height=0.8
    )
    
    diagram.add_block(
        "robot",
        "Robot Manipulator",
        width=2.0,
        height=1.0
    )
    
    diagram.add_block(
        "observer",
        "Extended State\nObserver",
        width=2.0,
        height=0.8
    )
    
    # Add summing junction
    diagram.add_summing_junction("sum1", inputs=['+', '+'], size=0.4)
    
    # Forward path connections
    diagram.add_edge("trajectories", "errors")
    diagram.add_edge("errors", "controller", label="z₁, z₂")
    diagram.add_edge("controller", "sum1", label="u₀", from_port='e')
    diagram.add_edge("sum1", "robot", label="u(t)", from_port='e')
    diagram.add_edge("robot", "observer", label="x₁", to_port='s')
    
    # Feedback paths
    diagram.add_feedback_path("robot", "errors", label="x₁")
    diagram.add_feedback_path("observer", "errors", label="x̂₂")
    diagram.add_feedback_path("observer", "sum1", label="uᶜ", constraint=False)
    
    # Output signal
    diagram.add_invisible_node("output")
    diagram.add_edge("robot", "output", label="x₁", from_port='e')
    
    # Disturbance inputs
    diagram.add_disturbance_input("disturbance1", "Uncertainty", "robot", 'n')
    diagram.add_disturbance_input("disturbance2", "Unknown Fault", "robot", 'n')
    
    return diagram


def create_feedback_linearization_diagram(filename: str = "fbl_diagram") -> BlockDiagram:
    """
    Create a simplified feedback linearization block diagram.
    """
    diagram = BlockDiagram(filename, "Feedback Linearization Control")
    
    # Blocks
    diagram.add_block("ref", "Reference\nr(t)", width=1.2, height=0.8)
    diagram.add_block("controller", "Outer Loop\nPID", width=1.5, height=0.8)
    diagram.add_block("fbl", "Feedback\nLinearization", width=2.0, height=0.8)
    diagram.add_block("plant", "Nonlinear\nPlant", width=1.5, height=0.8)
    
    # Summing junction for error
    diagram.add_summing_junction("sum_err", inputs=['+', '-'], size=0.4)
    
    # Forward path
    diagram.add_edge("ref", "sum_err", label="r")
    diagram.add_edge("sum_err", "controller", label="e", from_port='e')
    diagram.add_edge("controller", "fbl", label="v")
    diagram.add_edge("fbl", "plant", label="τ")
    
    # Feedback
    diagram.add_feedback_path("plant", "sum_err", label="y")
    
    # Output
    diagram.add_invisible_node("output")
    diagram.add_edge("plant", "output", label="y(t)", from_port='e')
    
    # Disturbance
    diagram.add_disturbance_input("dist", "Disturbance d", "plant", 'n')
    
    return diagram


def create_ndob_diagram(filename: str = "ndob_diagram") -> BlockDiagram:
    """
    Create NDOB (Nonlinear Disturbance Observer) block diagram.
    """
    diagram = BlockDiagram(filename, "NDOB Control Architecture")
    
    # Main blocks
    diagram.add_block("ref", "r(t)", width=1.0, height=0.6)
    diagram.add_block("controller", "Controller\nC(s)", width=1.5, height=0.8)
    diagram.add_block("plant", "Plant\nP(s)", width=1.5, height=0.8)
    diagram.add_block("ndob", "NDOB\nλ/(s+λ)", width=1.5, height=0.8)
    
    # Summing junctions
    diagram.add_summing_junction("sum_err", size=0.4)
    diagram.add_summing_junction("sum_disturbance", size=0.4)
    diagram.add_summing_junction("sum_comp", size=0.4)
    
    # Forward path
    diagram.add_edge("ref", "sum_err")
    diagram.add_edge("sum_err", "controller", label="e")
    diagram.add_edge("controller", "sum_comp", label="u₀")
    diagram.add_edge("sum_comp", "sum_disturbance", label="u")
    diagram.add_edge("sum_disturbance", "plant", label="u+d")
    
    # NDOB path
    diagram.add_edge("sum_disturbance", "ndob", label="u+d", to_port='s')
    diagram.add_feedback_path("ndob", "sum_comp", label="d̂", constraint=False)
    
    # Feedback
    diagram.add_feedback_path("plant", "sum_err", label="y")
    
    # Disturbance input
    diagram.add_disturbance_input("dist", "d", "sum_disturbance", 'n')
    
    # Output
    diagram.add_invisible_node("output")
    diagram.add_edge("plant", "output", label="y(t)")
    
    return diagram


def create_lasercom_gimbal_diagram(filename: str = "lasercom_gimbal_system") -> BlockDiagram:
    """
    Create complete digital twin block diagram for 2-DOF gimbal with:
    - Extended Kalman Filter (EKF) state estimation
    - Nonlinear Disturbance Observer (NDOB)
    - Feedback Linearization (FBL) control
    - Fast Steering Mirror (FSM) hierarchical control
    - Multi-rate execution (sensors, estimator, controllers, actuators)
    
    This matches the simulation_runner.py architecture.
    """
    diagram = BlockDiagram(filename, "Lasercom Gimbal Digital Twin")
    
    # Reference/Target Generation
    diagram.add_block("target_gen", "Target\nGeneration", width=1.5, height=0.8)
    
    # Summing junction for tracking error
    diagram.add_summing_junction("sum_track_err", inputs=['+', '-'], size=0.4)
    
    # Coarse Controller (FBL or PID)
    diagram.add_block("fbl_ctrl", "Feedback\nLinearization\n+ NDOB", width=2.0, height=1.2)
    
    # Summing junction for motor voltage
    diagram.add_summing_junction("sum_voltage", inputs=['+', '+'], size=0.4)
    
    # Actuators
    diagram.add_block("motors", "Gimbal Motors\n(Az, El)", width=1.8, height=0.8)
    diagram.add_block("fsm_act", "FSM\nActuator", width=1.5, height=0.8)
    
    # Plant/Dynamics
    diagram.add_block("gimbal_dyn", "2-DOF Gimbal\nDynamics\nM(q)q̈ + C + G", 
                      width=2.2, height=1.2)
    
    # Sensors
    diagram.add_block("sensors", "Sensors\n(Encoders, Gyros,\nQPD)", 
                      width=1.8, height=1.0)
    
    # Extended Kalman Filter
    diagram.add_block("ekf", "Extended\nKalman Filter\n(EKF)", width=1.8, height=1.0)
    
    # FSM Controller
    diagram.add_block("fsm_ctrl", "FSM\nPIDF\nController", width=1.5, height=0.8)
    
    # Optics
    diagram.add_block("optics", "Optical\nChain", width=1.3, height=0.7)
    
    # ========================================================================
    # FORWARD PATH - Coarse Loop
    # ========================================================================
    diagram.add_edge("target_gen", "sum_track_err", label="θ_ref")
    diagram.add_edge("sum_track_err", "fbl_ctrl", label="e_track", from_port='e')
    diagram.add_edge("fbl_ctrl", "sum_voltage", label="τ_cmd", from_port='e')
    diagram.add_edge("sum_voltage", "motors", label="V_motor", from_port='e')
    diagram.add_edge("motors", "gimbal_dyn", label="τ_motor")
    
    # ========================================================================
    # SENSING PATH
    # ========================================================================
    diagram.add_edge("gimbal_dyn", "sensors", label="q, q̇", to_port='w')
    diagram.add_edge("sensors", "ekf", label="z_meas\n(enc, gyro, QPD)")
    
    # ========================================================================
    # ESTIMATION FEEDBACK
    # ========================================================================
    diagram.add_feedback_path("ekf", "sum_track_err", label="q̂, q̂̇", constraint=False)
    diagram.add_feedback_path("ekf", "fbl_ctrl", label="state est", constraint=False)
    
    # ========================================================================
    # FSM FINE LOOP (Hierarchical Control)
    # ========================================================================
    # FSM residual error (coarse pointing error)
    diagram.add_feedback_path("ekf", "fsm_ctrl", label="e_fine", constraint=False)
    diagram.add_edge("fsm_ctrl", "fsm_act", label="α_cmd")
    diagram.add_edge("fsm_act", "optics", label="α_fsm")
    
    # ========================================================================
    # OUTPUT
    # ========================================================================
    diagram.add_invisible_node("output_los")
    diagram.add_edge("optics", "output_los", label="LOS", from_port='e')
    
    # ========================================================================
    # DISTURBANCES
    # ========================================================================
    diagram.add_disturbance_input("dist_friction", "Friction\nModel Mismatch", "gimbal_dyn", 'n')
    diagram.add_disturbance_input("dist_vibration", "Base\nVibration", "gimbal_dyn", 'n')
    diagram.add_disturbance_input("dist_sensor", "Sensor\nNoise", "sensors", 'n')
    
    # ========================================================================
    # ANNOTATIONS (MULTI-RATE EXECUTION)
    # ========================================================================
    # Note: Graphviz doesn't support easy text annotations, but we can add
    # invisible nodes with plaintext for labeling execution rates
    
    return diagram


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Generate block diagrams for research papers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generatefigs.py                              # Generate all examples
  python generatefigs.py --diagram robot --view       # View robot diagram
  python generatefigs.py --format png --output mydiag # Custom format & name
  python generatefigs.py --list-diagrams              # Show available diagrams
        """
    )
    
    parser.add_argument(
        '--diagram',
        type=str,
        default='gimbal',
        choices=['all', 'robot', 'fbl', 'ndob', 'gimbal'],
        help='Diagram to generate (default: gimbal)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output filename (without extension)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        default='pdf',
        choices=['pdf', 'png', 'svg', 'eps'],
        help='Output format (default: pdf)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='figures_diagrams',
        help='Output directory (default: figures_diagrams)'
    )
    
    parser.add_argument(
        '--view',
        action='store_true',
        help='Open the generated file after creation'
    )
    
    parser.add_argument(
        '--save-source',
        action='store_true',
        help='Save Graphviz source (.gv) for manual editing'
    )
    
    parser.add_argument(
        '--list-diagrams',
        action='store_true',
        help='List available diagram templates'
    )
    
    args = parser.parse_args()
    
    # List available diagrams
    if args.list_diagrams:
        print("Available diagram templates:")
        print("  robot  - Robot manipulator with extended state observer")
        print("  fbl    - Feedback linearization control")
        print("  ndob   - Nonlinear disturbance observer (NDOB)")
        print("  gimbal - Complete lasercom 2-DOF gimbal system (EKF + NDOB + FBL + FSM)")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("BLOCK DIAGRAM GENERATOR")
    print("=" * 70)
    print(f"\nOutput format: {args.format.upper()}")
    print(f"Output directory: {output_dir.absolute()}\n")
    
    # Generate diagrams
    diagrams_to_generate = []
    
    if args.diagram == 'all':
        diagrams_to_generate = [
            ('robot', 'robot_control_diagram'),
            ('fbl', 'feedback_linearization_diagram'),
            ('ndob', 'ndob_diagram'),
            ('gimbal', 'lasercom_gimbal_system')
        ]
    else:
        # Map diagram choice to function and default filename
        diagram_map = {
            'robot': ('robot', 'robot_control_diagram'),
            'fbl': ('fbl', 'feedback_linearization_diagram'),
            'ndob': ('ndob', 'ndob_diagram'),
            'gimbal': ('gimbal', 'lasercom_gimbal_system')
        }
        diagrams_to_generate = [diagram_map[args.diagram]]
    
    # Generate each diagram
    generated_files = []
    for diagram_type, default_filename in diagrams_to_generate:
        filename = args.output if args.output and len(diagrams_to_generate) == 1 else default_filename
        
        print(f"Generating {diagram_type} diagram...")
        
        # Create diagram
        if diagram_type == 'robot':
            diagram = create_robot_manipulator_diagram(filename)
        elif diagram_type == 'fbl':
            diagram = create_feedback_linearization_diagram(filename)
        elif diagram_type == 'ndob':
            diagram = create_ndob_diagram(filename)
        elif diagram_type == 'gimbal':
            diagram = create_lasercom_gimbal_diagram(filename)
        
        # Render
        output_file = diagram.render(
            output_dir=str(output_dir),
            format=args.format,
            view=args.view and len(diagrams_to_generate) == 1,
            cleanup=not args.save_source
        )
        generated_files.append(output_file)
        print(f"  ✓ Saved: {output_file}")
        
        # Save source if requested
        if args.save_source:
            source_file = diagram.save_source(output_dir=str(output_dir))
            print(f"  ✓ Source: {source_file}")
    
    print("\n" + "=" * 70)
    print(f"COMPLETE: Generated {len(generated_files)} diagram(s)")
    print("=" * 70)
    print("\nTo customize diagrams:")
    print("  1. Edit the diagram creation functions in this script")
    print("  2. Or use --save-source to get .gv files for manual editing")
    print("  3. Modify block labels, add/remove connections, adjust styling")
    print("\nFor more control system diagrams, add new functions following")
    print("the patterns in create_robot_manipulator_diagram() and similar.")


if __name__ == '__main__':
    main()
