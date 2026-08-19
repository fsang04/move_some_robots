"""
3D visualization of shape matching data using Plotly.
Visualizes wire shape at multiple time indices with different colors.
"""

import pickle
import numpy as np
import plotly.graph_objects as go


# Load data
pkl_path = "/home/yehengz/deformable_seg/data/shape_matching/Dec20_shapeMatching_panda3.pkl"
output_path = "/home/yehengz/deformable_seg/data/shape_matching/shape_matching_3d_viz1220_panda3.html"

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

bdlo_data = np.array(data)
print(f"Raw shape: {bdlo_data.shape}")
bdlo_data = bdlo_data.squeeze()
print(f"After squeeze: {bdlo_data.shape}")
bdlo_data = bdlo_data.T.reshape(-1, 20, 3)
print(f"After reshape: {bdlo_data.shape}")

# Time indices to visualize
# time_indices = [0, 1500, 3300, 4750, 7200, 9500] # 1220 panda1
# time_indices = [0, 2858, 3790, 5858, 7900, 9585] # 1220 panda2
time_indices = [0, 1458, 3558, 5558, 7058, 8558] # 1220 panda3

# Colors for each time step
colors = [
    'rgb(31, 119, 180)',   # blue
    'rgb(255, 127, 14)',   # orange
    'rgb(44, 160, 44)',    # green
    'rgb(214, 39, 40)',    # red
    'rgb(148, 103, 189)',  # purple
    'rgb(140, 86, 75)',    # brown
]

# Wire connections (1-indexed)
connections = [
    {'points': [1, 2, 3, 4, 5], 'name': 'segment1'},
    {'points': [5, 6, 7, 8, 9], 'name': 'segment2'},
    {'points': [9, 10, 11, 12, 13], 'name': 'segment3'},
    {'points': [5, 14, 15, 16, 17], 'name': 'branch1'},
    {'points': [9, 18, 19, 20], 'name': 'branch2'}
]

# Compute fixed axis limits from selected time indices (1.5x range)
selected_data = []
for t in time_indices:
    if t < len(bdlo_data):
        chunk = bdlo_data[t]
        if not np.any(np.isnan(chunk)) and not np.any(np.isinf(chunk)):
            selected_data.append(chunk)

all_points = np.vstack(selected_data)  # Combine all selected frames
x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()

# Compute center and expand by 1.5x
x_center = (x_max + x_min) / 2
y_center = (y_max + y_min) / 2
z_center = (z_max + z_min) / 2

x_range = (x_max - x_min) * 1.5 / 2
y_range = (y_max - y_min) * 1.5 / 2
z_range = (z_max - z_min) * 1.5 / 2

AXIS_LIMITS = {
    'x': [x_center - x_range, x_center + x_range],
    'y': [y_center - y_range, y_center + y_range],
    'z': [z_center - z_range, z_center + z_range]
}

print(f"\nFixed axis limits (1.5x range):")
print(f"  X: [{AXIS_LIMITS['x'][0]:.3f}, {AXIS_LIMITS['x'][1]:.3f}]")
print(f"  Y: [{AXIS_LIMITS['y'][0]:.3f}, {AXIS_LIMITS['y'][1]:.3f}]")
print(f"  Z: [{AXIS_LIMITS['z'][0]:.3f}, {AXIS_LIMITS['z'][1]:.3f}]")

# Create figure
fig = go.Figure()

for idx, (t, color) in enumerate(zip(time_indices, colors)):
    if t >= len(bdlo_data):
        print(f"Warning: time index {t} out of range (max: {len(bdlo_data)-1})")
        continue
    
    data_chunk = bdlo_data[t]  # (20, 3)
    
    # Check for NaN/Inf
    if np.any(np.isnan(data_chunk)) or np.any(np.isinf(data_chunk)):
        print(f"Warning: time index {t} contains NaN/Inf, skipping")
        continue
    
    # Add scatter points for keypoints
    fig.add_trace(go.Scatter3d(
        x=data_chunk[:, 0],
        y=data_chunk[:, 1],
        z=data_chunk[:, 2],
        mode='markers',
        marker=dict(size=5, color=color),
        name=f't={t}',
        legendgroup=f't={t}',
        showlegend=True
    ))
    
    # Add lines for wire connections
    for conn in connections:
        pts = np.array(conn['points']) - 1  # Convert to 0-indexed
        fig.add_trace(go.Scatter3d(
            x=data_chunk[pts, 0],
            y=data_chunk[pts, 1],
            z=data_chunk[pts, 2],
            mode='lines',
            line=dict(width=4, color=color),
            name=f't={t} {conn["name"]}',
            legendgroup=f't={t}',
            showlegend=False
        ))

# Update layout
fig.update_layout(
    title=dict(
        text='Shape Matching Data - Wire Deformation Over Time',
        x=0.5,
        font=dict(size=20)
    ),
    scene=dict(
        xaxis=dict(title='X', range=AXIS_LIMITS['x']),
        yaxis=dict(title='Y', range=AXIS_LIMITS['y']),
        zaxis=dict(title='Z', range=AXIS_LIMITS['z']),
        aspectmode='cube'
    ),
    legend=dict(
        title='Time Index',
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ),
    width=1200,
    height=800
)

# Save to HTML
fig.write_html(output_path)
print(f"\nVisualization saved to: {output_path}")

# Also show in browser if running interactively
# fig.show()
