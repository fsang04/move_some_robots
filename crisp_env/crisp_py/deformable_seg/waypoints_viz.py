import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R

# ── Raw Data ───────────────────────────────────────────────────────────────────
data = {"parameterized_poses":[{"name":"","pose_with_joint_angles":{"joint_angles":[0.01262837204989315,-0.7764109735949388,0.003070895050833101,-2.355196871819365,-0.0068000003614738305,1.5625538528817542,0.7658549699762368],"pose":[0.999278215918469,0.03426198104660694,-0.016405170435633318,0,0.03440686992608244,-0.9993706078430948,0.008632569297219927,0,-0.016099076775431632,-0.009190789325999843,-0.999828159794744,0,0.30669286453753924,0.003434570679094606,0.4840312734380403,1]}},{"name":"","pose_with_joint_angles":{"joint_angles":[0.2154161139672473,-0.2553900307594312,0.499983876051962,-2.096926423590492,0.06200018829457489,2.0089549144851295,1.3600665482915137],"pose":[0.9826119697783653,0.10967983650208663,0.14981327072263856,0,0.1143635396115063,-0.9931729402798433,-0.02298819860007155,0,0.1462691497191894,0.039721656388304716,-0.988447027338438,0,0.38150753920722164,0.34989248285715385,0.4266178422748894,1]}},{"name":"","pose_with_joint_angles":{"joint_angles":[0.02017161143715922,0.15203287820583458,0.3604405947228797,-1.6922050071036965,0.0361467628406207,2.1657518093033685,1.0156964922432248],"pose":[0.9511450841683304,0.0925116730918098,0.2945582879085767,0,0.1444307099500987,-0.9765473807497342,-0.15967137797865505,0,0.2728786675538053,0.19441391554560733,-0.9421997995310152,0,0.6240748173999493,0.24545308200630148,0.4435580985660357,1]}},{"name":"","pose_with_joint_angles":{"joint_angles":[0.0570404500019882,-0.6179578732596793,0.4165303939517411,-1.707610900889217,0.027141740450798646,1.384775786553275,1.069381255334614],"pose":[0.9411674794687765,0.17011228135601258,0.292002659394912,0,0.1358502354470934,-0.9816253653019613,0.13400119960738585,0,0.3094324775587519,-0.08644894420932088,-0.9469836967323916,0,0.2834569857401983,0.22073879670887928,0.688078842095759,1]}},{"name":"","pose_with_joint_angles":{"joint_angles":[0.17499616440898771,-0.6438918739652649,0.7849135542982598,-2.2722087743899393,0.3722597724089359,1.9142065506128878,1.5356526560626502],"pose":[0.9870057081237735,0.022928757649505083,0.15904078085571952,0,0.030928411396721034,-0.9983679986454782,-0.048007691264588453,0,0.15768047476479352,0.05230274580140079,-0.9861040972734996,0,0.20357281325354337,0.39968084812086346,0.42594632098074503,1]}},{"name":"","pose_with_joint_angles":{"joint_angles":[-0.04665925066001059,-0.6985046215822632,0.19557626273273615,-2.273534540280164,0.19612461687326388,1.8958505993364083,0.8236960751839236],"pose":[0.9523773630627763,0.03657528326748749,0.3027202879738225,0,0.06438547736920314,-0.9945172268849853,-0.08240122271178131,0,0.29804670345698353,0.0979678528032289,-0.9495106436346828,0,0.38799101322079477,0.09934953945738548,0.5309348408301401,1]}}]}

# ── Parse 4x4 Homogeneous Matrix (column-major) ────────────────────────────────
def parse_pose(pose_flat):
    return np.array(pose_flat).reshape(4, 4, order='F')

# ── Extract Waypoints ──────────────────────────────────────────────────────────
waypoints = []
for i, entry in enumerate(data["parameterized_poses"]):
    pwja   = entry["pose_with_joint_angles"]
    mat    = parse_pose(pwja["pose"])
    joints = np.degrees(pwja["joint_angles"])
    pos    = mat[:3, 3]
    rot    = mat[:3, :3]
    euler  = R.from_matrix(rot).as_euler('xyz', degrees=True)
    quat   = R.from_matrix(rot).as_quat()

    waypoints.append({
        "id"    : i + 1,
        "pos"   : pos,
        "rot"   : rot,
        "euler" : euler,
        "quat"  : quat,
        "joints": joints,
    })

positions  = np.array([wp["pos"] for wp in waypoints])
robot_base = np.array([0.0, 0.0, 0.0])
axis_len   = 0.06   # 6 cm orientation arrows

# ── Plotly Color Scale (one per waypoint) ──────────────────────────────────────
import plotly.express as px
palette = px.colors.sample_colorscale("Plasma", [i/(len(waypoints)-1) for i in range(len(waypoints))])

traces = []

# ══════════════════════════════════════════════════════════════════════════════
# 1. Trajectory path line
# ══════════════════════════════════════════════════════════════════════════════
traces.append(go.Scatter3d(
    x=positions[:, 0],
    y=positions[:, 1],
    z=positions[:, 2],
    mode='lines',
    line=dict(color='lightgray', width=4, dash='dash'),
    name='EE Path',
    hoverinfo='skip',
))

# ── Base → first waypoint connector ───────────────────────────────────────────
traces.append(go.Scatter3d(
    x=[robot_base[0], positions[0, 0]],
    y=[robot_base[1], positions[0, 1]],
    z=[robot_base[2], positions[0, 2]],
    mode='lines',
    line=dict(color='gray', width=2, dash='dot'),
    name='Base→P1',
    hoverinfo='skip',
))

# ══════════════════════════════════════════════════════════════════════════════
# 2. Waypoint spheres with hover info
# ══════════════════════════════════════════════════════════════════════════════
hover_texts = []
for wp in waypoints:
    p = wp["pos"]
    e = wp["euler"]
    q = wp["quat"]
    j = wp["joints"]
    txt = (
        f"<b>Pose {wp['id']}</b><br>"
        f"──────────────────<br>"
        f"X: {p[0]:.4f} m<br>"
        f"Y: {p[1]:.4f} m<br>"
        f"Z: {p[2]:.4f} m<br>"
        f"──────────────────<br>"
        f"Roll : {e[0]:.2f}°<br>"
        f"Pitch: {e[1]:.2f}°<br>"
        f"Yaw  : {e[2]:.2f}°<br>"
        f"──────────────────<br>"
        f"qx: {q[0]:.4f}<br>"
        f"qy: {q[1]:.4f}<br>"
        f"qz: {q[2]:.4f}<br>"
        f"qw: {q[3]:.4f}<br>"
        f"──────────────────<br>"
        f"J1: {j[0]:.2f}°  J2: {j[1]:.2f}°<br>"
        f"J3: {j[2]:.2f}°  J4: {j[3]:.2f}°<br>"
        f"J5: {j[4]:.2f}°  J6: {j[5]:.2f}°<br>"
        f"J7: {j[6]:.2f}°"
    )
    hover_texts.append(txt)

traces.append(go.Scatter3d(
    x=positions[:, 0],
    y=positions[:, 1],
    z=positions[:, 2],
    mode='markers+text',
    marker=dict(
        size=10,
        color=[wp["id"] for wp in waypoints],
        colorscale='Plasma',
        colorbar=dict(
            title=dict(text='Waypoint', side='right'),
            tickvals=list(range(1, len(waypoints)+1)),
            ticktext=[f"P{i}" for i in range(1, len(waypoints)+1)],
            thickness=15,
            len=0.5,
        ),
        line=dict(color='white', width=1.5),
        symbol='circle',
    ),
    text=[f"P{wp['id']}" for wp in waypoints],
    textposition='top center',
    textfont=dict(size=12, color='white', family='Arial Black'),
    hovertext=hover_texts,
    hoverinfo='text',
    name='Waypoints',
))

# ══════════════════════════════════════════════════════════════════════════════
# 3. Orientation axes as cone arrows (X=red, Y=green, Z=blue)
# ══════════════════════════════════════════════════════════════════════════════
axis_cfg = [
    ("X-axis", 0, "red"),
    ("Y-axis", 1, "green"),
    ("Z-axis", 2, "royalblue"),
]

for axis_name, axis_idx, color in axis_cfg:
    # Shaft lines
    shaft_x, shaft_y, shaft_z = [], [], []
    cone_x, cone_y, cone_z = [], [], []
    cone_u, cone_v, cone_w = [], [], []

    for wp in waypoints:
        p   = wp["pos"]
        vec = wp["rot"][:, axis_idx] * axis_len

        # Draw shaft as line
        shaft_x += [p[0], p[0]+vec[0], None]
        shaft_y += [p[1], p[1]+vec[1], None]
        shaft_z += [p[2], p[2]+vec[2], None]

        # Cone arrowhead position and direction
        cone_x.append(p[0] + vec[0])
        cone_y.append(p[1] + vec[1])
        cone_z.append(p[2] + vec[2])
        cone_u.append(vec[0])
        cone_v.append(vec[1])
        cone_w.append(vec[2])

    traces.append(go.Scatter3d(
        x=shaft_x, y=shaft_y, z=shaft_z,
        mode='lines',
        line=dict(color=color, width=2),
        name=axis_name,
        hoverinfo='skip',
        legendgroup=axis_name,
    ))

    traces.append(go.Cone(
        x=cone_x, y=cone_y, z=cone_z,
        u=cone_u, v=cone_v, w=cone_w,
        colorscale=[[0, color], [1, color]],
        showscale=False,
        sizemode='absolute',
        sizeref=0.005,
        anchor='tip',
        hoverinfo='skip',
        name=f'{axis_name} tip',
        legendgroup=axis_name,
        showlegend=False,
    ))

# ══════════════════════════════════════════════════════════════════════════════
# 4. Robot Base marker
# ══════════════════════════════════════════════════════════════════════════════
traces.append(go.Scatter3d(
    x=[robot_base[0]],
    y=[robot_base[1]],
    z=[robot_base[2]],
    mode='markers+text',
    marker=dict(size=14, color='white', symbol='diamond',
                line=dict(color='black', width=2)),
    text=['BASE'],
    textposition='top center',
    textfont=dict(size=12, color='white', family='Arial Black'),
    hovertext='<b>Robot Base</b><br>X:0  Y:0  Z:0',
    hoverinfo='text',
    name='Robot Base',
))

# ══════════════════════════════════════════════════════════════════════════════
# 5. Layout
# ══════════════════════════════════════════════════════════════════════════════
layout = go.Layout(
    title=dict(
        text='<b>Franka Panda — End-Effector Trajectory</b><br>'
             '<sup>Axes: <span style="color:red">■ X</span>  '
             '<span style="color:lime">■ Y</span>  '
             '<span style="color:royalblue">■ Z</span>  '
             '| Hover waypoints for full info</sup>',
        x=0.5,
        font=dict(size=18, color='white'),
    ),
    paper_bgcolor='#0d0d0d',
    scene=dict(
        bgcolor='#111111',
        xaxis=dict(title='X [m]', showgrid=True, gridcolor='#333',
                   zeroline=True, zerolinecolor='#555', color='white'),
        yaxis=dict(title='Y [m]', showgrid=True, gridcolor='#333',
                   zeroline=True, zerolinecolor='#555', color='white'),
        zaxis=dict(title='Z [m]', showgrid=True, gridcolor='#333',
                   zeroline=True, zerolinecolor='#555', color='white'),
        aspectmode='cube',
        camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
    ),
    legend=dict(
        font=dict(color='white', size=11),
        bgcolor='rgba(30,30,30,0.8)',
        bordercolor='gray',
        borderwidth=1,
        x=0.01, y=0.99,
    ),
    margin=dict(l=0, r=0, t=80, b=0),
    hoverlabel=dict(
        bgcolor='#1a1a2e',
        bordercolor='gray',
        font=dict(color='white', size=12, family='monospace'),
    ),
)

fig = go.Figure(data=traces, layout=layout)

# ══════════════════════════════════════════════════════════════════════════════
# 6. Save as HTML
# ══════════════════════════════════════════════════════════════════════════════
fig.write_html(
    "traj_viz.html",
    include_plotlyjs=True,       # fully self-contained HTML
    full_html=True,
    config={
        'displayModeBar': True,
        'scrollZoom'    : True,
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'franka_traj',
            'height' : 900,
            'width'  : 1400,
            'scale'  : 2,
        },
    },
)

print("✅ Saved → traj_viz.html")
print(f"   Waypoints  : {len(waypoints)}")
print(f"   Axis length: {axis_len*100:.0f} cm")
print("   Open traj_viz.html in any browser — fully interactive!")

# ── Print Waypoint Index Summary ───────────────────────────────────────────────
print("\n" + "═" * 70)
print("WAYPOINT INDEX SUMMARY")
print("═" * 70)
print(f"{'Index':<6} {'X (m)':<12} {'Y (m)':<12} {'Z (m)':<12} {'Roll (°)':<10} {'Pitch (°)':<10} {'Yaw (°)':<10}")
print("-" * 70)
for wp in waypoints:
    p = wp["pos"]
    e = wp["euler"]
    print(f"P{wp['id']:<5} {p[0]:<12.4f} {p[1]:<12.4f} {p[2]:<12.4f} {e[0]:<10.2f} {e[1]:<10.2f} {e[2]:<10.2f}")
print("═" * 70)