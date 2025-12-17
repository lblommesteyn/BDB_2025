"""
Create IMPROVED process diagram showing how DACS works
- Clearer visuals
- Daron Bland as example
- Better flow
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, Rectangle, Wedge
import numpy as np

fig = plt.figure(figsize=(18, 11))
fig.patch.set_facecolor('#f5f7fa')

# Main title
fig.text(0.5, 0.97, 'DACS: Defensive Air Control Score',
         ha='center', fontsize=26, fontweight='bold', color='#2c3e50')
fig.text(0.5, 0.94, 'Quantifying Defensive Pursuit During Ball Flight',
         ha='center', fontsize=15, style='italic', color='#7f8c8d')

# ========== STEP 1: INPUT ==========
ax1 = fig.add_axes([0.05, 0.6, 0.22, 0.28])
ax1.axis('off')

# Title
title1 = FancyBboxPatch((0.05, 0.88), 0.9, 0.10,
                        boxstyle="round,pad=0.02",
                        facecolor='#3498db', edgecolor='white',
                        linewidth=3, transform=ax1.transAxes)
ax1.add_patch(title1)
ax1.text(0.5, 0.93, 'Step 1: Track Players', ha='center', va='center',
         fontsize=14, fontweight='bold', color='white', transform=ax1.transAxes)

# Mini football field
field = Rectangle((10, 10), 100, 40, facecolor='#27ae60', edgecolor='white', linewidth=2)
ax1.add_patch(field)

# Yard lines
for x in range(20, 110, 10):
    ax1.plot([x, x], [10, 50], 'white', linewidth=0.8, alpha=0.4)

# QB
ax1.scatter([50], [30], s=400, c='#e74c3c', marker='o',
           edgecolors='white', linewidths=3, zorder=10)
ax1.text(50, 24, 'QB', ha='center', fontsize=10, fontweight='bold', color='white')

# Ball trajectory
ball_x = np.linspace(50, 85, 20)
ball_y = 30 + (22 - 30) * (ball_x - 50) / 35
ax1.plot(ball_x, ball_y, 'yellow', linewidth=3, linestyle=':', alpha=0.9)
ax1.scatter([85], [22], s=300, c='gold', marker='*',
           edgecolors='black', linewidths=2, zorder=10)

# WR
ax1.scatter([85], [22], s=350, c='cyan', marker='o',
           edgecolors='white', linewidths=2, zorder=9)
ax1.text(85, 16, 'WR', ha='center', fontsize=10, fontweight='bold', color='white')

# Defenders
ax1.scatter([75, 82, 88], [15, 28, 35], s=320, c='orange', marker='s',
           edgecolors='white', linewidths=2, zorder=10)
ax1.text(75, 10, 'CB\nBland', ha='center', fontsize=9, fontweight='bold',
         color='white', bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8, pad=0.3))

ax1.set_xlim(0, 120)
ax1.set_ylim(0, 60)
ax1.set_aspect('equal')

# Description
ax1.text(0.5, 0.08, '10Hz tracking:\nx, y, speed, accel, heading',
         ha='center', va='center', fontsize=11,
         transform=ax1.transAxes,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='#3498db', linewidth=2))

# ========== STEP 2: PHYSICS MODEL ==========
ax2 = fig.add_axes([0.32, 0.6, 0.22, 0.28])
ax2.axis('off')

# Title
title2 = FancyBboxPatch((0.05, 0.88), 0.9, 0.10,
                        boxstyle="round,pad=0.02",
                        facecolor='#e74c3c', edgecolor='white',
                        linewidth=3, transform=ax2.transAxes)
ax2.add_patch(title2)
ax2.text(0.5, 0.93, 'Step 2: Reach Model', ha='center', va='center',
         fontsize=14, fontweight='bold', color='white', transform=ax2.transAxes)

# Defender with reach ellipse
defender_pos = (0.5, 0.5)
ax2.scatter([defender_pos[0]], [defender_pos[1]], s=500, c='orange', marker='s',
           edgecolors='white', linewidths=3, zorder=10, transform=ax2.transAxes)

# Reach ellipse (rotated)
ellipse = Ellipse(defender_pos, 0.35, 0.22, angle=30,
                 facecolor='orange', alpha=0.25, edgecolor='orange',
                 linewidth=3, linestyle='--', transform=ax2.transAxes)
ax2.add_patch(ellipse)

ax2.text(0.5, 0.3, 'Reachable Zone', ha='center', fontsize=11,
         fontweight='bold', color='#e74c3c', transform=ax2.transAxes)

# Equations box
eq_box = FancyBboxPatch((0.08, 0.05), 0.84, 0.18,
                        boxstyle="round,pad=0.02",
                        facecolor='white', edgecolor='#e74c3c',
                        linewidth=2, transform=ax2.transAxes)
ax2.add_patch(eq_box)

ax2.text(0.5, 0.16, 'Physics:', ha='center', fontsize=10,
         fontweight='bold', transform=ax2.transAxes)
ax2.text(0.5, 0.10, r'$d = vt + \frac{1}{2}at^2$',
         ha='center', fontsize=11, transform=ax2.transAxes)

# ========== STEP 3: DACS COMPUTATION ==========
ax3 = fig.add_axes([0.59, 0.6, 0.22, 0.28])
ax3.axis('off')

# Title
title3 = FancyBboxPatch((0.05, 0.88), 0.9, 0.10,
                        boxstyle="round,pad=0.02",
                        facecolor='#9b59b6', edgecolor='white',
                        linewidth=3, transform=ax3.transAxes)
ax3.add_patch(title3)
ax3.text(0.5, 0.93, 'Step 3: Compute DACS', ha='center', va='center',
         fontsize=14, fontweight='bold', color='white', transform=ax3.transAxes)

# Catch zone corridor
corridor = Rectangle((0.1, 0.35), 0.8, 0.45, facecolor='#ecf0f1',
                     edgecolor='#34495e', linewidth=3, transform=ax3.transAxes)
ax3.add_patch(corridor)

ax3.text(0.5, 0.74, 'Catch Zone', ha='center', fontsize=11,
         fontweight='bold', transform=ax3.transAxes)

# Sample points (some covered, some not)
np.random.seed(42)
for _ in range(40):
    x = 0.1 + np.random.uniform(0, 0.8)
    y = 0.35 + np.random.uniform(0, 0.45)
    covered = np.random.rand() < 0.35  # 35% covered
    color = '#e74c3c' if covered else '#bdc3c7'
    ax3.scatter(x, y, s=60, c=color, alpha=0.7, transform=ax3.transAxes)

# Legend
ax3.scatter([0.15], [0.28], s=60, c='#e74c3c', alpha=0.7, transform=ax3.transAxes)
ax3.text(0.18, 0.28, 'Covered', va='center', fontsize=9, transform=ax3.transAxes)
ax3.scatter([0.15], [0.22], s=60, c='#bdc3c7', alpha=0.7, transform=ax3.transAxes)
ax3.text(0.18, 0.22, 'Open', va='center', fontsize=9, transform=ax3.transAxes)

# Formula
formula_box = FancyBboxPatch((0.08, 0.05), 0.84, 0.12,
                             boxstyle="round,pad=0.02",
                             facecolor='white', edgecolor='#9b59b6',
                             linewidth=2, transform=ax3.transAxes)
ax3.add_patch(formula_box)
ax3.text(0.5, 0.11, 'DACS = (Covered Points / Total Points) × 100',
         ha='center', fontsize=10, fontweight='bold', transform=ax3.transAxes)

# ========== STEP 4: OUTCOME ==========
ax4 = fig.add_axes([0.86, 0.6, 0.12, 0.28])
ax4.axis('off')

# Title
title4 = FancyBboxPatch((0.05, 0.88), 0.9, 0.10,
                        boxstyle="round,pad=0.02",
                        facecolor='#27ae60', edgecolor='white',
                        linewidth=3, transform=ax4.transAxes)
ax4.add_patch(title4)
ax4.text(0.5, 0.93, 'Result', ha='center', va='center',
         fontsize=14, fontweight='bold', color='white', transform=ax4.transAxes)

# Gauge/meter showing DACS score
from matplotlib.patches import Wedge
# Create a semi-circle gauge
theta1, theta2 = 180, 0
wedge_bg = Wedge((0.5, 0.3), 0.25, theta1, theta2,
                 facecolor='#ecf0f1', edgecolor='#34495e',
                 linewidth=2, transform=ax4.transAxes)
ax4.add_patch(wedge_bg)

# Filled portion (showing 68%)
dacs_value = 68
theta_fill = 180 - (dacs_value / 100 * 180)
wedge_fill = Wedge((0.5, 0.3), 0.25, 180, theta_fill,
                   facecolor='#27ae60', edgecolor='white',
                   linewidth=1, transform=ax4.transAxes)
ax4.add_patch(wedge_fill)

# Value text
ax4.text(0.5, 0.35, '68%', ha='center', va='center',
         fontsize=24, fontweight='bold', color='#27ae60',
         transform=ax4.transAxes)

ax4.text(0.5, 0.15, 'High\nControl', ha='center', va='center',
         fontsize=11, fontweight='bold', color='#27ae60',
         transform=ax4.transAxes)

# ========== ARROWS BETWEEN STEPS ==========
arrow_props = dict(arrowstyle='->', mutation_scale=35,
                  linewidth=4, color='#34495e')

arr1 = FancyArrowPatch((0.275, 0.74), (0.315, 0.74),
                       **arrow_props, transform=fig.transFigure)
fig.patches.append(arr1)

arr2 = FancyArrowPatch((0.545, 0.74), (0.585, 0.74),
                       **arrow_props, transform=fig.transFigure)
fig.patches.append(arr2)

arr3 = FancyArrowPatch((0.815, 0.74), (0.855, 0.74),
                       **arrow_props, transform=fig.transFigure)
fig.patches.append(arr3)

# ========== EXAMPLE CASE STUDY (BOTTOM) ==========
ax_case = fig.add_axes([0.05, 0.08, 0.93, 0.45])
ax_case.axis('off')

# Title
case_title = FancyBboxPatch((0.02, 0.88), 0.96, 0.10,
                            boxstyle="round,pad=0.02",
                            facecolor='#34495e', edgecolor='white',
                            linewidth=3, transform=ax_case.transAxes)
ax_case.add_patch(case_title)
ax_case.text(0.5, 0.93, 'Example: Daron Bland (DAL) - Elite "Eraser" Performance',
             ha='center', va='center', fontsize=16, fontweight='bold',
             color='white', transform=ax_case.transAxes)

# Three panels: Before, During, After

# === PANEL 1: At Release ===
panel1_x = 0.08
panel_width = 0.26

p1_box = FancyBboxPatch((panel1_x, 0.12), panel_width, 0.70,
                        boxstyle="round,pad=0.02",
                        facecolor='white', edgecolor='#e74c3c',
                        linewidth=3, transform=ax_case.transAxes)
ax_case.add_patch(p1_box)

ax_case.text(panel1_x + panel_width/2, 0.75, 'Ball Release (t=0.0s)',
             ha='center', fontsize=12, fontweight='bold',
             transform=ax_case.transAxes, color='#e74c3c')
ax_case.text(panel1_x + panel_width/2, 0.68, 'DACS: 15%',
             ha='center', fontsize=14, fontweight='bold',
             transform=ax_case.transAxes, color='#e74c3c')

# Minifield
mini_field1 = Rectangle((panel1_x + 0.03, 0.30), panel_width - 0.06, 0.32,
                         facecolor='#27ae60', edgecolor='white',
                         linewidth=2, transform=ax_case.transAxes, alpha=0.3)
ax_case.add_patch(mini_field1)

# Players
ax_case.scatter([panel1_x + 0.08], [0.50], s=250, c='#e74c3c', marker='o',
               edgecolors='white', linewidths=2, zorder=10, transform=ax_case.transAxes)
ax_case.scatter([panel1_x + 0.24], [0.42], s=250, c='cyan', marker='o',
               edgecolors='white', linewidths=2, zorder=10, transform=ax_case.transAxes)
ax_case.scatter([panel1_x + 0.18], [0.36], s=230, c='orange', marker='s',
               edgecolors='white', linewidths=2, zorder=10, transform=ax_case.transAxes)

ax_case.text(panel1_x + panel_width/2, 0.20,
             'Bland 8 yards away\nBall just released',
             ha='center', fontsize=10, transform=ax_case.transAxes)

# === PANEL 2: During Flight ===
panel2_x = 0.38

p2_box = FancyBboxPatch((panel2_x, 0.12), panel_width, 0.70,
                        boxstyle="round,pad=0.02",
                        facecolor='white', edgecolor='#f39c12',
                        linewidth=3, transform=ax_case.transAxes)
ax_case.add_patch(p2_box)

ax_case.text(panel2_x + panel_width/2, 0.75, 'Ball in Flight (t=1.2s)',
             ha='center', fontsize=12, fontweight='bold',
             transform=ax_case.transAxes, color='#f39c12')
ax_case.text(panel2_x + panel_width/2, 0.68, 'DACS: 45%',
             ha='center', fontsize=14, fontweight='bold',
             transform=ax_case.transAxes, color='#f39c12')

# Minifield
mini_field2 = Rectangle((panel2_x + 0.03, 0.30), panel_width - 0.06, 0.32,
                         facecolor='#27ae60', edgecolor='white',
                         linewidth=2, transform=ax_case.transAxes, alpha=0.3)
ax_case.add_patch(mini_field2)

# Players - defender moving closer
ax_case.scatter([panel2_x + 0.20], [0.44], s=250, c='cyan', marker='o',
               edgecolors='white', linewidths=2, zorder=10, transform=ax_case.transAxes)
ax_case.scatter([panel2_x + 0.17], [0.42], s=230, c='orange', marker='s',
               edgecolors='white', linewidths=2, zorder=10, transform=ax_case.transAxes)

# Ball trajectory
ball_traj_x = np.linspace(panel2_x + 0.08, panel2_x + 0.20, 10)
ball_traj_y = np.linspace(0.50, 0.44, 10)
ax_case.plot(ball_traj_x, ball_traj_y, 'yellow', linewidth=2,
            linestyle='--', transform=ax_case.transAxes)

ax_case.text(panel2_x + panel_width/2, 0.20,
             'Bland closing gap\nOptimal pursuit angle',
             ha='center', fontsize=10, transform=ax_case.transAxes)

# === PANEL 3: At Arrival ===
panel3_x = 0.68

p3_box = FancyBboxPatch((panel3_x, 0.12), panel_width, 0.70,
                        boxstyle="round,pad=0.02",
                        facecolor='white', edgecolor='#27ae60',
                        linewidth=3, transform=ax_case.transAxes)
ax_case.add_patch(p3_box)

ax_case.text(panel3_x + panel_width/2, 0.75, 'Ball Arrival (t=2.2s)',
             ha='center', fontsize=12, fontweight='bold',
             transform=ax_case.transAxes, color='#27ae60')
ax_case.text(panel3_x + panel_width/2, 0.68, 'DACS: 72%',
             ha='center', fontsize=14, fontweight='bold',
             transform=ax_case.transAxes, color='#27ae60')

# Minifield
mini_field3 = Rectangle((panel3_x + 0.03, 0.30), panel_width - 0.06, 0.32,
                         facecolor='#27ae60', edgecolor='white',
                         linewidth=2, transform=ax_case.transAxes, alpha=0.3)
ax_case.add_patch(mini_field3)

# Players - defender at catch point
ax_case.scatter([panel3_x + 0.16], [0.44], s=250, c='cyan', marker='o',
               edgecolors='white', linewidths=2, zorder=9, transform=ax_case.transAxes)
ax_case.scatter([panel3_x + 0.16], [0.44], s=230, c='orange', marker='s',
               edgecolors='white', linewidths=2, zorder=10, transform=ax_case.transAxes)
ax_case.scatter([panel3_x + 0.16], [0.44], s=150, c='gold', marker='*',
               edgecolors='black', linewidths=2, zorder=11, transform=ax_case.transAxes)

ax_case.text(panel3_x + panel_width/2, 0.20,
             'Bland arrives at catch point\n= PASS INCOMPLETE',
             ha='center', fontsize=10, fontweight='bold',
             transform=ax_case.transAxes, color='#27ae60')

# Footer
fig.text(0.5, 0.01,
         'DACS measures pursuit PROCESS, not just outcome | Big Data Bowl 2026',
         ha='center', fontsize=11, style='italic', color='#7f8c8d')

plt.savefig('analytics/outputs/presentation/viz_dacs_process_diagram.png',
            dpi=300, bbox_inches='tight', facecolor='#f5f7fa')
print("[OK] Saved improved process diagram with Daron Bland example")
plt.close()
