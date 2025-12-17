"""
Create a visual process diagram showing how DACS works
Similar to last year's winner but for ball-in-air analysis
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, Rectangle
import numpy as np

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#f8f9fa')

# Title
fig.text(0.5, 0.96, 'DACS: Defensive Air Control Score - How It Works',
         ha='center', fontsize=22, fontweight='bold', color='#2c3e50')

# ============= STEP 1: Input Data =============
ax1 = fig.add_axes([0.05, 0.6, 0.25, 0.3])
ax1.axis('off')

# Football field representation
field = Rectangle((0, 0), 120, 53.3, facecolor='#2d5016', edgecolor='white', linewidth=2)
ax1.add_patch(field)

# Yard lines
for x in range(10, 120, 10):
    ax1.plot([x, x], [0, 53.3], 'white', linewidth=1, alpha=0.3)

# Players
ax1.scatter([60], [26.65], s=300, c='red', marker='o', edgecolors='white', linewidths=2, label='QB')
ax1.scatter([80], [20], s=300, c='blue', marker='o', edgecolors='white', linewidths=2, label='WR')
ax1.scatter([75, 78, 82], [15, 25, 30], s=250, c='orange', marker='s',
           edgecolors='white', linewidths=2, label='Defenders')

# Ball trajectory
ball_path = np.linspace(60, 80, 20)
ball_y = 26.65 + (20 - 26.65) * (ball_path - 60) / 20
ax1.plot(ball_path, ball_y, 'yellow', linewidth=3, linestyle='--', alpha=0.8)
ax1.scatter([80], [20], s=200, c='yellow', marker='*', edgecolors='black', linewidths=2, zorder=10)

ax1.set_xlim(50, 90)
ax1.set_ylim(10, 43.3)
ax1.set_aspect('equal')

# Label
ax1.text(70, 5, 'Step 1: Ball Release', ha='center', fontsize=14,
         fontweight='bold', color='white',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#34495e', alpha=0.9))
ax1.text(70, 2, 'Track players & ball from\nthrow to catch/incompletion',
         ha='center', fontsize=10, color='white')

# ============= STEP 2: Physics Model =============
ax2 = fig.add_axes([0.35, 0.6, 0.25, 0.3])
ax2.axis('off')

# Title box
title_box = FancyBboxPatch((0.05, 0.85), 0.9, 0.12, boxstyle="round,pad=0.02",
                           facecolor='#e74c3c', edgecolor='white', linewidth=2)
ax2.add_patch(title_box)
ax2.text(0.5, 0.91, 'Step 2: Residual Reach Model', ha='center', va='center',
         fontsize=14, fontweight='bold', color='white', transform=ax2.transAxes)

# Physics equations
ax2.text(0.5, 0.70, r'Physics Baseline:', ha='center', fontsize=12,
         fontweight='bold', transform=ax2.transAxes)
ax2.text(0.5, 0.60, r'$d_\parallel(t) = s \cdot t + \frac{1}{2}a_{max} \cdot t^2$',
         ha='center', fontsize=11, transform=ax2.transAxes)
ax2.text(0.5, 0.52, '(Forward reach with max acceleration)',
         ha='center', fontsize=9, style='italic', transform=ax2.transAxes)

# ML correction
ax2.text(0.5, 0.38, r'+ ML Residual Correction:', ha='center', fontsize=12,
         fontweight='bold', transform=ax2.transAxes)
ax2.text(0.5, 0.28, r'$(\lambda_\parallel, \lambda_\perp) = MLP(features)$',
         ha='center', fontsize=11, transform=ax2.transAxes)
ax2.text(0.5, 0.18, 'Features: speed, heading, ball angle,\nrole, time remaining',
         ha='center', fontsize=9, style='italic', transform=ax2.transAxes)

# Ellipse visualization
ellipse = Ellipse((0.5, 0.05), 0.3, 0.15, angle=30,
                  facecolor='orange', alpha=0.4, edgecolor='orange', linewidth=2,
                  transform=ax2.transAxes)
ax2.add_patch(ellipse)
ax2.text(0.5, 0.05, 'Reachable\nZone', ha='center', va='center',
         fontsize=9, fontweight='bold', transform=ax2.transAxes)

# ============= STEP 3: DACS Computation =============
ax3 = fig.add_axes([0.65, 0.6, 0.30, 0.3])
ax3.axis('off')

# Title
title_box3 = FancyBboxPatch((0.05, 0.85), 0.9, 0.12, boxstyle="round,pad=0.02",
                            facecolor='#3498db', edgecolor='white', linewidth=2)
ax3.add_patch(title_box3)
ax3.text(0.5, 0.91, 'Step 3: Compute DACS', ha='center', va='center',
         fontsize=14, fontweight='bold', color='white', transform=ax3.transAxes)

# Corridor visualization
corridor_rect = Rectangle((0.15, 0.45), 0.7, 0.3, facecolor='#ecf0f1',
                          edgecolor='#34495e', linewidth=2, transform=ax3.transAxes)
ax3.add_patch(corridor_rect)
ax3.text(0.5, 0.72, 'Catch Zone Corridor', ha='center', fontsize=11,
         fontweight='bold', transform=ax3.transAxes)

# Sample points
np.random.seed(42)
for _ in range(30):
    x = 0.15 + np.random.uniform(0, 0.7)
    y = 0.45 + np.random.uniform(0, 0.3)
    color = 'red' if np.random.rand() < 0.3 else 'lightgray'
    ax3.scatter(x, y, s=40, c=color, alpha=0.6, transform=ax3.transAxes)

# Coverage ellipses
ellipse1 = Ellipse((0.35, 0.6), 0.2, 0.15, angle=20,
                   facecolor='orange', alpha=0.3, edgecolor='orange', linewidth=2,
                   transform=ax3.transAxes)
ax3.add_patch(ellipse1)

# Formula
ax3.text(0.5, 0.30, r'DACS = $\frac{\text{Covered Points}}{\text{Total Points}} \times 100$',
         ha='center', fontsize=12, fontweight='bold', transform=ax3.transAxes,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#34495e'))

# Example output
ax3.text(0.5, 0.15, 'Example: 35% DACS', ha='center', fontsize=13,
         fontweight='bold', color='#e74c3c', transform=ax3.transAxes)
ax3.text(0.5, 0.08, 'Defense controls 35% of catch zone', ha='center',
         fontsize=10, style='italic', transform=ax3.transAxes)

# ============= STEP 4: Applications =============
ax4 = fig.add_axes([0.05, 0.15, 0.90, 0.35])
ax4.axis('off')

# Title
ax4.text(0.5, 0.95, 'Step 4: Applications & Insights', ha='center', fontsize=16,
         fontweight='bold', transform=ax4.transAxes, color='#2c3e50')

# Three application boxes
app_width = 0.28
app_height = 0.75
app_y = 0.1

# App 1: Player Evaluation
app1_box = FancyBboxPatch((0.05, app_y), app_width, app_height,
                          boxstyle="round,pad=0.02", facecolor='#2ecc71',
                          edgecolor='white', linewidth=2, alpha=0.9, transform=ax4.transAxes)
ax4.add_patch(app1_box)
ax4.text(0.05 + app_width/2, app_y + app_height - 0.08, 'Player Evaluation',
         ha='center', fontsize=13, fontweight='bold', color='white', transform=ax4.transAxes)
ax4.text(0.05 + app_width/2, app_y + app_height/2 + 0.05,
         'Identify "Erasers":\n\n• Kyle Hamilton\n• Best pursuit efficiency\n• High EPA prevented\n\nCredit for PROCESS\nnot just outcome',
         ha='center', va='center', fontsize=10, color='white', transform=ax4.transAxes)

# App 2: Scheme Analysis
app2_box = FancyBboxPatch((0.36, app_y), app_width, app_height,
                          boxstyle="round,pad=0.02", facecolor='#9b59b6',
                          edgecolor='white', linewidth=2, alpha=0.9, transform=ax4.transAxes)
ax4.add_patch(app2_box)
ax4.text(0.36 + app_width/2, app_y + app_height - 0.08, 'Scheme Diagnosis',
         ha='center', fontsize=13, fontweight='bold', color='white', transform=ax4.transAxes)
ax4.text(0.36 + app_width/2, app_y + app_height/2 + 0.05,
         'Coverage Analysis:\n\n• Zone vs Man DACS\n• Coverage busts\n• Pursuit coordination\n\nFind structural\nweaknesses early',
         ha='center', va='center', fontsize=10, color='white', transform=ax4.transAxes)

# App 3: Game Planning
app3_box = FancyBboxPatch((0.67, app_y), app_width, app_height,
                          boxstyle="round,pad=0.02", facecolor='#e67e22',
                          edgecolor='white', linewidth=2, alpha=0.9, transform=ax4.transAxes)
ax4.add_patch(app3_box)
ax4.text(0.67 + app_width/2, app_y + app_height - 0.08, 'Game Planning',
         ha='center', fontsize=13, fontweight='bold', color='white', transform=ax4.transAxes)
ax4.text(0.67 + app_width/2, app_y + app_height/2 + 0.05,
         'Weekly Prep:\n\n• Field vulnerability maps\n• Opponent DACS trends\n• Attack low-DACS zones\n\nScript plays to\nexploit weaknesses',
         ha='center', va='center', fontsize=10, color='white', transform=ax4.transAxes)

# Arrows connecting steps
arrow1 = FancyArrowPatch((0.3, 0.75), (0.35, 0.75),
                        arrowstyle='->', mutation_scale=30, linewidth=3,
                        color='#34495e', transform=fig.transFigure)
fig.patches.append(arrow1)

arrow2 = FancyArrowPatch((0.6, 0.75), (0.65, 0.75),
                        arrowstyle='->', mutation_scale=30, linewidth=3,
                        color='#34495e', transform=fig.transFigure)
fig.patches.append(arrow2)

arrow3 = FancyArrowPatch((0.5, 0.58), (0.5, 0.52),
                        arrowstyle='->', mutation_scale=30, linewidth=3,
                        color='#34495e', transform=fig.transFigure)
fig.patches.append(arrow3)

# Footer
fig.text(0.5, 0.02, 'Big Data Bowl 2026 - Analytics Track | Post-Throw Pursuit Analysis',
         ha='center', fontsize=11, style='italic', color='#7f8c8d')

plt.savefig('analytics/outputs/presentation/viz_dacs_process_diagram.png',
            dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
print("[OK] Saved DACS process diagram to analytics/outputs/presentation/viz_dacs_process_diagram.png")
plt.close()
