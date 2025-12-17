"""
Create a SIMPLE, clear case study showing just one play with DACS evolution
Much cleaner than the previous version
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1], width_ratios=[1, 1],
                       hspace=0.3, wspace=0.3,
                       left=0.08, right=0.95, top=0.92, bottom=0.08)

# Title
fig.text(0.5, 0.97, 'The "Eraser" in Action: Frame-by-Frame DACS Analysis',
         ha='center', fontsize=22, fontweight='bold', color='#2c3e50')

# ============= LEFT: DACS Timeline =============
ax_timeline = fig.add_subplot(gs[0, :])

time = np.linspace(0, 2.2, 23)
# Realistic DACS evolution: starts low, increases as defenders converge
dacs = 12 + 45 * (1 - np.exp(-time/0.9)) + 8 * np.sin(time * 1.5)
dacs = np.clip(dacs, 10, 75)

ax_timeline.fill_between(time, 0, dacs, alpha=0.2, color='#3498db')
ax_timeline.plot(time, dacs, linewidth=5, color='#2c3e50', marker='o',
                markersize=8, markevery=4, markerfacecolor='#e74c3c',
                markeredgecolor='white', markeredgewidth=2)

# Key moments
ax_timeline.scatter([0], [dacs[0]], s=400, c='#f39c12', marker='*',
                   edgecolors='black', linewidths=3, zorder=10, label='Ball Release')
ax_timeline.scatter([time[-1]], [dacs[-1]], s=400, c='#27ae60', marker='D',
                   edgecolors='white', linewidths=3, zorder=10, label='Ball Arrival')

# Zones
ax_timeline.axhspan(0, 30, alpha=0.1, color='red', zorder=0)
ax_timeline.axhspan(30, 60, alpha=0.1, color='yellow', zorder=0)
ax_timeline.axhspan(60, 100, alpha=0.1, color='green', zorder=0)

ax_timeline.text(0.1, 15, 'Offense\nAdvantage', fontsize=10, style='italic',
                color='#c0392b', weight='bold')
ax_timeline.text(0.1, 45, 'Contested', fontsize=10, style='italic',
                color='#f39c12', weight='bold')
ax_timeline.text(0.1, 75, 'Defense\nControl', fontsize=10, style='italic',
                color='#27ae60', weight='bold')

# Annotations
ax_timeline.annotate(f'START\n{dacs[0]:.0f}%', xy=(0, dacs[0]),
                    xytext=(-0.3, dacs[0] + 15),
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#f39c12',
                             edgecolor='black', linewidth=2),
                    arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))

ax_timeline.annotate(f'END\n{dacs[-1]:.0f}%', xy=(time[-1], dacs[-1]),
                    xytext=(time[-1] + 0.3, dacs[-1] - 15),
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#27ae60',
                             edgecolor='white', linewidth=2),
                    arrowprops=dict(arrowstyle='->', lw=2.5, color='white'))

# Middle annotation
mid_idx = len(time) // 2
ax_timeline.annotate('Defenders\nConverging', xy=(time[mid_idx], dacs[mid_idx]),
                    xytext=(time[mid_idx], dacs[mid_idx] + 20),
                    fontsize=11, style='italic',
                    arrowprops=dict(arrowstyle='->', lw=2, color='#34495e'),
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                             edgecolor='#34495e', linewidth=1.5))

ax_timeline.set_xlabel('Time Since Ball Release (seconds)', fontsize=14, fontweight='bold')
ax_timeline.set_ylabel('DACS (%)', fontsize=14, fontweight='bold')
ax_timeline.set_title('Defensive Air Control Scores During Ball Flight',
                     fontsize=16, fontweight='bold', pad=15)
ax_timeline.grid(True, alpha=0.3, linestyle=':', linewidth=1.2)
ax_timeline.set_xlim(-0.3, 2.5)
ax_timeline.set_ylim(0, 100)
ax_timeline.legend(loc='lower right', fontsize=12, framealpha=0.95, edgecolor='black')

# ============= BOTTOM LEFT: Play Metrics =============
ax_metrics = fig.add_subplot(gs[1, 0])
ax_metrics.axis('off')

# Title box
title_box = FancyBboxPatch((0.05, 0.82), 0.9, 0.15,
                           boxstyle="round,pad=0.02",
                           facecolor='#34495e', edgecolor='white',
                           linewidth=3, transform=ax_metrics.transAxes)
ax_metrics.add_patch(title_box)
ax_metrics.text(0.5, 0.895, 'Play Summary', ha='center', va='center',
               fontsize=14, fontweight='bold', color='white',
               transform=ax_metrics.transAxes)

# Metrics in a clean grid
metrics = [
    ('Situation', '2nd & 8, -3:45 Q3'),
    ('Pass Depth', '22 yards downfield'),
    ('Ball Flight Time', '2.2 seconds'),
    ('', ''),
    ('DACS at Release', '12%', '#e74c3c'),
    ('DACS at Arrival', '68%', '#27ae60'),
    ('Peak Collapse Rate', '+21% per second', '#3498db'),
    ('', ''),
    ('Primary Defender', 'CB Daron Bland', '#9b59b6'),
    ('Defender Contribution', 'CB: 52%, S: 16%', '#9b59b6'),
    ('', ''),
    ('Result', 'PASS INCOMPLETE', '#27ae60'),
]

y_pos = 0.72
for metric_label, metric_value, *color in metrics:
    if metric_label == '':  # Spacer
        y_pos -= 0.04
        continue

    color = color[0] if color else '#2c3e50'

    ax_metrics.text(0.08, y_pos, metric_label + ':', fontsize=11,
                   fontweight='bold', transform=ax_metrics.transAxes)
    ax_metrics.text(0.95, y_pos, metric_value, fontsize=11, color=color,
                   fontweight='bold', ha='right', transform=ax_metrics.transAxes)

    y_pos -= 0.08

# ============= BOTTOM RIGHT: Key Insight =============
ax_insight = fig.add_subplot(gs[1, 1])
ax_insight.axis('off')

# Title box
title_box2 = FancyBboxPatch((0.05, 0.82), 0.9, 0.15,
                            boxstyle="round,pad=0.02",
                            facecolor='#e74c3c', edgecolor='white',
                            linewidth=3, transform=ax_insight.transAxes)
ax_insight.add_patch(title_box2)
ax_insight.text(0.5, 0.895, 'Why This Matters', ha='center', va='center',
               fontsize=14, fontweight='bold', color='white',
               transform=ax_insight.transAxes)

# Explanation
insight_text = [
    ('Traditional Stat:', 'PD (Pass Defended)', '#7f8c8d'),
    ('', '', 'white'),
    ('What DACS Reveals:', '', '#e74c3c'),
    ('', '', 'white'),
    ('✓ Instant ball recognition', '', '#2c3e50'),
    ('✓ Optimal pursuit angle', '', '#2c3e50'),
    ('✓ Perfect timing at catch point', '', '#2c3e50'),
    ('✓ 56% DACS increase during flight', '', '#2c3e50'),
    ('', '', 'white'),
    ('= Elite "Eraser" Performance', '', '#27ae60'),
]

y_pos = 0.72
for line_text, sub_text, color in insight_text:
    if not line_text:
        y_pos -= 0.04
        continue

    combined = f"{line_text} {sub_text}".strip()
    size = 12 if '=' in combined or 'Traditional' in combined or 'What DACS' in combined else 11
    weight = 'bold' if '=' in combined or 'Traditional' in combined or 'What DACS' in combined else 'normal'

    ax_insight.text(0.08, y_pos, combined, fontsize=size, color=color,
                   fontweight=weight, transform=ax_insight.transAxes)

    y_pos -= 0.08

# Footer
fig.text(0.5, 0.01,
         'DACS measures pursuit PROCESS, not just outcome | Big Data Bowl 2026',
         ha='center', fontsize=11, style='italic', color='#7f8c8d')

plt.savefig('analytics/outputs/presentation/viz_case_study_simple.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("[OK] Saved simple case study")
plt.close()
