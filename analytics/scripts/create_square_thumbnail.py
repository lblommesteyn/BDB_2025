"""
Create a square thumbnail/cover image for the Kaggle submission
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Wedge
import numpy as np
from pathlib import Path

def create_thumbnail(output_dir='analytics/outputs/presentation'):
    """Create a professional square thumbnail for the submission"""

    # Create square figure
    fig = plt.figure(figsize=(10, 10), facecolor='#1a1a2e')

    # Main content area
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Title section (top)
    ax.text(50, 85, 'Defensive Air Control Score',
            ha='center', va='center', fontsize=32, fontweight='bold',
            color='white', family='sans-serif')

    ax.text(50, 78, 'DACS',
            ha='center', va='center', fontsize=48, fontweight='bold',
            color='#3498db', family='sans-serif')

    # Subtitle
    ax.text(50, 71, 'Quantifying Space Ownership During Ball Flight',
            ha='center', va='center', fontsize=14, style='italic',
            color='#ecf0f1', family='sans-serif')

    # Divider line
    ax.plot([15, 85], [68, 68], 'white', linewidth=2, alpha=0.5)

    # Center visualization - DACS meter
    center_x, center_y = 50, 45

    # Background semi-circle
    wedge_bg = Wedge((center_x, center_y), 18, 180, 0,
                     facecolor='#34495e', edgecolor='white',
                     linewidth=3, transform=ax.transData)
    ax.add_patch(wedge_bg)

    # DACS fill (showing 72% like Bland example)
    dacs_value = 72
    theta_fill = 180 - (dacs_value / 100 * 180)
    wedge_fill = Wedge((center_x, center_y), 18, 180, theta_fill,
                       facecolor='#27ae60', edgecolor='white',
                       linewidth=2, transform=ax.transData)
    ax.add_patch(wedge_fill)

    # DACS percentage in center
    ax.text(center_x, center_y + 2, f'{dacs_value}%',
            ha='center', va='center', fontsize=38, fontweight='bold',
            color='#27ae60', family='monospace')

    # Labels on sides of meter
    ax.text(center_x - 20, center_y, '0%',
            ha='right', va='center', fontsize=12, color='white', fontweight='bold')
    ax.text(center_x + 20, center_y, '100%',
            ha='left', va='center', fontsize=12, color='white', fontweight='bold')

    # Bottom text - Key stat
    ax.text(50, 25, '9,852 Plays Analyzed',
            ha='center', va='center', fontsize=18, fontweight='bold',
            color='#3498db', family='sans-serif')

    ax.text(50, 20, '2023 NFL Season | Full Coverage',
            ha='center', va='center', fontsize=12,
            color='#ecf0f1', family='sans-serif')

    # Bottom branding
    ax.text(50, 12, 'Identifying the "Erasers"',
            ha='center', va='center', fontsize=16, style='italic',
            color='#e74c3c', fontweight='bold', family='sans-serif')

    # Conference/track
    ax.text(50, 5, 'Big Data Bowl 2026 | Analytics Track',
            ha='center', va='center', fontsize=11,
            color='#95a5a6', family='sans-serif')

    # Add subtle field lines in background
    for i in range(5):
        y = 15 + i * 10
        ax.plot([10, 90], [y, y], color='white', linewidth=0.5, alpha=0.1)

    plt.tight_layout(pad=0)

    # Save
    output_path = Path(output_dir) / 'thumbnail_square.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    print(f"[OK] Saved square thumbnail to {output_path}")
    plt.close()

    # Also create a simpler version
    create_simple_thumbnail(output_dir)

def create_simple_thumbnail(output_dir='analytics/outputs/presentation'):
    """Create a simpler, cleaner square thumbnail"""

    fig = plt.figure(figsize=(10, 10), facecolor='#2c3e50')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Large DACS text
    ax.text(50, 60, 'DACS',
            ha='center', va='center', fontsize=80, fontweight='bold',
            color='#3498db', family='sans-serif')

    # Tagline
    ax.text(50, 45, 'Defensive Air Control Score',
            ha='center', va='center', fontsize=16,
            color='white', family='sans-serif')

    # Key visual - mini meter
    center_x, center_y = 50, 28

    # Semi-circle
    wedge_bg = Wedge((center_x, center_y), 8, 180, 0,
                     facecolor='#34495e', edgecolor='white',
                     linewidth=2, transform=ax.transData)
    ax.add_patch(wedge_bg)

    # Fill at 72%
    dacs_value = 72
    theta_fill = 180 - (dacs_value / 100 * 180)
    wedge_fill = Wedge((center_x, center_y), 8, 180, theta_fill,
                       facecolor='#27ae60', edgecolor='white',
                       linewidth=1, transform=ax.transData)
    ax.add_patch(wedge_fill)

    ax.text(center_x, center_y + 1, '72%',
            ha='center', va='center', fontsize=14, fontweight='bold',
            color='#27ae60')

    # Bottom stats
    ax.text(50, 15, '9,852 Plays | 2023 NFL Season',
            ha='center', va='center', fontsize=13, fontweight='bold',
            color='#ecf0f1', family='sans-serif')

    ax.text(50, 8, 'Big Data Bowl 2026',
            ha='center', va='center', fontsize=12,
            color='#95a5a6', family='sans-serif')

    plt.tight_layout(pad=0)

    output_path = Path(output_dir) / 'thumbnail_square_simple.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='#2c3e50', edgecolor='none')
    print(f"[OK] Saved simple square thumbnail to {output_path}")
    plt.close()

if __name__ == '__main__':
    print("="*60)
    print("Creating Square Thumbnails")
    print("="*60)

    create_thumbnail()

    print(f"\n{'='*60}")
    print("Thumbnails created!")
    print("="*60)
